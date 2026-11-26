from datetime import date, timedelta

import numpy as np
import pytest
from conftest import mini_connection, rendered

from weathernorm.config import PipelineParams

D = date


@pytest.fixture(scope="module")
def mini():
    """Ten days of weather and one accepted period over days 3 to 8 (start exclusive)."""
    temps = [70.0, 66.0, 60.0, 50.0, 40.0, 55.0, 64.0, 65.0, 30.0, 80.0]
    weather = [(D(2024, 1, 1) + timedelta(days=i), t) for i, t in enumerate(temps)]
    # Jan 1 is day 0; the period (Jan 3, Jan 8] covers Jan 4 to Jan 8 = 50, 40, 55, 64, 65 F.
    reads = [(1, D(2024, 1, 3), 100.0, False), (1, D(2024, 1, 8), 150.0, False)]
    homes = [(1, "participant", D(2030, 1, 15))]
    con = mini_connection(weather, homes, reads)
    params = PipelineParams(min_days=1, base_min=60, base_max=66)
    con.execute(rendered("010_billing_periods.sql", params))
    con.execute(rendered("020_degree_days.sql", params))
    return con


def test_candidate_bases_are_every_degree_in_range(mini):
    assert [
        r[0] for r in mini.execute("SELECT base FROM candidate_bases ORDER BY 1").fetchall()
    ] == [60, 61, 62, 63, 64, 65, 66]


def test_daily_hdd_is_zero_above_the_base(mini):
    hdd = dict(
        mini.execute("SELECT date, hdd FROM daily_hdd WHERE base = 65 ORDER BY date").fetchall()
    )
    assert hdd[D(2024, 1, 1)] == 0.0 and hdd[D(2024, 1, 4)] == 15.0 and hdd[D(2024, 1, 10)] == 0.0


def test_period_hdd_sums_days_after_start_through_end(mini):
    # Base 65 over 50, 40, 55, 64, 65 F: 15 + 25 + 10 + 1 + 0 = 51; base 60 gives 10 + 20 + 5 = 35.
    got = dict(mini.execute("SELECT base, hdd FROM period_hdd ORDER BY base").fetchall())
    assert got[65] == 51.0
    assert got[60] == 35.0
    assert got[66] == 56.0  # 16 + 26 + 11 + 2 + 1
    assert {w for (w,) in mini.execute("SELECT weather_days FROM period_hdd").fetchall()} == {5}


def test_normal_year_is_the_mean_by_calendar_day_without_29_february():
    start = D(2019, 1, 1)  # 2019 and 2020: 2020 is a leap year
    days = [start + timedelta(days=i) for i in range(366 + 365)]
    rng = np.random.default_rng(3)
    temps = np.round(rng.uniform(20, 80, len(days)), 1)
    con = mini_connection(list(zip(days, temps.tolist(), strict=True)), [], [])
    params = PipelineParams(base_min=60, base_max=60)
    con.execute(
        "CREATE TABLE accepted_periods (home_id INT, period_seq INT, start_date DATE, "
        "end_date DATE, phase VARCHAR)"
    )
    con.execute(rendered("020_degree_days.sql", params))
    normal = con.execute("SELECT COUNT(*), SUM(hdd) FROM normal_daily_hdd").fetchone()
    assert normal[0] == 365  # no 29 February row
    by_key: dict[tuple[int, int], list[float]] = {}
    for d, t in zip(days, temps, strict=True):
        if (d.month, d.day) != (2, 29):
            by_key.setdefault((d.month, d.day), []).append(max(60.0 - t, 0.0))
    expected = sum(np.mean(v) for v in by_key.values())
    assert normal[1] == pytest.approx(expected, rel=1e-12)
    assert con.execute("SELECT hdd, days FROM normal_annual_hdd").fetchone() == (
        pytest.approx(expected, rel=1e-12),
        365,
    )
