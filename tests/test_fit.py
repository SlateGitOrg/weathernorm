from datetime import date, timedelta

import duckdb
import numpy as np
import pytest
from conftest import mini_connection, rendered

from weathernorm import check
from weathernorm.config import PipelineParams

BASE_TRUE, ALPHA, BETA_PRE, BETA_POST = 60, 2.0, 0.20, 0.10


@pytest.fixture(scope="module")
def exact():
    """One home generated without noise: therms = alpha + beta * HDD60, beta halves at retrofit."""
    start = date(2022, 1, 1)
    n_days = 730
    doy = np.arange(n_days)
    temps = np.round(50 - 22 * np.cos(2 * np.pi * doy / 365) + 6 * np.sin(doy * 1.7), 1)
    weather = [(start + timedelta(days=int(i)), float(t)) for i, t in enumerate(temps)]
    event = date(2022, 12, 15)
    daily = np.zeros(n_days)
    for i in range(1, n_days):  # day i is the date start + i; day 0 is never inside a period
        beta = BETA_PRE if start + timedelta(days=i) < event else BETA_POST
        daily[i] = ALPHA + beta * max(BASE_TRUE - temps[i], 0.0)
    reads = []
    for k in range(27):
        meter_k = 5000.0 + daily[1 : 28 * k + 1].sum()
        reads.append((1, start + timedelta(days=28 * k), meter_k, False))
    con = mini_connection(weather, [(1, "participant", event)], reads)
    params = PipelineParams(min_days=20, min_periods=9, base_min=55, base_max=65)
    for name in ("010_billing_periods.sql", "020_degree_days.sql", "030_fit.sql", "040_nac.sql"):
        con.execute(rendered(name, params))
    return con, temps


def test_noise_free_home_recovers_base_slope_and_intercept(exact):
    con, _ = exact
    rows = con.execute(
        "SELECT phase, base, slope, intercept, r2, n, fail_reason "
        "FROM home_fits ORDER BY phase DESC"
    ).fetchall()
    (pre, post) = rows
    assert pre[0] == "pre" and post[0] == "post"
    assert pre[1] == post[1] == BASE_TRUE
    assert pre[2] == pytest.approx(BETA_PRE, rel=1e-9) and post[2] == pytest.approx(
        BETA_POST, rel=1e-9
    )
    assert pre[3] == pytest.approx(ALPHA, rel=1e-9) and post[3] == pytest.approx(ALPHA, rel=1e-9)
    assert pre[4] == pytest.approx(1.0) and post[4] == pytest.approx(1.0)
    assert (pre[5], post[5]) == (11, 13) and pre[6] is None and post[6] is None


def test_r2_peaks_at_the_true_base_and_falls_either_side(exact):
    con, _ = exact
    curve = check.r2_curve(con, 1, "pre").set_index("base")["r2"]
    assert curve.idxmax() == BASE_TRUE
    assert curve.loc[55] < curve.loc[58] < curve.loc[60] > curve.loc[62] > curve.loc[65]


def test_nac_matches_hand_computed_normal_year(exact):
    con, temps = exact
    # Two years of weather (730 days): the normal day is the mean of the two years' HDD60.
    hdd = np.maximum(BASE_TRUE - temps, 0.0)
    normal = float(((hdd[:365] + hdd[365:730]) / 2).sum())
    nac = dict(con.execute("SELECT phase, nac FROM nac").fetchall())
    assert nac["pre"] == pytest.approx(365 * ALPHA + BETA_PRE * normal, rel=1e-9)
    assert nac["post"] == pytest.approx(365 * ALPHA + BETA_POST * normal, rel=1e-9)


def test_sql_regression_equals_numpy_lstsq_for_50_random_triples(con):
    grid = con.execute("SELECT home_id, phase, base FROM fit_grid WHERE n >= 5").df()
    rng = np.random.default_rng(2024)
    picks = rng.choice(len(grid), 50, replace=False)
    worst = 0.0
    for i in picks:
        home, phase, base = grid.iloc[i]
        sql = check.sql_fit(con, int(home), phase, int(base))
        ref = check.reference_fit(con, int(home), phase, int(base))
        assert sql.n == ref.n
        worst = max(worst, check.max_relative_difference(sql, ref))
    assert worst < 1e-9


def test_lstsq_reference_on_a_known_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    fit = check.lstsq_fit(x, 1.5 + 0.5 * x)
    assert (fit.slope, fit.intercept, fit.r2, fit.n) == (
        pytest.approx(0.5),
        pytest.approx(1.5),
        pytest.approx(1.0),
        4,
    )


def _fit_only(periods, hdd_rows):
    """Run 030_fit.sql alone on crafted accepted_periods and period_hdd tables."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE homes (home_id INTEGER, grp VARCHAR, event_date DATE)")
    con.execute("INSERT INTO homes VALUES (1, 'participant', DATE '2024-06-01')")
    con.execute(
        "CREATE TABLE accepted_periods (home_id INTEGER, period_seq INTEGER, phase VARCHAR, "
        "therms DOUBLE, days INTEGER)"
    )
    con.execute(
        "CREATE TABLE period_hdd (home_id INTEGER, period_seq INTEGER, base INTEGER, hdd DOUBLE)"
    )
    con.executemany("INSERT INTO accepted_periods VALUES (?, ?, ?, ?, ?)", periods)
    con.executemany("INSERT INTO period_hdd VALUES (?, ?, ?, ?)", hdd_rows)
    con.execute(rendered("030_fit.sql"))
    return con


def _periods(n, phase="pre"):
    # y = therms/day = 30/30 .. ; x = hdd/30 grows with the period number
    return [(1, i, phase, 30.0 + 3.0 * i, 30) for i in range(1, n + 1)]


def test_equal_r2_goes_to_the_lower_base():
    periods = _periods(10)
    hdd = [(1, i, b, 30.0 * i) for i in range(1, 11) for b in (60, 61, 62)]
    con = _fit_only(periods, hdd)
    row = con.execute("SELECT base, r2 FROM home_fits WHERE phase = 'pre'").fetchone()
    assert row[0] == 60 and row[1] == pytest.approx(1.0)


def test_fewer_than_nine_periods_fails_with_insufficient_periods():
    con = _fit_only(_periods(8), [(1, i, 60, 30.0 * i) for i in range(1, 9)])
    row = con.execute("SELECT n, fail_reason FROM home_fits WHERE phase = 'pre'").fetchone()
    assert row == (8, "insufficient_periods")


def test_exactly_nine_periods_is_enough():
    con = _fit_only(_periods(9), [(1, i, 60, 30.0 * i) for i in range(1, 10)])
    assert con.execute("SELECT fail_reason FROM home_fits WHERE phase = 'pre'").fetchone() == (
        None,
    )


def test_non_positive_slope_fails_with_no_heating_signal():
    periods = [(1, i, "pre", 100.0 - 3.0 * i, 30) for i in range(1, 11)]
    con = _fit_only(periods, [(1, i, 60, 30.0 * i) for i in range(1, 11)])
    row = con.execute("SELECT slope, fail_reason FROM home_fits WHERE phase = 'pre'").fetchone()
    assert row[0] < 0 and row[1] == "no_heating_signal"


def test_a_phase_with_no_periods_is_reported_as_insufficient():
    con = _fit_only(_periods(10), [(1, i, 60, 30.0 * i) for i in range(1, 11)])
    row = con.execute("SELECT n, base, fail_reason FROM home_fits WHERE phase = 'post'").fetchone()
    assert row == (0, None, "insufficient_periods")


def test_chosen_base_is_within_two_degrees_for_90_percent_of_participants(con, synth_data):
    result = check.base_recovery(con, synth_data.truth_homes)
    assert result["n"] > 250
    assert result["share_within"] >= 0.90
    assert abs(result["bias"]) < 0.5


def test_failed_fits_never_reach_nac(con):
    assert con.execute(
        "SELECT COUNT(*) FROM nac n JOIN home_fits f USING (home_id, phase) "
        "WHERE f.fail_reason IS NOT NULL"
    ).fetchone() == (0,)
