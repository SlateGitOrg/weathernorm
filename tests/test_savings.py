import duckdb
import pytest
from conftest import rendered

from weathernorm import report
from weathernorm.bootstrap import bootstrap_did
from weathernorm.config import PipelineParams


@pytest.fixture(scope="module")
def crafted():
    """Hand-made NAC and period tables for 050_savings.sql with known answers."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE nac (home_id INT, grp VARCHAR, phase VARCHAR, base INT, nac DOUBLE)")
    nac = [
        (1, "participant", 100.0, 80.0),
        (2, "participant", 200.0, 150.0),
        (3, "comparison", 100.0, 98.0),
        (4, "comparison", 100.0, 102.0),
    ]
    for home, grp, pre, post in nac:
        con.execute(
            "INSERT INTO nac VALUES (?, ?, 'pre', 60, ?), (?, ?, 'post', 60, ?)",
            [home, grp, pre, home, grp, post],
        )
    con.execute(
        "CREATE TABLE accepted_periods (home_id INT, grp VARCHAR, phase VARCHAR, therms DOUBLE, "
        "days INT, in_year_window BOOLEAN)"
    )
    raw = [
        (1, "participant", 30.0, 27.0),
        (2, "participant", 60.0, 48.0),
        (3, "comparison", 30.0, 30.0),
        (4, "comparison", 30.0, 27.0),
    ]
    insert = "INSERT INTO accepted_periods VALUES (?, ?, ?, ?, 30, TRUE)"
    for home, grp, pre, post in raw:
        con.execute(insert, [home, grp, "pre", pre])
        con.execute(insert, [home, grp, "post", post])
    # An out-of-window period must not enter the naive figure.
    con.execute("INSERT INTO accepted_periods VALUES (1, 'participant', 'pre', 999, 30, FALSE)")
    con.execute(rendered("050_savings.sql", PipelineParams(naive_min_periods=1)))
    return con


def test_home_savings_is_nac_pre_minus_nac_post(crafted):
    rows = crafted.execute("SELECT home_id, savings, savings_pct FROM home_savings").fetchall()
    assert rows == [(1, 20.0, 20.0), (2, 50.0, 25.0), (3, 2.0, 2.0), (4, -2.0, -2.0)]


def test_difference_in_differences_in_therms_and_percent(crafted):
    p = report.programme_row(crafted)
    assert p["participant_savings"] == 35.0 and p["comparison_savings"] == 0.0
    assert p["did_therms"] == 35.0
    assert p["did_pct"] == pytest.approx(100 * 35 / 150)  # of participants' mean pre NAC


def test_naive_figure_uses_only_the_year_windows(crafted):
    p = report.programme_row(crafted)
    # Participants annualise to 547.5 before and 456.25 after; comparison homes 365 and 346.75.
    assert p["naive_pct"] == pytest.approx(100 * (547.5 - 456.25) / 547.5)
    assert p["comparison_raw_change_pct"] == pytest.approx(5.0)


def test_did_is_within_two_points_of_the_generator_truth(con, synth_data):
    did = report.programme_row(con)["did_pct"]
    assert abs(did - synth_data.truth["programme_true_savings_pct"]) <= 2.0


def test_bootstrap_interval_contains_the_true_savings(con, synth_data, config):
    savings = con.execute("SELECT * FROM home_savings").df()
    result = bootstrap_did(savings, config.bootstrap.resamples, config.bootstrap.seed)
    truth = synth_data.truth["programme_true_savings_pct"]
    assert result.resamples == 500
    assert result.ci_low_pct <= truth <= result.ci_high_pct
    assert result.ci_low_pct < result.did_pct < result.ci_high_pct


def test_naive_pre_post_overstates_the_truth_by_at_least_five_points(con, synth_data):
    p = report.programme_row(con)
    assert p["naive_pct"] - synth_data.truth["programme_true_savings_pct"] >= 5.0


def test_comparison_bills_fall_but_their_nac_does_not(con):
    p = report.programme_row(con)
    assert p["comparison_raw_change_pct"] >= 6.0
    assert abs(p["comparison_nac_change_pct"]) <= 2.0


def test_fixed_65_fits_have_lower_mean_r2_than_per_home_balance_points(con, fixed_con):
    home_r2, fixed_r2, n = report.paired_mean_r2(con, fixed_con)
    assert n > 1000
    assert fixed_r2 < home_r2


def test_fixed_run_only_fits_the_fixed_base(fixed_con):
    assert fixed_con.execute("SELECT base FROM candidate_bases").fetchall() == [(65,)]
    assert fixed_con.execute("SELECT DISTINCT base FROM fit_grid").fetchall() == [(65,)]


def test_per_home_balance_points_give_tighter_home_level_results_than_a_fixed_base(con, fixed_con):
    # Comparison homes did nothing, so the spread of their NAC change is pure method noise.
    per_home = report.programme_row(con)["comparison_spread_pct"]
    fixed = report.programme_row(fixed_con)["comparison_spread_pct"]
    assert per_home < 0.75 * fixed


def test_comparison_spread_is_the_sample_std_of_home_savings_percent(crafted):
    # Comparison homes save 2% and -2%: sample standard deviation sqrt(8).
    assert report.programme_row(crafted)["comparison_spread_pct"] == pytest.approx(8**0.5)
