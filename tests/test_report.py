import pytest

from weathernorm import report
from weathernorm.config import BootstrapConfig


@pytest.fixture(scope="module")
def summaries(con, fixed_con, synth_data):
    boot = BootstrapConfig(resamples=100)
    per_home = report.summarise(con, "Per-home", boot, synth_data.truth_homes)
    fixed = report.summarise(fixed_con, "Fixed 65 F", boot, synth_data.truth_homes)
    return per_home, fixed


def test_audit_counts_match_the_generated_defects(con, synth_data):
    rejected, failed = report.audit_counts(con)
    by_reason = dict(zip(rejected["reason"], rejected["periods"], strict=True))
    assert by_reason["negative_consumption"] == 6
    assert set(by_reason) == {"negative_consumption", "too_long", "blackout"}
    pre = failed[(failed["reason"] == "insufficient_periods") & (failed["phase"] == "pre")]
    assert int(pre["fits"].iloc[0]) >= 12  # the 12 late move-ins, plus a few merge casualties
    assert set(failed["reason"]) == {"insufficient_periods"}


def test_recovery_is_scored_against_the_set_points(summaries):
    recovery = summaries[0].recovery
    assert recovery["share_within"] >= 0.90 and recovery["mean_abs_error"] < 1.0


def test_report_places_fixed_and_per_home_results_side_by_side(
    con, fixed_con, summaries, synth_data
):
    per_home, fixed = summaries
    text = report.render_report(
        per_home,
        fixed,
        report.audit_counts(con),
        synth_data.truth,
        report.paired_mean_r2(con, fixed_con),
    )
    header = next(line for line in text.splitlines() if line.startswith("| Measure"))
    assert header == "| Measure | Per-home | Fixed 65 F |"
    assert "True programme savings (generator)" in text
    assert "Mean R2 over" in text and "within 2 F of the home's set point" in text


def test_report_without_a_fixed_run_has_one_column(con, summaries):
    text = report.render_report(summaries[0], None, report.audit_counts(con))
    header = next(line for line in text.splitlines() if line.startswith("| Measure"))
    assert header == "| Measure | Per-home |"
    assert "True programme savings" not in text


def test_load_truth_returns_none_for_a_directory_without_truth(tmp_path):
    assert report.load_truth(tmp_path) == (None, None)
