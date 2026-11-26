import pytest
import yaml
from typer.testing import CliRunner

from weathernorm.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def small_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("cfg") / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "synth": {
                    "n_participants": 30,
                    "n_comparison": 30,
                    "n_meter_swap": 2,
                    "n_short_history": 4,
                },
                "bootstrap": {"resamples": 50},
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def workdir(tmp_path_factory, small_config):
    base = tmp_path_factory.mktemp("work")
    data, out = base / "data", base / "out"
    result = runner.invoke(app, ["generate", "--out", str(data), "--config", str(small_config)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app, ["run", "--data", str(data), "--out", str(out), "--config", str(small_config)]
    )
    assert result.exit_code == 0, result.output
    return data, out, small_config


def test_generate_writes_the_input_and_truth_files(workdir):
    data, _, _ = workdir
    names = {p.name for p in data.iterdir()}
    assert {"weather.csv", "homes.csv", "meter_reads.csv", "truth.json", "truth_homes.csv"} <= names


def test_run_writes_both_databases_and_the_report(workdir):
    _, out, _ = workdir
    assert (out / "weathernorm.duckdb").exists() and (out / "weathernorm_fixed.duckdb").exists()
    text = (out / "report.md").read_text()
    for heading in ("## Programme savings", "## Balance points and fit quality", "## Audit"):
        assert heading in text
    assert "Fixed 65 F" in text and "negative_consumption" in text


def test_savings_prints_per_home_and_fixed_columns(workdir):
    data, out, cfg = workdir
    result = runner.invoke(
        app, ["savings", "--data", str(data), "--out", str(out), "--config", str(cfg)]
    )
    assert result.exit_code == 0
    assert "Per-home balance point" in result.output and "Fixed 65 F" in result.output
    assert "Programme savings (DiD)" in result.output


def test_home_shows_the_r2_curve_and_numpy_check(workdir):
    data, out, _ = workdir
    result = runner.invoke(app, ["home", "1", "--data", str(data), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "R2 by base, pre phase" in result.output and "chosen" in result.output
    assert "NumPy lstsq check" in result.output and "Generated set point" in result.output


def test_home_rejects_an_unknown_id(workdir):
    data, out, _ = workdir
    result = runner.invoke(app, ["home", "99999", "--data", str(data), "--out", str(out)])
    assert result.exit_code == 1


def test_audit_counts_rejections_by_reason(workdir):
    _, out, _ = workdir
    result = runner.invoke(app, ["audit", "--out", str(out)])
    assert result.exit_code == 0
    assert "negative_consumption" in result.output and "insufficient_periods" in result.output


def test_commands_fail_cleanly_before_a_run(tmp_path):
    result = runner.invoke(app, ["audit", "--out", str(tmp_path / "nothing")])
    assert result.exit_code == 1


def test_run_accepts_another_fixed_base_and_labels_it_from_the_database(workdir, tmp_path):
    data, _, cfg = workdir
    out = tmp_path / "out60"
    args = ["--data", str(data), "--out", str(out), "--config", str(cfg)]
    assert runner.invoke(app, ["run", "--fixed-base", "60", *args]).exit_code == 0
    assert "Fixed 60 F" in (out / "report.md").read_text()
    result = runner.invoke(app, ["savings", *args])
    assert "Fixed 60 F" in result.output


def test_no_fixed_skips_the_second_run(workdir, tmp_path):
    data, _, cfg = workdir
    out = tmp_path / "out_nofixed"
    args = ["--data", str(data), "--out", str(out), "--config", str(cfg)]
    assert runner.invoke(app, ["run", "--no-fixed", *args]).exit_code == 0
    assert not (out / "weathernorm_fixed.duckdb").exists()
    assert "Fixed" not in (out / "report.md").read_text()
