"""Command line interface: generate, run, home, savings, audit."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Annotated

import duckdb
import typer
from rich.console import Console
from rich.table import Table

from weathernorm import check, pipeline, report, synth
from weathernorm.config import Config, load_config

app = typer.Typer(help="Weather-normalised retrofit savings.", no_args_is_help=True)
console = Console()

DATA_OPT = Annotated[Path, typer.Option("--data", help="Directory with the input CSVs.")]
OUT_OPT = Annotated[Path, typer.Option("--out", help="Directory for pipeline output.")]
CONFIG_OPT = Annotated[Path | None, typer.Option("--config", help="Path to config.yaml.")]

PER_HOME_DB = "weathernorm.duckdb"
FIXED_DB = "weathernorm_fixed.duckdb"


def _open(out: Path, name: str) -> duckdb.DuckDBPyConnection:
    path = out / name
    if not path.exists():
        console.print(f"[red]{path} not found. Run `weathernorm run` first.[/red]")
        raise typer.Exit(1)
    return duckdb.connect(str(path), read_only=True)


def _open_optional(out: Path, name: str) -> duckdb.DuckDBPyConnection | None:
    path = out / name
    return duckdb.connect(str(path), read_only=True) if path.exists() else None


def _summaries(
    cfg: Config, out: Path, data: Path
) -> tuple[report.RunSummary, report.RunSummary | None, dict | None, object]:
    truth, truth_homes = report.load_truth(data)
    con = _open(out, PER_HOME_DB)
    per_home = report.summarise(con, "Per-home balance point", cfg.bootstrap, truth_homes)
    fixed_con = _open_optional(out, FIXED_DB)
    fixed = None
    if fixed_con is not None:
        base = fixed_con.execute("SELECT MIN(base) FROM candidate_bases").fetchone()[0]
        label = f"Fixed {base} F"
        fixed = report.summarise(fixed_con, label, cfg.bootstrap, truth_homes)
    return per_home, fixed, truth, (con, fixed_con)


@app.command()
def generate(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("data"),
    seed: Annotated[int | None, typer.Option(help="Override the seed in config.yaml.")] = None,
    config: CONFIG_OPT = None,
) -> None:
    """Write the synthetic weather, homes, meter reads and truth files."""
    cfg = load_config(config)
    synth_cfg = cfg.synth if seed is None else dataclasses.replace(cfg.synth, seed=seed)
    data = synth.generate(synth_cfg)
    synth.write(data, out)
    console.print(
        f"Wrote {len(data.homes)} homes, {len(data.reads)} meter reads and "
        f"{len(data.weather)} days of weather to {out}/"
    )


@app.command()
def run(
    data: DATA_OPT = Path("data"),
    out: OUT_OPT = Path("out"),
    fixed_base: Annotated[
        int | None, typer.Option("--fixed-base", help="Base for the comparison run (F).")
    ] = None,
    no_fixed: Annotated[bool, typer.Option("--no-fixed", help="Skip the fixed-base run.")] = False,
    config: CONFIG_OPT = None,
) -> None:
    """Run the SQL pipeline per home and at a fixed base, and write the report."""
    cfg = load_config(config)
    out.mkdir(parents=True, exist_ok=True)
    con = pipeline.run(data, cfg.pipeline, db_path=out / PER_HOME_DB)
    con.close()
    if not no_fixed:
        base = fixed_base if fixed_base is not None else cfg.pipeline.fixed_base
        fixed_con = pipeline.run(data, cfg.pipeline, fixed_base=base, db_path=out / FIXED_DB)
        fixed_con.close()
    per_home, fixed, truth, (con, fixed_con) = _summaries(cfg, out, data)
    r2_pair = report.paired_mean_r2(con, fixed_con) if fixed_con is not None else None
    text = report.render_report(per_home, fixed, report.audit_counts(con), truth, r2_pair)
    (out / "report.md").write_text(text)
    console.print(f"Pipeline finished; report written to {out / 'report.md'}")


@app.command()
def savings(data: DATA_OPT = Path("data"), out: OUT_OPT = Path("out"), config: CONFIG_OPT = None):
    """Print programme savings: per-home balance points beside the fixed base."""
    cfg = load_config(config)
    per_home, fixed, truth, _ = _summaries(cfg, out, data)
    columns = [per_home] + ([fixed] if fixed else [])
    table = Table(title="Programme savings")
    table.add_column("Measure")
    for column in columns:
        table.add_column(column.label, justify="right")
    cells = [dict(report.programme_rows(c, truth)) for c in columns]
    for name, _ in report.programme_rows(per_home, truth):
        table.add_row(name, *[c[name] for c in cells])
    console.print(table)


@app.command()
def home(
    home_id: int,
    data: DATA_OPT = Path("data"),
    out: OUT_OPT = Path("out"),
    span: Annotated[int, typer.Option(help="Show bases within this many F of the chosen one.")] = 5,
    all_bases: Annotated[bool, typer.Option("--all", help="Show every candidate base.")] = False,
) -> None:
    """Show one home's R2 by base for each phase and its chosen fit, checked against NumPy."""
    con = _open(out, PER_HOME_DB)
    _, truth_homes = report.load_truth(data)
    info = con.execute("SELECT grp, event_date FROM homes WHERE home_id = ?", [home_id]).fetchone()
    if info is None:
        console.print(f"[red]No home {home_id}.[/red]")
        raise typer.Exit(1)
    console.print(f"Home {home_id}: {info[0]}, event date {info[1]}")
    if truth_homes is not None:
        t = truth_homes[truth_homes["home_id"] == home_id].iloc[0]
        console.print(
            f"Generated set point {int(t['true_base'])} F, "
            f"alpha {t['alpha']:.3f}, beta {t['beta']:.4f}"
        )
    for phase in ("pre", "post"):
        fit = con.execute(
            "SELECT base, n, slope, intercept, r2, fail_reason FROM home_fits "
            "WHERE home_id = ? AND phase = ?",
            [home_id, phase],
        ).fetchone()
        if fit is None or fit[0] is None:
            console.print(f"[yellow]{phase}: no accepted periods.[/yellow]")
            continue
        curve = check.r2_curve(con, home_id, phase)
        if not all_bases:
            curve = curve[(curve["base"] - fit[0]).abs() <= span]
        table = Table(title=f"R2 by base, {phase} phase")
        for name in ("Base F", "Periods", "Slope", "Intercept", "R2", ""):
            table.add_column(name, justify="right")
        for row in curve.itertuples():
            table.add_row(
                str(row.base),
                str(row.n),
                f"{row.slope:.5f}",
                f"{row.intercept:.4f}",
                f"{row.r2:.5f}",
                "chosen" if row.chosen else "",
            )
        console.print(table)
        status = fit[5] or "ok"
        console.print(
            f"{phase}: base {fit[0]} F, slope {fit[2]:.5f}, intercept {fit[3]:.4f}, "
            f"R2 {fit[4]:.5f}, {fit[1]} periods, status {status}"
        )
        reference = check.reference_fit(con, home_id, phase, int(fit[0]))
        sql = check.sql_fit(con, home_id, phase, int(fit[0]))
        gap = check.max_relative_difference(sql, reference)
        console.print(f"NumPy lstsq check: max relative difference {gap:.2e}")


@app.command()
def audit(out: OUT_OPT = Path("out")) -> None:
    """Rejected billing periods and failed fits, counted by reason."""
    con = _open(out, PER_HOME_DB)
    rejected, failed = report.audit_counts(con)
    table = Table(title="Rejected billing periods")
    for name in ("Reason", "Periods", "Homes"):
        table.add_column(name, justify="right" if name != "Reason" else "left")
    for row in rejected.itertuples():
        table.add_row(row.reason, str(row.periods), str(row.homes))
    console.print(table)
    table = Table(title="Failed fits")
    for name in ("Reason", "Phase", "Fits"):
        table.add_column(name, justify="right" if name == "Fits" else "left")
    for row in failed.itertuples():
        table.add_row(row.reason, row.phase, str(row.fits))
    console.print(table)
