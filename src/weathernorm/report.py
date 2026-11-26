"""Summaries of a pipeline run and the Markdown report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from weathernorm.bootstrap import BootstrapResult, bootstrap_did
from weathernorm.check import base_recovery
from weathernorm.config import BootstrapConfig


@dataclass
class RunSummary:
    """Headline numbers of one run (per-home balance points, or one fixed base)."""

    label: str
    programme: dict[str, float]
    bootstrap: BootstrapResult
    recovery: dict[str, float] | None
    mean_r2: float


def load_truth(data_dir: Path | str) -> tuple[dict | None, pd.DataFrame | None]:
    """The generator's truth files, or (None, None) when the data directory has none."""
    data = Path(data_dir)
    truth_path, homes_path = data / "truth.json", data / "truth_homes.csv"
    if not (truth_path.exists() and homes_path.exists()):
        return None, None
    return json.loads(truth_path.read_text()), pd.read_csv(homes_path)


def programme_row(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    return con.execute("SELECT * FROM programme_savings").df().iloc[0].to_dict()


def summarise(
    con: duckdb.DuckDBPyConnection,
    label: str,
    boot: BootstrapConfig,
    truth_homes: pd.DataFrame | None = None,
) -> RunSummary:
    savings = con.execute("SELECT * FROM home_savings").df()
    recovery = None if truth_homes is None else base_recovery(con, truth_homes)
    mean_r2 = con.execute("SELECT AVG(r2) FROM home_fits WHERE fail_reason IS NULL").fetchone()[0]
    return RunSummary(
        label=label,
        programme=programme_row(con),
        bootstrap=bootstrap_did(savings, boot.resamples, boot.seed, boot.ci),
        recovery=recovery,
        mean_r2=float(mean_r2),
    )


def audit_counts(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rejected periods by reason, and failed home fits by phase and reason."""
    rejected = con.execute(
        """
        SELECT reject_reason AS reason, COUNT(*) AS periods, COUNT(DISTINCT home_id) AS homes
        FROM rejected_periods GROUP BY reject_reason ORDER BY reject_reason
        """
    ).df()
    failed = con.execute(
        """
        SELECT fail_reason AS reason, phase, COUNT(*) AS fits
        FROM home_fits WHERE fail_reason IS NOT NULL
        GROUP BY fail_reason, phase ORDER BY fail_reason, phase
        """
    ).df()
    return rejected, failed


def paired_mean_r2(
    per_home: duckdb.DuckDBPyConnection, fixed: duckdb.DuckDBPyConnection
) -> tuple[float, float, int]:
    """Mean R2 of the per-home and the fixed-base fit over the home-phases both could fit."""
    query = "SELECT home_id, phase, r2 FROM home_fits WHERE fail_reason IS NULL"
    merged = (
        per_home.execute(query)
        .df()
        .merge(fixed.execute(query).df(), on=["home_id", "phase"], suffixes=("_home", "_fixed"))
    )
    return float(merged["r2_home"].mean()), float(merged["r2_fixed"].mean()), len(merged)


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def programme_rows(summary: RunSummary, truth: dict | None) -> list[tuple[str, str]]:
    p, b = summary.programme, summary.bootstrap
    rows = [
        (
            "Homes in the estimate (participants / comparison)",
            f"{int(p['n_participants'])} / {int(p['n_comparison'])}",
        ),
        ("Participants' NAC change, therms per year", _fmt(p["participant_savings"])),
        ("Comparison homes' NAC change, therms per year", _fmt(p["comparison_savings"])),
        ("Programme savings (DiD), therms per year", _fmt(b.did_therms)),
        ("Programme savings (DiD), % of participants' pre NAC", f"{b.did_pct:.2f}%"),
        (f"{b.ci:.0%} bootstrap interval, %", f"{b.ci_low_pct:.2f}% to {b.ci_high_pct:.2f}%"),
        ("Participants' NAC change, %", f"{p['participant_nac_change_pct']:.2f}%"),
        ("Comparison homes' NAC change, %", f"{p['comparison_nac_change_pct']:.2f}%"),
        ("Spread of comparison homes' NAC change, std dev %", f"{p['comparison_spread_pct']:.2f}"),
        ("Naive pre/post bill change, participants", f"{p['naive_pct']:.2f}%"),
        ("Pre/post bill change, comparison homes", f"{p['comparison_raw_change_pct']:.2f}%"),
    ]
    if truth is not None:
        rows.append(
            ("True programme savings (generator)", f"{truth['programme_true_savings_pct']:.2f}%")
        )
    return rows


def render_report(
    per_home: RunSummary,
    fixed: RunSummary | None,
    audit: tuple[pd.DataFrame, pd.DataFrame],
    truth: dict | None = None,
    r2_pair: tuple[float, float, int] | None = None,
) -> str:
    """Markdown report: programme savings, balance-point recovery and the audit."""
    lines = ["# Retrofit savings report", "", "## Programme savings", ""]
    columns = [per_home] + ([fixed] if fixed else [])
    lines.append("| Measure | " + " | ".join(c.label for c in columns) + " |")
    lines.append("|---|" + "---:|" * len(columns))
    tables = [dict(programme_rows(c, truth)) for c in columns]
    for name, _ in programme_rows(per_home, truth):
        lines.append(f"| {name} | " + " | ".join(t[name] for t in tables) + " |")
    lines += [
        "",
        "NAC is normalised annual consumption. The DiD figure is the participants' mean NAC",
        "change minus the comparison homes' mean NAC change; the naive figure is the raw bill",
        "change with no weather adjustment, and includes the milder winter.",
        "",
        "## Balance points and fit quality",
        "",
    ]
    if per_home.recovery is not None:
        r = per_home.recovery
        lines += [
            f"Participants' accepted pre-phase fits ({int(r['n'])} homes): "
            f"{100 * r['share_within']:.1f}% chose a base within 2 F of the home's set point, "
            f"mean absolute error {r['mean_abs_error']:.2f} F, bias {r['bias']:+.2f} F.",
            "",
        ]
    if r2_pair is not None:
        home_r2, fixed_r2, n = r2_pair
        lines += [
            f"Mean R2 over {n} home-phase fits: {home_r2:.4f} with per-home balance points, "
            f"{fixed_r2:.4f} with the fixed base.",
            "",
        ]
    rejected, failed = audit
    lines += [
        "## Audit",
        "",
        "Rejected billing periods:",
        "",
        "| Reason | Periods | Homes |",
        "|---|---:|---:|",
    ]
    lines += [f"| {r.reason} | {r.periods} | {r.homes} |" for r in rejected.itertuples()]
    lines += ["", "Failed fits:", "", "| Reason | Phase | Fits |", "|---|---|---:|"]
    lines += [f"| {r.reason} | {r.phase} | {r.fits} |" for r in failed.itertuples()]
    return "\n".join(lines) + "\n"
