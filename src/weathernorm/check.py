"""NumPy reference for the SQL regressions, and scoring against the generator's truth."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fit:
    slope: float
    intercept: float
    r2: float
    n: int


def fit_points(
    con: duckdb.DuckDBPyConnection, home_id: int, phase: str, base: int
) -> tuple[np.ndarray, np.ndarray]:
    """(HDD/day, therms/day) for one home, phase and base, in period order."""
    rows = con.execute(
        """
        SELECT h.hdd / p.days, p.therms / p.days
        FROM accepted_periods p
        JOIN period_hdd h USING (home_id, period_seq)
        WHERE p.home_id = ? AND p.phase = ? AND h.base = ?
        ORDER BY p.period_seq
        """,
        [home_id, phase, base],
    ).fetchall()
    data = np.array(rows, dtype=float).reshape(-1, 2)
    return data[:, 0], data[:, 1]


def lstsq_fit(x: np.ndarray, y: np.ndarray) -> Fit:
    """Ordinary least squares of y on x with an intercept, by numpy.linalg.lstsq."""
    design = np.column_stack([np.ones_like(x), x])
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - (intercept + slope * x)
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / total if total > 0 else float("nan")
    return Fit(float(slope), float(intercept), r2, len(x))


def reference_fit(con: duckdb.DuckDBPyConnection, home_id: int, phase: str, base: int) -> Fit:
    x, y = fit_points(con, home_id, phase, base)
    return lstsq_fit(x, y)


def sql_fit(con: duckdb.DuckDBPyConnection, home_id: int, phase: str, base: int) -> Fit | None:
    row = con.execute(
        "SELECT slope, intercept, r2, n FROM fit_grid WHERE home_id = ? AND phase = ? AND base = ?",
        [home_id, phase, base],
    ).fetchone()
    return None if row is None else Fit(float(row[0]), float(row[1]), float(row[2]), int(row[3]))


def max_relative_difference(sql: Fit, reference: Fit) -> float:
    """Largest relative gap between the SQL and NumPy slope, intercept and R2."""
    gaps = []
    for a, b in ((sql.slope, reference.slope), (sql.intercept, reference.intercept)):
        gaps.append(abs(a - b) / max(abs(b), 1e-300))
    gaps.append(abs(sql.r2 - reference.r2) / max(abs(reference.r2), 1e-300))
    return max(gaps)


def r2_curve(con: duckdb.DuckDBPyConnection, home_id: int, phase: str) -> pd.DataFrame:
    """The home's SQL fit at every candidate base, with a flag on the chosen one."""
    return con.execute(
        """
        SELECT g.base, g.n, g.slope, g.intercept, g.r2, (g.base = f.base) AS chosen
        FROM fit_grid g
        JOIN home_fits f ON f.home_id = g.home_id AND f.phase = g.phase
        WHERE g.home_id = ? AND g.phase = ?
        ORDER BY g.base
        """,
        [home_id, phase],
    ).df()


def base_recovery(
    con: duckdb.DuckDBPyConnection,
    truth_homes: pd.DataFrame,
    group: str = "participant",
    phase: str = "pre",
    tolerance: int = 2,
) -> dict[str, float]:
    """Compare chosen balance points with each home's generated set point."""
    fits = con.execute(
        """
        SELECT f.home_id, f.base
        FROM home_fits f
        WHERE f.fail_reason IS NULL AND f.phase = ? AND f.grp = ?
        """,
        [phase, group],
    ).df()
    merged = fits.merge(truth_homes[["home_id", "true_base"]], on="home_id")
    error = merged["base"] - merged["true_base"]
    return {
        "n": float(len(merged)),
        "share_within": float((error.abs() <= tolerance).mean()),
        "mean_abs_error": float(error.abs().mean()),
        "bias": float(error.mean()),
    }
