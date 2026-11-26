"""Runs the SQL files in order on an in-process DuckDB."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

from weathernorm.config import PipelineParams, default_sql_dir

SQL_FILES = (
    "010_billing_periods.sql",
    "020_degree_days.sql",
    "030_fit.sql",
    "040_nac.sql",
    "050_savings.sql",
)

# Tables that make up a run's output, with the ordering that identifies each row.
OUTPUT_TABLES = {
    "periods": "home_id, period_seq",
    "rejected_periods": "home_id, period_seq",
    "period_hdd": "home_id, period_seq, base",
    "normal_annual_hdd": "base",
    "fit_grid": "home_id, phase, base",
    "home_fits": "home_id, phase",
    "nac": "home_id, phase",
    "home_savings": "home_id",
    "naive_raw": "home_id",
    "group_savings": "grp",
    "programme_savings": "n_participants",
}

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def render_sql(text: str, values: dict[str, int]) -> str:
    """Substitute ${name} placeholders; an unknown name is an error rather than a silent gap."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise KeyError(f"SQL placeholder ${{{name}}} has no value")
        return str(int(values[name]))

    return _PLACEHOLDER.sub(replace, text)


def load_inputs(con: duckdb.DuckDBPyConnection, data_dir: Path | str) -> None:
    """Load weather.csv, homes.csv and meter_reads.csv with explicit column types."""
    data = Path(data_dir)
    con.execute(
        "CREATE OR REPLACE TABLE weather AS SELECT * FROM read_csv(?, "
        "columns={'date': 'DATE', 'temp_f': 'DOUBLE'}, header=true)",
        [str(data / "weather.csv")],
    )
    con.execute(
        "CREATE OR REPLACE TABLE homes AS SELECT * FROM read_csv(?, "
        "columns={'home_id': 'INTEGER', 'grp': 'VARCHAR', 'event_date': 'DATE'}, header=true)",
        [str(data / "homes.csv")],
    )
    con.execute(
        "CREATE OR REPLACE TABLE meter_reads AS SELECT * FROM read_csv(?, "
        "columns={'home_id': 'INTEGER', 'read_date': 'DATE', 'reading': 'DOUBLE', "
        "'is_estimated': 'BOOLEAN'}, header=true)",
        [str(data / "meter_reads.csv")],
    )


def run_sql(
    con: duckdb.DuckDBPyConnection,
    params: PipelineParams,
    fixed_base: int | None = None,
    sql_dir: Path | str | None = None,
) -> None:
    """Execute the numbered SQL files in order against tables already loaded in `con`."""
    directory = Path(sql_dir) if sql_dir else default_sql_dir()
    values = params.sql_values(fixed_base)
    for name in SQL_FILES:
        con.execute(render_sql((directory / name).read_text(), values))


def run(
    data_dir: Path | str,
    params: PipelineParams | None = None,
    fixed_base: int | None = None,
    db_path: Path | str = ":memory:",
    sql_dir: Path | str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Load the inputs and run the whole pipeline.

    One thread keeps floating-point aggregation order, and so every output value, identical
    from run to run. `fixed_base` limits the candidate bases to that single temperature.
    """
    params = params or PipelineParams()
    path = str(db_path)
    if path != ":memory:":
        Path(path).unlink(missing_ok=True)
    con = duckdb.connect(path)
    con.execute("SET threads = 1")
    load_inputs(con, data_dir)
    run_sql(con, params, fixed_base=fixed_base, sql_dir=sql_dir)
    return con


def output_tables(con: duckdb.DuckDBPyConnection) -> dict[str, list[tuple]]:
    """Every output table as ordered rows, for comparing two runs."""
    return {
        name: con.execute(f"SELECT * FROM {name} ORDER BY {order}").fetchall()
        for name, order in OUTPUT_TABLES.items()
    }
