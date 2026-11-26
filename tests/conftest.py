from __future__ import annotations

import duckdb
import pytest

from weathernorm import pipeline, synth
from weathernorm.config import PipelineParams, default_sql_dir, load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def synth_data(config):
    return synth.generate(config.synth)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, synth_data):
    path = tmp_path_factory.mktemp("data")
    synth.write(synth_data, path)
    return path


@pytest.fixture(scope="session")
def con(data_dir, config):
    return pipeline.run(data_dir, config.pipeline)


@pytest.fixture(scope="session")
def fixed_con(data_dir, config):
    return pipeline.run(data_dir, config.pipeline, fixed_base=config.pipeline.fixed_base)


def sql_text(name: str) -> str:
    return (default_sql_dir() / name).read_text()


def rendered(name: str, params: PipelineParams | None = None) -> str:
    params = params or PipelineParams()
    return pipeline.render_sql(sql_text(name), params.sql_values())


def mini_connection(weather, homes, reads) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB holding hand-built inputs: weather, homes and meter reads."""
    con = duckdb.connect(":memory:")
    con.execute("SET threads = 1")
    con.execute("CREATE TABLE weather (date DATE, temp_f DOUBLE)")
    con.execute("CREATE TABLE homes (home_id INTEGER, grp VARCHAR, event_date DATE)")
    con.execute(
        "CREATE TABLE meter_reads (home_id INTEGER, read_date DATE, reading DOUBLE, "
        "is_estimated BOOLEAN)"
    )
    for table, rows, marks in (
        ("weather", weather, "?, ?"),
        ("homes", homes, "?, ?, ?"),
        ("meter_reads", reads, "?, ?, ?, ?"),
    ):
        if rows:
            con.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
    return con
