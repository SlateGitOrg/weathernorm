import pytest

from weathernorm import pipeline
from weathernorm.config import PipelineParams, load_config


def test_rerunning_on_the_same_inputs_gives_identical_tables(data_dir, config, con):
    first = pipeline.output_tables(con)
    second = pipeline.output_tables(pipeline.run(data_dir, config.pipeline))
    assert first.keys() == second.keys() == pipeline.OUTPUT_TABLES.keys()
    for name in first:
        assert first[name] == second[name], name
    assert len(first["periods"]) > 17_000


def test_output_tables_are_stored_in_their_ordering_key(con):
    for name in ("periods", "fit_grid", "home_fits", "nac", "home_savings"):
        stored = con.execute(f"SELECT * FROM {name}").fetchall()
        ordered = con.execute(
            f"SELECT * FROM {name} ORDER BY {pipeline.OUTPUT_TABLES[name]}"
        ).fetchall()
        assert stored == ordered, name


def test_database_file_run_matches_the_in_memory_run(data_dir, config, con, tmp_path):
    on_disk = pipeline.run(data_dir, config.pipeline, db_path=tmp_path / "w.duckdb")
    assert pipeline.output_tables(on_disk) == pipeline.output_tables(con)
    on_disk.close()


def test_every_period_has_a_weather_value_for_each_of_its_days(con):
    bad = con.execute(
        "SELECT COUNT(*) FROM period_hdd h JOIN accepted_periods p USING (home_id, period_seq) "
        "WHERE h.weather_days <> p.days"
    ).fetchone()[0]
    assert bad == 0


def test_render_sql_substitutes_and_rejects_unknown_placeholders():
    assert pipeline.render_sql("x > ${a} AND y < ${b}", {"a": 20, "b": 40}) == "x > 20 AND y < 40"
    with pytest.raises(KeyError, match="missing"):
        pipeline.render_sql("${missing}", {"a": 1})


def test_every_sql_placeholder_is_provided_by_the_parameters():
    from conftest import sql_text

    values = PipelineParams().sql_values()
    for name in pipeline.SQL_FILES:
        pipeline.render_sql(sql_text(name), values)  # raises KeyError on an unknown name


def test_fixed_base_collapses_the_candidate_range():
    values = PipelineParams().sql_values(fixed_base=65)
    assert values["base_min"] == values["base_max"] == 65


def test_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text("pipeline:\n  min_dayz: 3\n")
    with pytest.raises(ValueError, match="min_dayz"):
        load_config(bad)


def test_default_config_matches_the_documented_design(config):
    assert config.pipeline.min_days == 20 and config.pipeline.max_days == 40
    assert config.pipeline.min_periods == 9
    assert (config.pipeline.base_min, config.pipeline.base_max) == (50, 70)
    assert config.bootstrap.resamples == 500
    assert config.synth.n_participants == config.synth.n_comparison == 300
