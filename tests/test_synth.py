import numpy as np
import pandas as pd

from weathernorm import synth
from weathernorm.config import SynthConfig


def test_sizes_and_groups(synth_data):
    assert len(synth_data.homes) == 600
    assert (synth_data.homes["grp"] == "participant").sum() == 300
    assert (synth_data.homes["grp"] == "comparison").sum() == 300
    assert len(synth_data.weather) == 3653  # ten years, 2016 to 2025
    assert 20_000 < len(synth_data.reads) < 22_500


def test_same_seed_same_data_and_other_seed_differs():
    small = SynthConfig(n_participants=10, n_comparison=10, n_meter_swap=1, n_short_history=2)
    a, b = synth.generate(small), synth.generate(small)
    pd.testing.assert_frame_equal(a.reads, b.reads)
    pd.testing.assert_frame_equal(a.weather, b.weather)
    other = synth.generate(SynthConfig(**{**vars(small), "seed": 8}))
    assert not a.weather["temp_f"].equals(other.weather["temp_f"])


def test_true_parameters_within_documented_ranges(synth_data):
    t = synth_data.truth_homes
    assert t["true_base"].between(55, 65).all()
    assert t["alpha"].between(0.8, 2.5).all()
    assert t["beta"].between(0.10, 0.25).all()
    assert set(t["true_base"]) == set(range(55, 66))


def test_retrofit_cuts_beta_by_18_percent_and_leaves_alpha(synth_data):
    t = synth_data.truth_homes
    part = t[t["is_participant"]]
    cut = part["nac_pre"] - part["nac_post"]
    np.testing.assert_allclose(cut, 0.18 * part["beta"] * part["normal_hdd"], rtol=1e-12)
    comparison = t[~t["is_participant"]]
    assert (comparison["nac_pre"] == comparison["nac_post"]).all()


def test_post_retrofit_winter_is_about_15_percent_milder(synth_data):
    weather = synth_data.weather.assign(date=lambda d: pd.to_datetime(d["date"]))
    hdd = np.maximum(65.0 - weather["temp_f"], 0.0)

    def season(year):
        mask = (weather["date"] >= f"{year}-10-01") & (weather["date"] <= f"{year + 1}-04-30")
        return hdd[mask].sum()

    normal = np.mean([season(y) for y in range(2016, 2024)])
    assert 0.80 < season(2024) / normal < 0.90


def test_injected_defects_are_present(synth_data):
    truth = synth_data.truth
    assert len(truth["meter_swap_homes"]) == 6
    assert len(truth["short_history_homes"]) == 12
    assert not set(truth["meter_swap_homes"]) & set(truth["short_history_homes"])
    share = truth["n_estimated_reads"] / truth["n_reads"]
    assert 0.065 < share < 0.085


def test_first_and_last_read_of_every_home_are_actual(synth_data):
    reads = synth_data.reads.sort_values(["home_id", "read_date"])
    assert not reads.groupby("home_id")["is_estimated"].first().any()
    assert not reads.groupby("home_id")["is_estimated"].last().any()


def test_read_intervals_are_27_to_35_days(synth_data):
    reads = synth_data.reads.sort_values(["home_id", "read_date"])
    gaps = pd.to_datetime(reads["read_date"]).groupby(reads["home_id"]).diff().dropna().dt.days
    assert gaps.min() == 27 and gaps.max() == 35


def test_programme_truth_is_the_ratio_of_participant_sums(synth_data):
    t = synth_data.truth_homes
    part = t[t["is_participant"]]
    expected = 100 * (part["nac_pre"] - part["nac_post"]).sum() / part["nac_pre"].sum()
    assert synth_data.truth["programme_true_savings_pct"] == expected
    assert 9.0 < expected < 12.0


def test_normal_annual_hdd_drops_29_february():
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")  # 2020 is a leap year
    weather = pd.DataFrame({"date": dates.date, "temp_f": 40.0})
    assert synth.normal_annual_hdd(weather, np.array([65]))[65] == 365 * 25.0
