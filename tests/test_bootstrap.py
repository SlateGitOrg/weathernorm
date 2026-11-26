import numpy as np
import pandas as pd
import pytest

from weathernorm.bootstrap import bootstrap_did, did_from_arrays


def _savings(p, c, pre=1000.0):
    rows = [("participant", s, pre) for s in p] + [("comparison", s, pre) for s in c]
    return pd.DataFrame(rows, columns=["grp", "savings", "nac_pre"])


def test_point_estimate_is_the_difference_of_group_means():
    did, pct = did_from_arrays(
        np.array([100.0, 200.0]), np.array([1000.0, 2000.0]), np.array([10.0])
    )
    assert did == 140.0
    assert pct == pytest.approx(100 * 140 / 1500)


def test_interval_collapses_when_every_home_is_identical():
    result = bootstrap_did(_savings([100.0] * 20, [10.0] * 20))
    assert result.did_pct == pytest.approx(9.0)
    assert result.ci_low_pct == pytest.approx(9.0)
    assert result.ci_high_pct == pytest.approx(9.0)


def test_same_seed_same_interval_and_other_seed_differs():
    rng = np.random.default_rng(1)
    frame = _savings(rng.normal(150, 40, 80), rng.normal(0, 40, 80))
    a, b = bootstrap_did(frame, seed=5), bootstrap_did(frame, seed=5)
    assert a == b
    assert bootstrap_did(frame, seed=6).ci_low_pct != a.ci_low_pct


def test_interval_width_shrinks_with_more_homes():
    rng = np.random.default_rng(2)
    small = bootstrap_did(_savings(rng.normal(150, 40, 30), rng.normal(0, 40, 30)))
    large = bootstrap_did(_savings(rng.normal(150, 40, 600), rng.normal(0, 40, 600)))
    assert (large.ci_high_pct - large.ci_low_pct) < 0.5 * (small.ci_high_pct - small.ci_low_pct)


def test_interval_covers_the_true_difference_in_most_simulations():
    covered = 0
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        frame = _savings(rng.normal(150, 40, 100), rng.normal(0, 40, 100))
        result = bootstrap_did(frame, resamples=200, seed=seed)
        covered += result.ci_low_therms <= 150.0 <= result.ci_high_therms
    assert covered >= 34  # a 95% interval: expect about 38 of 40
