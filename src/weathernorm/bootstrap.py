"""Cluster bootstrap over homes for the difference-in-differences interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    did_therms: float
    did_pct: float
    ci_low_pct: float
    ci_high_pct: float
    ci_low_therms: float
    ci_high_therms: float
    resamples: int
    seed: int
    ci: float


def did_from_arrays(
    participant_savings: np.ndarray, participant_nac_pre: np.ndarray, comparison_savings: np.ndarray
) -> tuple[float, float]:
    """(DiD in therms, DiD as % of participants' mean pre-phase NAC)."""
    did = float(participant_savings.mean() - comparison_savings.mean())
    return did, 100.0 * did / float(participant_nac_pre.mean())


def bootstrap_did(
    home_savings: pd.DataFrame, resamples: int = 500, seed: int = 11, ci: float = 0.95
) -> BootstrapResult:
    """Resample whole homes with replacement within each group and recompute the DiD.

    A home is the cluster: both of its phases travel together, so the interval reflects
    between-home variation, which is what the programme estimate averages over.
    """
    part = home_savings[home_savings["grp"] == "participant"]
    comp = home_savings[home_savings["grp"] == "comparison"]
    p_sav, p_pre = part["savings"].to_numpy(), part["nac_pre"].to_numpy()
    c_sav = comp["savings"].to_numpy()
    point_therms, point_pct = did_from_arrays(p_sav, p_pre, c_sav)

    rng = np.random.default_rng(seed)
    p_idx = rng.integers(0, len(p_sav), (resamples, len(p_sav)))
    c_idx = rng.integers(0, len(c_sav), (resamples, len(c_sav)))
    therms = p_sav[p_idx].mean(axis=1) - c_sav[c_idx].mean(axis=1)
    pct = 100.0 * therms / p_pre[p_idx].mean(axis=1)
    lo, hi = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    return BootstrapResult(
        did_therms=point_therms,
        did_pct=point_pct,
        ci_low_pct=float(np.quantile(pct, lo)),
        ci_high_pct=float(np.quantile(pct, hi)),
        ci_low_therms=float(np.quantile(therms, lo)),
        ci_high_therms=float(np.quantile(therms, hi)),
        resamples=resamples,
        seed=seed,
        ci=ci,
    )
