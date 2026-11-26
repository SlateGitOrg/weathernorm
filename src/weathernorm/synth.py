"""Seeded synthetic gas-utility data: daily weather, homes and irregular meter reads.

Each home heats to its own balance-point temperature. Daily therms are
``alpha + beta * max(base - T, 0)``; a retrofit cuts beta (never alpha). Meter reads arrive
every 27-35 days with multiplicative noise. Injected defects: estimated reads that the next
actual read catches up, meter swaps (one negative interval) and late move-ins (short history).
The generator keeps every home's true values so tests can score the analysis against them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from weathernorm.config import SynthConfig

HDD_REFERENCE_BASE = 65.0


@dataclass
class SynthData:
    weather: pd.DataFrame
    homes: pd.DataFrame
    reads: pd.DataFrame
    truth_homes: pd.DataFrame
    truth: dict


def _hdd_sum(temps: np.ndarray, base: float) -> float:
    return float(np.maximum(base - temps, 0.0).sum())


def _mild_winter_offset(dates: pd.DatetimeIndex, temps: np.ndarray, cfg: SynthConfig) -> float:
    """Degrees F added in the mild-winter window so its HDD65 is `cut` below other winters."""
    start, end = pd.Timestamp(cfg.mild_winter_start), pd.Timestamp(cfg.mild_winter_end)
    window = (dates >= start) & (dates <= end)
    seasons = []
    for shift in range(1, 10):
        s = start - pd.DateOffset(years=shift)
        e = end - pd.DateOffset(years=shift)
        if s < dates[0] or e > dates[-1]:
            continue
        mask = (dates >= s) & (dates <= e)
        seasons.append(_hdd_sum(temps[mask], HDD_REFERENCE_BASE))
    target = (1.0 - cfg.mild_winter_hdd_cut) * float(np.mean(seasons))
    low, high = 0.0, 40.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if _hdd_sum(temps[window] + mid, HDD_REFERENCE_BASE) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def make_weather(cfg: SynthConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Daily mean temperature in F: sinusoid plus AR(1) noise, one mild winter."""
    dates = pd.date_range(cfg.weather_start, cfg.weather_end, freq="D")
    doy = dates.day_of_year.to_numpy()
    seasonal = cfg.mean_temp_f - cfg.amplitude_f * np.cos(2 * np.pi * (doy - 15) / 365.25)
    shocks = rng.normal(0.0, cfg.ar_sigma_f, len(dates))
    noise = np.empty(len(dates))
    noise[0] = shocks[0] / np.sqrt(1.0 - cfg.ar_phi**2)
    for i in range(1, len(dates)):
        noise[i] = cfg.ar_phi * noise[i - 1] + shocks[i]
    temps = seasonal + noise
    offset = _mild_winter_offset(dates, temps, cfg)
    window = (dates >= pd.Timestamp(cfg.mild_winter_start)) & (
        dates <= pd.Timestamp(cfg.mild_winter_end)
    )
    temps = np.where(window, temps + offset, temps)
    weather = pd.DataFrame({"date": dates.date, "temp_f": np.round(temps, 1)})
    return weather, offset


def normal_annual_hdd(weather: pd.DataFrame, bases: np.ndarray) -> dict[int, float]:
    """Sum over the 365 calendar days of the mean-by-day-of-year HDD (29 February dropped)."""
    frame = pd.to_datetime(weather["date"])
    keep = ~((frame.dt.month == 2) & (frame.dt.day == 29))
    temps = weather["temp_f"].to_numpy()[keep.to_numpy()]
    keys = (frame.dt.month * 100 + frame.dt.day).to_numpy()[keep.to_numpy()]
    out = {}
    for base in bases:
        hdd = np.maximum(base - temps, 0.0)
        daily = pd.Series(hdd).groupby(keys).mean()
        out[int(base)] = float(daily.sum())
    return out


def _random_dates(rng: np.random.Generator, first: date, last: date, n: int) -> list[date]:
    span = (last - first).days + 1
    return [first + timedelta(days=int(d)) for d in rng.integers(0, span, n)]


def generate(cfg: SynthConfig | None = None) -> SynthData:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    weather, offset = make_weather(cfg, rng)
    temps = weather["temp_f"].to_numpy()
    day0 = pd.Timestamp(cfg.weather_start)

    def day_index(d: date) -> int:
        return int((pd.Timestamp(d) - day0).days)

    n_home = cfg.n_participants + cfg.n_comparison
    groups = np.array(["participant"] * cfg.n_participants + ["comparison"] * cfg.n_comparison)
    groups = groups[rng.permutation(n_home)]
    event_dates = _random_dates(rng, cfg.retrofit_first, cfg.retrofit_last, n_home)
    bases = rng.integers(cfg.base_low, cfg.base_high + 1, n_home)
    alphas = rng.uniform(cfg.alpha_low, cfg.alpha_high, n_home)
    betas = rng.uniform(cfg.beta_low, cfg.beta_high, n_home)

    ids = np.arange(1, n_home + 1)
    participants = ids[groups == "participant"]
    comparison = ids[groups == "comparison"]
    half = cfg.n_short_history // 2
    short = set(rng.choice(participants, half, replace=False).tolist()) | set(
        rng.choice(comparison, cfg.n_short_history - half, replace=False).tolist()
    )
    eligible = np.array([i for i in ids if i not in short])
    swap = set(rng.choice(eligible, cfg.n_meter_swap, replace=False).tolist())

    last_read_idx = day_index(cfg.reads_end)
    first_read_idx = day_index(cfg.reads_start)
    read_rows = []
    truth_rows = []
    n_estimated = 0
    for pos, home_id in enumerate(ids):
        is_participant = groups[pos] == "participant"
        base, alpha, beta = float(bases[pos]), float(alphas[pos]), float(betas[pos])
        event_idx = day_index(event_dates[pos])
        hdd = np.maximum(base - temps, 0.0)
        beta_day = np.full(len(temps), beta)
        if is_participant:
            beta_day[event_idx:] = beta * (1.0 - cfg.retrofit_beta_cut)
        cum = np.concatenate([[0.0], np.cumsum(alpha + beta_day * hdd)])

        def used(i0: int, i1: int, cum: np.ndarray = cum) -> float:
            """Therms over days (i0, i1]."""
            return float(cum[i1 + 1] - cum[i0 + 1])

        if int(home_id) in short:
            idx = event_idx - int(rng.integers(130, 191))
        else:
            idx = first_read_idx + int(rng.integers(0, cfg.read_gap_high + 1))
        read_idx = [idx]
        while True:
            idx += int(rng.integers(cfg.read_gap_low, cfg.read_gap_high + 1))
            if idx > last_read_idx:
                break
            read_idx.append(idx)
        n_reads = len(read_idx)
        estimated = rng.random(n_reads) < cfg.estimated_share
        estimated[0] = estimated[-1] = False
        swap_at = -1
        if int(home_id) in swap:
            candidates = [k for k in range(4, n_reads - 4) if not estimated[k]]
            swap_at = int(rng.choice(candidates))
        meter = float(rng.uniform(3000.0, 20000.0))
        shown = meter
        offset_meter = 0.0
        readings = [shown]
        for k in range(1, n_reads):
            noisy = used(read_idx[k - 1], read_idx[k]) * (1.0 + cfg.read_noise * rng.normal())
            prev_meter = meter
            meter += noisy
            if estimated[k]:
                shown += used(read_idx[k - 1] - 365, read_idx[k] - 365)
            else:
                if k == swap_at:
                    offset_meter = float(rng.uniform(0.0, 50.0)) - prev_meter
                shown = meter + offset_meter
            readings.append(shown)
        n_estimated += int(estimated.sum())
        for k in range(n_reads):
            read_rows.append(
                (
                    int(home_id),
                    (day0 + pd.Timedelta(days=read_idx[k])).date(),
                    round(readings[k], 2),
                    bool(estimated[k]),
                )
            )
        truth_rows.append((int(home_id), is_participant, base, alpha, beta, int(home_id) in swap))

    homes = pd.DataFrame({"home_id": ids, "grp": groups, "event_date": event_dates})
    reads = pd.DataFrame(read_rows, columns=["home_id", "read_date", "reading", "is_estimated"])
    normals = normal_annual_hdd(weather, np.arange(cfg.base_low, cfg.base_high + 1))
    truth_homes = pd.DataFrame(
        truth_rows,
        columns=["home_id", "is_participant", "true_base", "alpha", "beta", "meter_swap"],
    )
    truth_homes["short_history"] = truth_homes["home_id"].isin(short)
    truth_homes["normal_hdd"] = truth_homes["true_base"].map(lambda b: normals[int(b)])
    truth_homes["nac_pre"] = (
        365.0 * truth_homes["alpha"] + truth_homes["beta"] * truth_homes["normal_hdd"]
    )
    cut = np.where(truth_homes["is_participant"], cfg.retrofit_beta_cut, 0.0)
    truth_homes["nac_post"] = (
        truth_homes["nac_pre"] - cut * truth_homes["beta"] * truth_homes["normal_hdd"]
    )
    truth_homes["true_savings_pct"] = (
        100.0 * (truth_homes["nac_pre"] - truth_homes["nac_post"]) / truth_homes["nac_pre"]
    )
    part = truth_homes[truth_homes["is_participant"]]
    window = (pd.to_datetime(weather["date"]) >= pd.Timestamp(cfg.mild_winter_start)) & (
        pd.to_datetime(weather["date"]) <= pd.Timestamp(cfg.mild_winter_end)
    )
    truth = {
        "seed": cfg.seed,
        "programme_true_savings_pct": float(
            100.0 * (part["nac_pre"] - part["nac_post"]).sum() / part["nac_pre"].sum()
        ),
        "mild_winter_offset_f": float(offset),
        "mild_winter_hdd65": _hdd_sum(temps[window.to_numpy()], HDD_REFERENCE_BASE),
        "n_reads": len(reads),
        "n_estimated_reads": n_estimated,
        "meter_swap_homes": sorted(swap),
        "short_history_homes": sorted(short),
    }
    return SynthData(weather, homes, reads, truth_homes, truth)


def write(data: SynthData, out_dir: Path | str) -> None:
    """Write the input CSVs the pipeline loads, and the truth files kept apart from them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.weather.to_csv(out / "weather.csv", index=False)
    data.homes.to_csv(out / "homes.csv", index=False)
    data.reads.to_csv(out / "meter_reads.csv", index=False)
    data.truth_homes.to_csv(out / "truth_homes.csv", index=False)
    (out / "truth.json").write_text(json.dumps(data.truth, indent=2) + "\n")
