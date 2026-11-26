"""Typed access to config.yaml."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"none of these paths exist: {[str(c) for c in candidates]}")


def default_config_path() -> Path:
    """config.yaml from the working directory, the repository root, or the installed package."""
    return _first_existing(
        [Path("config.yaml"), _PACKAGE_DIR.parents[1] / "config.yaml", _PACKAGE_DIR / "config.yaml"]
    )


def default_sql_dir() -> Path:
    """The sql/ directory of a repository checkout, or the copy shipped inside the wheel."""
    return _first_existing([_PACKAGE_DIR.parents[1] / "sql", _PACKAGE_DIR / "sql"])


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 7
    n_participants: int = 300
    n_comparison: int = 300
    weather_start: date = date(2016, 1, 1)
    weather_end: date = date(2025, 12, 31)
    reads_start: date = date(2023, 1, 1)
    reads_end: date = date(2025, 12, 31)
    retrofit_first: date = date(2024, 6, 1)
    retrofit_last: date = date(2024, 8, 31)
    mild_winter_start: date = date(2024, 10, 1)
    mild_winter_end: date = date(2025, 4, 30)
    mild_winter_hdd_cut: float = 0.15
    mean_temp_f: float = 50.0
    amplitude_f: float = 22.0
    ar_phi: float = 0.7
    ar_sigma_f: float = 4.5
    base_low: int = 55
    base_high: int = 65
    alpha_low: float = 0.8
    alpha_high: float = 2.5
    beta_low: float = 0.10
    beta_high: float = 0.25
    retrofit_beta_cut: float = 0.18
    read_gap_low: int = 27
    read_gap_high: int = 35
    read_noise: float = 0.03
    estimated_share: float = 0.08
    n_meter_swap: int = 6
    n_short_history: int = 12


@dataclass(frozen=True)
class PipelineParams:
    min_days: int = 20
    max_days: int = 40
    min_periods: int = 9
    base_min: int = 50
    base_max: int = 70
    fixed_base: int = 65
    naive_min_periods: int = 6

    def sql_values(self, fixed_base: int | None = None) -> dict[str, int]:
        """Values for the ${name} placeholders; a fixed base collapses the candidate range."""
        values = {
            "min_days": self.min_days,
            "max_days": self.max_days,
            "min_periods": self.min_periods,
            "base_min": self.base_min,
            "base_max": self.base_max,
            "naive_min_periods": self.naive_min_periods,
        }
        if fixed_base is not None:
            values["base_min"] = fixed_base
            values["base_max"] = fixed_base
        return values


@dataclass(frozen=True)
class BootstrapConfig:
    resamples: int = 500
    seed: int = 11
    ci: float = 0.95


@dataclass(frozen=True)
class Config:
    synth: SynthConfig = field(default_factory=SynthConfig)
    pipeline: PipelineParams = field(default_factory=PipelineParams)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: Path | str | None = None) -> Config:
    """Read config.yaml; missing sections fall back to the defaults above."""
    config_path = Path(path) if path else default_config_path()
    raw = yaml.safe_load(config_path.read_text()) or {}
    return Config(
        synth=_build(SynthConfig, raw.get("synth")),
        pipeline=_build(PipelineParams, raw.get("pipeline")),
        bootstrap=_build(BootstrapConfig, raw.get("bootstrap")),
    )
