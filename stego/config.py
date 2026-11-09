"""Experiment settings. Every setting here changes behaviour; unknown keys are errors."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

DATASETS = ("voc2007", "voc2012", "coco2014", "synthetic")
SSIM_MODES = ("global", "windowed")
CLUSTER_ALGORITHMS = ("kmeans", "minibatch")


class ConfigError(ValueError):
    """Raised for an invalid experiment configuration."""


@dataclass(frozen=True)
class Config:
    dataset: str
    n: int = 6  # bits per secret segment (paper: Example 1)
    M: int = 4  # largest objects used per image (paper: III-A2)
    conf_threshold: float = 0.90  # strictly greater than (paper: III-A2)
    k: int = 256  # clusters (paper: Table V)
    cluster_algorithm: str = "kmeans"
    cluster_max_iter: int = 100  # paper: Table V
    cluster_seed: int = 0
    hash_threshold: float = 0.5  # paper: Eq. 1
    ssim_mode: str = "global"  # "global" is Eq. 2 as written
    ssim_size: int = 256  # comparison size in pixels; the paper does not state one
    capacity_bytes: tuple = (10, 100, 1024)  # paper: Table VI (1 KB = 1024 B)
    capacity_trials: int = 5  # paper: IV-B1
    robustness_trials: int = 25  # paper: IV-B2
    robustness_segments: int = 16  # the paper does not state a length
    gamma_exponent: float = 0.8  # paper: Table IX "Factor: 0.8"
    seed: int = 0
    bootstrap_samples: int = 1000
    expected_images: int = 0  # 0 means use the dataset's published count

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ConfigError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.ssim_mode not in SSIM_MODES:
            raise ConfigError(f"ssim_mode must be one of {SSIM_MODES}, got {self.ssim_mode!r}")
        if self.cluster_algorithm not in CLUSTER_ALGORITHMS:
            raise ConfigError(
                f"cluster_algorithm must be one of {CLUSTER_ALGORITHMS}, got {self.cluster_algorithm!r}"
            )
        if not 1 <= self.n <= 8:
            raise ConfigError(f"n must be between 1 and 8, got {self.n}")
        if not 1 <= self.M <= 9:
            raise ConfigError(f"M must be between 1 and 9, got {self.M}")
        if not 0.0 <= self.conf_threshold < 1.0:
            raise ConfigError("conf_threshold must be in [0, 1)")
        if self.k < 1:
            raise ConfigError("k must be at least 1")
        if self.capacity_trials < 1 or self.robustness_trials < 1 or self.robustness_segments < 1:
            raise ConfigError("trial and segment counts must be at least 1")
        if not self.capacity_bytes or any(int(b) < 1 for b in self.capacity_bytes):
            raise ConfigError("capacity_bytes must be positive")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["capacity_bytes"] = list(self.capacity_bytes)
        return data

    @property
    def tag(self) -> str:
        """Short name used in run folders."""
        return f"{self.dataset}_n{self.n}_M{self.M}_seed{self.seed}"


def _coerce(name: str, value):
    target = {f.name: f.type for f in fields(Config)}[name]
    try:
        if target == "int":
            if isinstance(value, bool) or float(value) != int(float(value)):
                raise ValueError
            return int(float(value))
        if target == "float":
            return float(value)
        if target == "str":
            return str(value)
        if target == "tuple":
            if isinstance(value, str):
                value = ast.literal_eval(value)
            if isinstance(value, int):
                value = [value]
            return tuple(int(v) for v in value)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise ConfigError(f"Invalid value for {name}: {value!r}") from exc
    raise ConfigError(f"Unsupported field type for {name}")


def build_config(values: dict) -> Config:
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(f"Unknown setting(s): {', '.join(unknown)}. Valid settings: {', '.join(sorted(known))}")
    if "dataset" not in values:
        raise ConfigError("The setting 'dataset' is required")
    return Config(**{key: _coerce(key, val) for key, val in values.items()})


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ConfigError(f"Config file must contain key: value lines: {path}")
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        values[key.strip()] = val.strip()
    return build_config(values)
