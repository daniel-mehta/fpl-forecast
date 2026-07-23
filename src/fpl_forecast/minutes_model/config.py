from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fpl_forecast.config import PROJECT_ROOT


MINUTES_REPORTS_DIR = PROJECT_ROOT / "reports" / "minutes_backtests"
CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class MinutesModelConfig:
    model_names: tuple[str, ...]
    reference_model: str
    state_order: tuple[str, ...]
    state_minutes: dict[str, float]
    ewma_alpha: float
    shrink_matches: float
    softmax_l2: float
    softmax_learning_rate: float
    softmax_iterations: int
    max_model_training_rows: int
    ensemble_min_leaf: int
    candidate_rule: dict[str, str]
    calibration_bins: tuple[float, ...]
    bootstrap_samples: int
    bootstrap_seed: int
    lineup_adjustment_min_candidates: int
    forbidden_frozen_columns: tuple[str, ...]


def load_minutes_config(path: Path = CONFIG_PATH) -> MinutesModelConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MinutesModelConfig(
        model_names=tuple(data["model_names"]),
        reference_model=str(data["reference_model"]),
        state_order=tuple(data["state_order"]),
        state_minutes={str(key): float(value) for key, value in data["state_minutes"].items()},
        ewma_alpha=float(data["ewma_alpha"]),
        shrink_matches=float(data["shrink_matches"]),
        softmax_l2=float(data["softmax_l2"]),
        softmax_learning_rate=float(data["softmax_learning_rate"]),
        softmax_iterations=int(data["softmax_iterations"]),
        max_model_training_rows=int(data["max_model_training_rows"]),
        ensemble_min_leaf=int(data["ensemble_min_leaf"]),
        candidate_rule={str(key): str(value) for key, value in data["candidate_rule"].items()},
        calibration_bins=tuple(float(value) for value in data["calibration_bins"]),
        bootstrap_samples=int(data["bootstrap_samples"]),
        bootstrap_seed=int(data["bootstrap_seed"]),
        lineup_adjustment_min_candidates=int(data["lineup_adjustment_min_candidates"]),
        forbidden_frozen_columns=tuple(data["forbidden_frozen_columns"]),
    )
