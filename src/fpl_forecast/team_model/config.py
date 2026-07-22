from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fpl_forecast.config import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "src" / "fpl_forecast" / "team_model" / "config.json"
TEAM_REPORTS_DIR = PROJECT_ROOT / "reports" / "team_backtests"


@dataclass(frozen=True)
class TeamModelConfig:
    version: str
    t0_min_training_fixtures: int
    t1_recent_window: int
    t1_shrink_matches: int
    t2_ridge_penalty: float
    t2_max_iterations: int
    t2_convergence_tolerance: float
    t2_low_history_threshold: int
    t3_rho_lower_bound: float
    t3_rho_upper_bound: float
    t3_rho_penalty: float
    t3_max_iterations: int
    t3_convergence_tolerance: float
    recency_half_life_days: int
    min_expected_goals: float
    max_expected_goals: float
    probability_max_goals: int
    outcome_grid_max_goals: int
    log_loss_epsilon: float
    bootstrap_samples: int
    bootstrap_seed: int
    reference_model: str
    model_families: dict[str, str]


def load_team_model_config(path: Path = CONFIG_PATH) -> TeamModelConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TeamModelConfig(
        version=str(raw["version"]),
        t0_min_training_fixtures=int(raw["t0_min_training_fixtures"]),
        t1_recent_window=int(raw["t1_recent_window"]),
        t1_shrink_matches=int(raw["t1_shrink_matches"]),
        t2_ridge_penalty=float(raw["t2_ridge_penalty"]),
        t2_max_iterations=int(raw["t2_max_iterations"]),
        t2_convergence_tolerance=float(raw["t2_convergence_tolerance"]),
        t2_low_history_threshold=int(raw["t2_low_history_threshold"]),
        t3_rho_lower_bound=float(raw["t3_rho_lower_bound"]),
        t3_rho_upper_bound=float(raw["t3_rho_upper_bound"]),
        t3_rho_penalty=float(raw["t3_rho_penalty"]),
        t3_max_iterations=int(raw["t3_max_iterations"]),
        t3_convergence_tolerance=float(raw["t3_convergence_tolerance"]),
        recency_half_life_days=int(raw["recency_half_life_days"]),
        min_expected_goals=float(raw["min_expected_goals"]),
        max_expected_goals=float(raw["max_expected_goals"]),
        probability_max_goals=int(raw["probability_max_goals"]),
        outcome_grid_max_goals=int(raw["outcome_grid_max_goals"]),
        log_loss_epsilon=float(raw["log_loss_epsilon"]),
        bootstrap_samples=int(raw["bootstrap_samples"]),
        bootstrap_seed=int(raw["bootstrap_seed"]),
        reference_model=str(raw["reference_model"]),
        model_families={str(key): str(value) for key, value in raw["model_families"].items()},
    )
