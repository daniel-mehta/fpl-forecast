from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fpl_forecast.config import PROJECT_ROOT


DECISION_REPORTS_DIR = PROJECT_ROOT / "reports" / "decision_backtests"
DECISION_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "decisions"
CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class DecisionConfig:
    decision_rules_version: str
    default_model: str
    comparison_models: tuple[str, ...]
    xpoints_runs: dict[str, str]
    squad_candidate_limits: dict[str, int]
    squad_evaluation_limit: int
    lineup_objective: str
    future_discount: float
    risk_mode: str
    near_optimal_gap: float
    transfer_candidate_limit: int
    optimizer_variants: tuple[str, ...]
    expected_realized_search_limit: int
    expected_realized_search_iterations: int
    expected_realized_scenarios: int
    expected_realized_seed: int
    forbidden_frozen_columns: tuple[str, ...]


def load_decision_config(path: Path = CONFIG_PATH) -> DecisionConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return DecisionConfig(
        decision_rules_version=str(data["decision_rules_version"]),
        default_model=str(data["default_model"]),
        comparison_models=tuple(str(value) for value in data["comparison_models"]),
        xpoints_runs={str(key): str(value) for key, value in data["xpoints_runs"].items()},
        squad_candidate_limits={str(key): int(value) for key, value in data["squad_candidate_limits"].items()},
        squad_evaluation_limit=int(data["squad_evaluation_limit"]),
        lineup_objective=str(data["lineup_objective"]),
        future_discount=float(data["future_discount"]),
        risk_mode=str(data["risk_mode"]),
        near_optimal_gap=float(data["near_optimal_gap"]),
        transfer_candidate_limit=int(data["transfer_candidate_limit"]),
        optimizer_variants=tuple(str(value) for value in data.get("optimizer_variants", ["D1_MEAN_ONLY_MILP"])),
        expected_realized_search_limit=int(data.get("expected_realized_search_limit", 0)),
        expected_realized_search_iterations=int(data.get("expected_realized_search_iterations", 3)),
        expected_realized_scenarios=int(data.get("expected_realized_scenarios", 32768)),
        expected_realized_seed=int(data.get("expected_realized_seed", 9113)),
        forbidden_frozen_columns=tuple(str(value) for value in data["forbidden_frozen_columns"]),
    )
