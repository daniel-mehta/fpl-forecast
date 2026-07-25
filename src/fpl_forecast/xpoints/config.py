from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fpl_forecast.config import PROJECT_ROOT


XPOINTS_REPORTS_DIR = PROJECT_ROOT / "reports" / "xpoints_backtests"
CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class XPointsConfig:
    rules_version: str
    simulation_version: str
    architecture_type: str
    model_contract_version: str
    draw_count: int
    random_seed: int
    seed_derivation_policy: str
    mean_median_abs_tolerance: float
    mean_p95_abs_tolerance: float
    probability_p95_abs_tolerance: float
    component_reconciliation_tolerance: float
    reference_phase3_baseline: str
    default_model: str
    model_names: tuple[str, ...]
    minutes_runs: dict[str, str]
    team_runs: dict[str, str]
    phase3_runs: dict[str, str]
    component_shrink_minutes: float
    rare_event_shrink_appearances: float
    conditional_points_prior_strength: float
    conditional_points_global_prior: float
    assisted_goal_prior: float
    position_goal_prior_per90: dict[str, float]
    position_assist_prior_per90: dict[str, float]
    position_bps_prior_per90: dict[str, float]
    point_support_min: int
    point_support_max: int
    probability_clip: float
    bootstrap_samples: int
    bootstrap_seed: int
    forbidden_frozen_columns: tuple[str, ...]


def load_xpoints_config(path: Path = CONFIG_PATH) -> XPointsConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return XPointsConfig(
        rules_version=str(data["rules_version"]),
        simulation_version=str(data["simulation_version"]),
        architecture_type=str(data["architecture_type"]),
        model_contract_version=str(data["model_contract_version"]),
        draw_count=int(data["draw_count"]),
        random_seed=int(data["random_seed"]),
        seed_derivation_policy=str(data["seed_derivation_policy"]),
        mean_median_abs_tolerance=float(data["mean_median_abs_tolerance"]),
        mean_p95_abs_tolerance=float(data["mean_p95_abs_tolerance"]),
        probability_p95_abs_tolerance=float(data["probability_p95_abs_tolerance"]),
        component_reconciliation_tolerance=float(data["component_reconciliation_tolerance"]),
        reference_phase3_baseline=str(data["reference_phase3_baseline"]),
        default_model=str(data["default_model"]),
        model_names=tuple(data["model_names"]),
        minutes_runs={str(key): str(value) for key, value in data["minutes_runs"].items()},
        team_runs={str(key): str(value) for key, value in data["team_runs"].items()},
        phase3_runs={str(key): str(value) for key, value in data["phase3_runs"].items()},
        component_shrink_minutes=float(data["component_shrink_minutes"]),
        rare_event_shrink_appearances=float(data["rare_event_shrink_appearances"]),
        conditional_points_prior_strength=float(data.get("conditional_points_prior_strength", 5.0)),
        conditional_points_global_prior=float(data.get("conditional_points_global_prior", 2.0)),
        assisted_goal_prior=float(data["assisted_goal_prior"]),
        position_goal_prior_per90={
            str(key): float(value) for key, value in data["position_goal_prior_per90"].items()
        },
        position_assist_prior_per90={
            str(key): float(value) for key, value in data["position_assist_prior_per90"].items()
        },
        position_bps_prior_per90={
            str(key): float(value) for key, value in data["position_bps_prior_per90"].items()
        },
        point_support_min=int(data["point_support_min"]),
        point_support_max=int(data["point_support_max"]),
        probability_clip=float(data["probability_clip"]),
        bootstrap_samples=int(data["bootstrap_samples"]),
        bootstrap_seed=int(data["bootstrap_seed"]),
        forbidden_frozen_columns=tuple(str(value) for value in data["forbidden_frozen_columns"]),
    )
