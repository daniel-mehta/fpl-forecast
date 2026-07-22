from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fpl_forecast.config import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "src" / "fpl_forecast" / "backtest" / "config.json"
REPORTS_DIR = PROJECT_ROOT / "reports" / "backtests"


@dataclass(frozen=True)
class BacktestConfig:
    version: str
    rolling_windows: tuple[int, ...]
    points_per_90_prior_matches: int
    minutes_cap: int
    missing_history_fallback: str
    candidate_pool_rule: str
    population_rules: dict[str, str]
    b6_formula: str
    bootstrap_samples: int
    bootstrap_seed: int
    minimum_training_rows: int
    reference_baseline: str
    forbidden_fields: tuple[str, ...]


def load_backtest_config(path: Path = CONFIG_PATH) -> BacktestConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return BacktestConfig(
        version=str(raw["version"]),
        rolling_windows=tuple(int(value) for value in raw["rolling_windows"]),
        points_per_90_prior_matches=int(raw["points_per_90_prior_matches"]),
        minutes_cap=int(raw["minutes_cap"]),
        missing_history_fallback=str(raw["missing_history_fallback"]),
        candidate_pool_rule=str(raw["candidate_pool_rule"]),
        population_rules={str(key): str(value) for key, value in raw["population_rules"].items()},
        b6_formula=str(raw["b6_formula"]),
        bootstrap_samples=int(raw["bootstrap_samples"]),
        bootstrap_seed=int(raw["bootstrap_seed"]),
        minimum_training_rows=int(raw["minimum_training_rows"]),
        reference_baseline=str(raw["reference_baseline"]),
        forbidden_fields=tuple(str(value) for value in raw["forbidden_fields"]),
    )
