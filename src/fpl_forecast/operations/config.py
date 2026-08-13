from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fpl_forecast.config import OUTPUTS_DIR


OPERATIONAL_OUTPUT_DIR = OUTPUTS_DIR / "operational"
OPERATIONAL_RUNS_DIR = OPERATIONAL_OUTPUT_DIR / "runs"
OPERATIONAL_FAILED_DIR = OPERATIONAL_OUTPUT_DIR / "failed"
LATEST_SUCCESSFUL_PATH = OPERATIONAL_OUTPUT_DIR / "latest_successful.json"
STATUS_PATH = OPERATIONAL_OUTPUT_DIR / "operational_status.json"
LOCK_PATH = OPERATIONAL_OUTPUT_DIR / "refresh.lock"
CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class OperationalConfig:
    schema_version: str
    default_target_season: str
    cooldown_seconds: int
    frontend_schema_version: str
    default_models: dict[str, str]
    current_result_finalization: dict[str, object]


@dataclass(frozen=True)
class OperationalPaths:
    output_dir: Path
    runs_dir: Path
    failed_dir: Path
    latest_successful_path: Path
    status_path: Path
    lock_path: Path


def load_operational_config(path: Path = CONFIG_PATH) -> OperationalConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return OperationalConfig(
        schema_version=str(data["schema_version"]),
        default_target_season=str(data["default_target_season"]),
        cooldown_seconds=int(data["cooldown_seconds"]),
        frontend_schema_version=str(data["frontend_schema_version"]),
        default_models={str(key): str(value) for key, value in data["default_models"].items()},
        current_result_finalization=dict(data["current_result_finalization"]),
    )


def resolve_operational_paths(root: Path | None = None) -> OperationalPaths:
    output_dir = (root or OPERATIONAL_OUTPUT_DIR).resolve()
    return OperationalPaths(
        output_dir=output_dir,
        runs_dir=output_dir / "runs",
        failed_dir=output_dir / "failed",
        latest_successful_path=output_dir / "latest_successful.json",
        status_path=output_dir / "operational_status.json",
        lock_path=output_dir / "refresh.lock",
    )
