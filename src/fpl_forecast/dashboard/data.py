from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.operations.config import LATEST_SUCCESSFUL_PATH


FRONTEND_SCHEMA_VERSION = "phase8_frontend_v1"


@dataclass(frozen=True)
class DashboardData:
    run_dir: Path | None
    status: dict
    projections: pd.DataFrame
    squad: pd.DataFrame
    lineup: pd.DataFrame
    comparison: pd.DataFrame
    freshness: dict
    error: str | None = None


def load_dashboard_data(pointer_path: Path = LATEST_SUCCESSFUL_PATH) -> DashboardData:
    if not pointer_path.exists():
        return DashboardData(
            run_dir=None,
            status={"state": "WAITING_FOR_SEASON_LAUNCH", "reason": "No successful operational run yet."},
            projections=pd.DataFrame(),
            squad=pd.DataFrame(),
            lineup=pd.DataFrame(),
            comparison=pd.DataFrame(),
            freshness={},
            error=None,
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    run_dir = Path(pointer["run_dir"])
    try:
        status = json.loads((run_dir / "operational_status.json").read_text(encoding="utf-8"))
        freshness = json.loads((run_dir / "data_freshness.json").read_text(encoding="utf-8"))
        if pointer.get("schema_version") != FRONTEND_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported dashboard schema {pointer.get('schema_version')}; expected {FRONTEND_SCHEMA_VERSION}."
            )
        return DashboardData(
            run_dir=run_dir,
            status=status,
            projections=pd.read_csv(run_dir / "player_gameweek_projections.csv"),
            squad=pd.read_csv(run_dir / "optimized_squad.csv"),
            lineup=pd.read_csv(run_dir / "optimized_lineup.csv"),
            comparison=pd.read_csv(run_dir / "model_comparison.csv"),
            freshness=freshness,
        )
    except Exception as exc:  # noqa: BLE001
        return DashboardData(
            run_dir=run_dir,
            status={"state": "FAILED_USING_LAST_SUCCESS", "reason": str(exc)},
            projections=pd.DataFrame(),
            squad=pd.DataFrame(),
            lineup=pd.DataFrame(),
            comparison=pd.DataFrame(),
            freshness={},
            error=str(exc),
        )
