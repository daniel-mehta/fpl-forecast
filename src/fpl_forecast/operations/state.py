from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from fpl_forecast.operations.config import STATUS_PATH


class OperationalStateName(StrEnum):
    WAITING_FOR_SEASON_LAUNCH = "WAITING_FOR_SEASON_LAUNCH"
    READY_TO_REFRESH = "READY_TO_REFRESH"
    REFRESHING = "REFRESHING"
    NEEDS_RULE_REVIEW = "NEEDS_RULE_REVIEW"
    NEEDS_TEAM_IDENTITY_REVIEW = "NEEDS_TEAM_IDENTITY_REVIEW"
    NEEDS_PLAYER_IDENTITY_REVIEW = "NEEDS_PLAYER_IDENTITY_REVIEW"
    NO_FORECASTABLE_GAMEWEEK = "NO_FORECASTABLE_GAMEWEEK"
    SUCCEEDED = "SUCCEEDED"
    FAILED_USING_LAST_SUCCESS = "FAILED_USING_LAST_SUCCESS"


@dataclass(frozen=True)
class OperationalStatus:
    state: OperationalStateName
    target_season: str
    inferred_official_season: str | None
    checked_at: str
    latest_official_deadline: str | None = None
    latest_completed_gameweek: int | None = None
    latest_completed_fixture: int | None = None
    latest_successful_run_id: str | None = None
    data_age_seconds: float | None = None
    model_age_seconds: float | None = None
    reason: str = ""
    warning: str | None = None
    dashboard_can_display_forecasts: bool = False
    retry_automatically: bool = False
    status_path: str | None = None
    review_artifact: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_status(status: OperationalStatus, path: Path = STATUS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = status.to_dict()
    payload["status_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_status(path: Path = STATUS_PATH) -> OperationalStatus | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["state"] = OperationalStateName(data["state"])
    return OperationalStatus(**{key: value for key, value in data.items() if key != "status_path"})
