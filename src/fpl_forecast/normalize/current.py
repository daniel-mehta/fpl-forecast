from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, POSITION_BY_ID, RAW_FPL_API_DIR
from fpl_forecast.ingest.fpl_api import BOOTSTRAP_STATIC, FIXTURES, load_latest_fpl_snapshot
from fpl_forecast.ingest.season import infer_and_validate_current_season


def normalize_current(
    *,
    season: str,
    raw_dir=RAW_FPL_API_DIR,
    normalized_dir=NORMALIZED_DIR,
) -> list[Path]:
    bootstrap = load_latest_fpl_snapshot(
        raw_dir=raw_dir,
        season=season,
        endpoint_name=BOOTSTRAP_STATIC,
    )
    fixtures = load_latest_fpl_snapshot(
        raw_dir=raw_dir,
        season=season,
        endpoint_name=FIXTURES,
    )
    season_identity = infer_and_validate_current_season(
        requested_season=season,
        bootstrap_payload=bootstrap.payload,
        fixtures_payload=fixtures.payload,
    )

    output_dir = Path(normalized_dir) / season
    output_dir.mkdir(parents=True, exist_ok=True)

    players = _players_frame(
        bootstrap.payload,
        metadata=bootstrap.metadata,
        raw_path=bootstrap.raw_path,
        season=season_identity.inferred_season,
    )
    teams = _teams_frame(
        bootstrap.payload,
        metadata=bootstrap.metadata,
        raw_path=bootstrap.raw_path,
        season=season_identity.inferred_season,
    )
    fixtures_frame = _fixtures_frame(
        fixtures.payload,
        metadata=fixtures.metadata,
        raw_path=fixtures.raw_path,
        season=season_identity.inferred_season,
    )

    outputs = [
        output_dir / "current_players.parquet",
        output_dir / "current_teams.parquet",
        output_dir / "current_fixtures.parquet",
    ]
    players.to_parquet(outputs[0], index=False)
    teams.to_parquet(outputs[1], index=False)
    fixtures_frame.to_parquet(outputs[2], index=False)
    return outputs


def _provenance_columns(
    *,
    metadata: dict[str, Any],
    raw_path: str,
    season: str,
) -> dict[str, Any]:
    return {
        "source": metadata["source"],
        "source_version": metadata.get("source_version") or metadata["sha256"],
        "retrieved_at": metadata["retrieved_at"],
        "season": season,
        "raw_snapshot_path": raw_path,
    }


def _players_frame(
    payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
    raw_path: str,
    season: str,
) -> pd.DataFrame:
    rows = payload.get("elements", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_code",
                "first_name",
                "second_name",
                "web_name",
                "team_id",
                "position_id",
                "position",
                "price_tenths",
                "status",
                "minutes",
                "total_points",
                "source",
                "source_version",
                "retrieved_at",
                "season",
                "raw_snapshot_path",
            ]
        )

    frame = _ensure_columns(
        frame,
        [
            "id",
            "code",
            "first_name",
            "second_name",
            "web_name",
            "team",
            "element_type",
            "now_cost",
            "status",
            "minutes",
            "total_points",
        ],
    )
    normalized = pd.DataFrame(
        {
            "player_id": pd.to_numeric(frame["id"], errors="coerce").astype("Int64"),
            "player_code": pd.to_numeric(frame["code"], errors="coerce").astype("Int64"),
            "first_name": frame["first_name"].astype("string"),
            "second_name": frame["second_name"].astype("string"),
            "web_name": frame["web_name"].astype("string"),
            "team_id": pd.to_numeric(frame["team"], errors="coerce").astype("Int64"),
            "position_id": pd.to_numeric(frame["element_type"], errors="coerce").astype("Int64"),
            "price_tenths": pd.to_numeric(frame["now_cost"], errors="coerce").astype("Int64"),
            "status": frame["status"].astype("string"),
            "minutes": pd.to_numeric(frame["minutes"], errors="coerce").astype("Int64"),
            "total_points": pd.to_numeric(frame["total_points"], errors="coerce").astype("Int64"),
        }
    )
    normalized["position"] = normalized["position_id"].map(POSITION_BY_ID).astype("string")
    for key, value in _provenance_columns(metadata=metadata, raw_path=raw_path, season=season).items():
        normalized[key] = value
    return normalized


def _teams_frame(
    payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
    raw_path: str,
    season: str,
) -> pd.DataFrame:
    rows = payload.get("teams", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "team_id",
                "team_code",
                "team_name",
                "short_name",
                "strength",
                "source",
                "source_version",
                "retrieved_at",
                "season",
                "raw_snapshot_path",
            ]
        )
    frame = _ensure_columns(frame, ["id", "code", "name", "short_name", "strength"])
    normalized = pd.DataFrame(
        {
            "team_id": pd.to_numeric(frame["id"], errors="coerce").astype("Int64"),
            "team_code": pd.to_numeric(frame["code"], errors="coerce").astype("Int64"),
            "team_name": frame["name"].astype("string"),
            "short_name": frame["short_name"].astype("string"),
            "strength": pd.to_numeric(frame["strength"], errors="coerce").astype("Int64"),
        }
    )
    for key, value in _provenance_columns(metadata=metadata, raw_path=raw_path, season=season).items():
        normalized[key] = value
    return normalized


def _fixtures_frame(
    payload: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    raw_path: str,
    season: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "fixture_code",
                "gameweek",
                "home_team_id",
                "away_team_id",
                "kickoff_time",
                "finished",
                "started",
                "team_h_score",
                "team_a_score",
                "source",
                "source_version",
                "retrieved_at",
                "season",
                "raw_snapshot_path",
            ]
        )
    frame = _ensure_columns(
        frame,
        [
            "id",
            "code",
            "event",
            "team_h",
            "team_a",
            "kickoff_time",
            "finished",
            "started",
            "team_h_score",
            "team_a_score",
        ],
    )
    normalized = pd.DataFrame(
        {
            "fixture_id": pd.to_numeric(frame["id"], errors="coerce").astype("Int64"),
            "fixture_code": pd.to_numeric(frame["code"], errors="coerce").astype("Int64"),
            "gameweek": pd.to_numeric(frame["event"], errors="coerce").astype("Int64"),
            "home_team_id": pd.to_numeric(frame["team_h"], errors="coerce").astype("Int64"),
            "away_team_id": pd.to_numeric(frame["team_a"], errors="coerce").astype("Int64"),
            "kickoff_time": frame["kickoff_time"].astype("string"),
            "finished": frame["finished"].astype("boolean"),
            "started": frame["started"].astype("boolean"),
            "team_h_score": pd.to_numeric(frame["team_h_score"], errors="coerce").astype("Int64"),
            "team_a_score": pd.to_numeric(frame["team_a_score"], errors="coerce").astype("Int64"),
        }
    )
    for key, value in _provenance_columns(metadata=metadata, raw_path=raw_path, season=season).items():
        normalized[key] = value
    return normalized


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame
