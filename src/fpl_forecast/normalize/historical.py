from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, RAW_VAASTAV_DIR
from fpl_forecast.ingest.vaastav import MERGED_GW, PLAYERS_RAW, load_latest_vaastav_csv

HISTORICAL_PLAYER_FIXTURES = "historical_player_fixtures.parquet"
LEGACY_HISTORICAL_PLAYER_GAMEWEEKS = "historical_player_gameweeks.parquet"


def normalize_historical(
    *,
    season: str,
    raw_dir=RAW_VAASTAV_DIR,
    normalized_dir=NORMALIZED_DIR,
) -> list[Path]:
    merged, merged_metadata, merged_raw_path = load_latest_vaastav_csv(
        raw_dir=raw_dir,
        season=season,
        dataset_name=MERGED_GW,
    )
    players, _, _ = load_latest_vaastav_csv(
        raw_dir=raw_dir,
        season=season,
        dataset_name=PLAYERS_RAW,
    )
    output_dir = Path(normalized_dir) / season
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = _historical_player_fixtures_frame(
        merged,
        players,
        metadata=merged_metadata,
        raw_path=merged_raw_path,
        season=season,
    )
    output = output_dir / HISTORICAL_PLAYER_FIXTURES
    normalized.to_parquet(output, index=False)
    legacy_output = output_dir / LEGACY_HISTORICAL_PLAYER_GAMEWEEKS
    if legacy_output.exists():
        legacy_output.unlink()
    return [output]


def _historical_player_fixtures_frame(
    merged: pd.DataFrame,
    players: pd.DataFrame,
    *,
    metadata: dict[str, str],
    raw_path: str,
    season: str,
) -> pd.DataFrame:
    for column in ["element", "fixture", "round", "minutes", "total_points"]:
        if column not in merged.columns:
            raise ValueError(f"Historical merged_gw.csv is missing required column: {column}")

    frame = merged.copy()
    source_position = frame.get("position", pd.Series(pd.NA, index=frame.index)).replace({"GK": "GKP"})

    player_columns = [column for column in ["id", "code", "element_type"] if column in players.columns]
    player_lookup = pd.DataFrame(columns=["id", "code", "element_type"])
    if {"id", "code"}.issubset(players.columns):
        player_lookup = players[player_columns].drop_duplicates("id")

    frame = frame.merge(
        player_lookup,
        how="left",
        left_on="element",
        right_on="id",
        suffixes=("", "_player"),
    )
    element_type = pd.to_numeric(
        frame.get("element_type", pd.Series(pd.NA, index=frame.index)),
        errors="coerce",
    ).astype("Int64")
    fpl_position = element_type.map({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD", 5: "AM"}).astype(
        "string"
    )

    output = pd.DataFrame(
        {
            "player_id": pd.to_numeric(frame["element"], errors="coerce").astype("Int64"),
            "player_code": pd.to_numeric(frame["code"], errors="coerce").astype("Int64"),
            "fixture_id": pd.to_numeric(frame["fixture"], errors="coerce").astype("Int64"),
            "gameweek": pd.to_numeric(frame["round"], errors="coerce").astype("Int64"),
            "player_name": frame.get("name", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "team_name": frame.get("team", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "source_position": source_position.astype("string"),
            "element_type": element_type,
            "fpl_position": fpl_position,
            "kickoff_time": frame.get("kickoff_time", pd.Series(pd.NA, index=frame.index)).astype(
                "string"
            ),
            "minutes": pd.to_numeric(frame["minutes"], errors="coerce").astype("Int64"),
            "total_points": pd.to_numeric(frame["total_points"], errors="coerce").astype("Int64"),
            "price_tenths": pd.to_numeric(
                frame.get("value", pd.Series(pd.NA, index=frame.index)),
                errors="coerce",
            ).astype("Int64"),
            "was_home": frame.get("was_home", pd.Series(pd.NA, index=frame.index)).astype("boolean"),
            "opponent_team": pd.to_numeric(
                frame.get("opponent_team", pd.Series(pd.NA, index=frame.index)),
                errors="coerce",
            ).astype("Int64"),
        }
    )

    for component_column in (
        "assists",
        "bonus",
        "bps",
        "clean_sheets",
        "goals_conceded",
        "goals_scored",
        "own_goals",
        "penalties_missed",
        "penalties_saved",
        "red_cards",
        "saves",
        "starts",
        "yellow_cards",
        "mng_clean_sheets",
        "mng_draw",
        "mng_goals_scored",
        "mng_loss",
        "mng_underdog_draw",
        "mng_underdog_win",
        "mng_win",
    ):
        if component_column in frame.columns:
            output[component_column] = pd.to_numeric(frame[component_column], errors="coerce")

    for optional_column in (
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ):
        if optional_column in frame.columns:
            output[optional_column] = pd.to_numeric(frame[optional_column], errors="coerce")

    output["source"] = metadata["source"]
    output["source_version"] = metadata.get("source_version") or metadata["sha256"]
    output["retrieved_at"] = metadata["retrieved_at"]
    output["season"] = season
    output["raw_snapshot_path"] = raw_path
    return output
