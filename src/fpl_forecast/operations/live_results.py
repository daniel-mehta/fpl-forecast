from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


COMPONENTS = {
    "minutes": "minutes",
    "goals_scored": "goals",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "saves": "saves",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "own_goals": "own_goals",
    "bonus": "bonus",
    "bps": "bps",
    "total_points": "total_points",
}


def normalize_event_live(
    *,
    season: str,
    gameweek: int,
    payload: dict[str, Any],
    retrieved_at: str,
    raw_snapshot_path: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        player_id = element.get("id")
        for fixture_block in element.get("explain", []) or []:
            fixture_id = fixture_block.get("fixture")
            row = {
                "season": season,
                "gameweek": gameweek,
                "player_id": player_id,
                "fixture_id": fixture_id,
                "exact_start": pd.NA,
                "source_available_time": retrieved_at,
                "source_available_method": "official_event_live_retrieved_after_fixture_final",
                "retrieved_at": retrieved_at,
                "raw_snapshot_path": raw_snapshot_path,
            }
            for column in COMPONENTS:
                row[column] = 0
            for stat in _stats(fixture_block.get("stats")):
                identifier = stat.get("identifier")
                if identifier in COMPONENTS:
                    row[COMPONENTS[identifier]] = stat.get("value", 0)
                if identifier == "total_points":
                    row["total_points"] = stat.get("points", stat.get("value", 0))
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "gameweek",
                "player_id",
                "fixture_id",
                *COMPONENTS,
                "exact_start",
                "source_available_time",
                "source_available_method",
                "retrieved_at",
                "raw_snapshot_path",
            ]
        )
    return frame.drop_duplicates(["season", "gameweek", "player_id", "fixture_id"], keep="last")


def finalized_fixture_ids(fixtures: Iterable[dict[str, Any]]) -> set[int]:
    return {
        int(fixture["id"])
        for fixture in fixtures
        if fixture.get("finished") and fixture.get("finished_provisional") and fixture.get("id") is not None
    }


def _stats(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []
