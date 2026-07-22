from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.panel.common import load_historical_player_fixtures, phase2_dir, write_parquet


@dataclass(frozen=True)
class FactBuildResult:
    fact_path: Path
    fact: pd.DataFrame


def build_fact_player_fixture(
    *,
    seasons: list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> FactBuildResult:
    output_dir = phase2_dir(normalized_dir)
    player_map_path = output_dir / "player_season_map.parquet"
    team_map_path = output_dir / "team_season_map.parquet"
    fixture_path = output_dir / "dim_fixture.parquet"
    for path in (player_map_path, team_map_path, fixture_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 2 dependency: {path}")
    history = load_historical_player_fixtures(normalized_dir=normalized_dir, seasons=seasons)
    player_map = pd.read_parquet(player_map_path)
    team_map = pd.read_parquet(team_map_path)
    fixtures = pd.read_parquet(fixture_path)
    fact = build_fact_player_fixture_from_frames(history, player_map, team_map, fixtures)
    path = write_parquet(fact, output_dir / "fact_player_fixture.parquet")
    return FactBuildResult(path, fact)


def build_fact_player_fixture_from_frames(
    history: pd.DataFrame,
    player_map: pd.DataFrame,
    team_map: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    fact = history.copy()
    fact = fact.merge(
        player_map[
            [
                "season",
                "source_player_id",
                "player_uid",
                "match_method",
                "manual_review_status",
            ]
        ],
        left_on=["season", "player_id"],
        right_on=["season", "source_player_id"],
        how="left",
    )
    if fact["player_uid"].isna().any():
        missing = int(fact["player_uid"].isna().sum())
        raise ValueError(f"{missing} player-fixture rows lack a resolved player_uid.")

    team_lookup = team_map[["season", "source_team_name", "team_uid"]].drop_duplicates()
    fact = fact.merge(
        team_lookup.rename(columns={"source_team_name": "team_name", "team_uid": "player_team_uid"}),
        on=["season", "team_name"],
        how="left",
    )
    fact = fact.merge(
        fixtures[
            [
                "season",
                "source_fixture_id",
                "fixture_key",
                "home_team_uid",
                "away_team_uid",
                "kickoff_time",
                "source_available_time",
                "source_available_method",
            ]
        ].rename(
            columns={
                "source_fixture_id": "fixture_id",
                "kickoff_time": "fixture_kickoff_time",
                "source_available_time": "fixture_source_available_time",
                "source_available_method": "fixture_source_available_method",
            }
        ),
        on=["season", "fixture_id"],
        how="left",
    )
    if fact[["player_team_uid", "fixture_key", "home_team_uid", "away_team_uid"]].isna().any().any():
        raise ValueError("Fact table has missing player, team, or fixture identity joins.")

    fact["opponent_team_uid"] = fact.apply(
        lambda row: row.away_team_uid if bool(row.was_home) else row.home_team_uid,
        axis=1,
    )
    expected_team = fact.apply(
        lambda row: row.home_team_uid if bool(row.was_home) else row.away_team_uid,
        axis=1,
    )
    mismatched_team = fact.loc[fact["player_team_uid"] != expected_team]
    if not mismatched_team.empty:
        raise ValueError(f"{len(mismatched_team)} fact rows have a team/fixture side mismatch.")
    fact["kickoff_time"] = pd.to_datetime(fact["kickoff_time"], utc=True, errors="coerce")
    fact["fixture_kickoff_time"] = pd.to_datetime(
        fact["fixture_kickoff_time"], utc=True, errors="coerce"
    )
    if not (fact["kickoff_time"] == fact["fixture_kickoff_time"]).all():
        raise ValueError("Player rows and fixture dimension disagree on kickoff_time.")

    fact["information_cutoff"] = fact.groupby(["season", "gameweek"])["kickoff_time"].transform("min")
    fact["cutoff_method"] = "inferred_earliest_gameweek_kickoff"
    fact["cutoff_source"] = "historical_player_fixture.kickoff_time"
    fact["cutoff_is_exact"] = False
    fact["source_available_time"] = pd.to_datetime(
        fact["fixture_source_available_time"],
        utc=True,
    )
    fact["source_available_method"] = fact["fixture_source_available_method"]
    fact["entity_type"] = "player"
    fact.loc[
        (pd.to_numeric(fact.get("element_type"), errors="coerce") == 5)
        | (fact.get("fpl_position") == "AM"),
        "entity_type",
    ] = "assistant_manager"
    fact["target_total_points"] = fact["total_points"]

    fact = fact.drop(
        columns=[
            "source_player_id",
            "fixture_kickoff_time",
            "fixture_source_available_time",
            "fixture_source_available_method",
        ]
    )
    duplicate_count = fact.duplicated(["season", "player_uid", "fixture_key"]).sum()
    if duplicate_count:
        raise ValueError(f"fact_player_fixture has {duplicate_count} duplicate key rows.")
    return fact.sort_values(["season", "kickoff_time", "fixture_id", "player_uid"])


def standard_player_fact(fact: pd.DataFrame) -> pd.DataFrame:
    if "entity_type" not in fact.columns:
        raise ValueError("fact_player_fixture requires entity_type for deterministic filtering.")
    return fact.loc[fact["entity_type"] == "player"].copy()
