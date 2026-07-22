from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.panel.availability import (
    CONSERVATIVE_AVAILABILITY_METHOD,
    match_result_available_time,
)
from fpl_forecast.panel.common import load_historical_player_fixtures, phase2_dir, write_parquet


@dataclass(frozen=True)
class FixtureBuildResult:
    fixture_path: Path
    fixtures: pd.DataFrame


def build_fixture_dimension(
    *,
    seasons: list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> FixtureBuildResult:
    history = load_historical_player_fixtures(normalized_dir=normalized_dir, seasons=seasons)
    team_map_path = phase2_dir(normalized_dir) / "team_season_map.parquet"
    if not team_map_path.exists():
        raise FileNotFoundError(f"Missing team identity map: {team_map_path}")
    team_map = pd.read_parquet(team_map_path)
    fixtures = build_fixture_dimension_from_frames(history, team_map)
    output = write_parquet(fixtures, phase2_dir(normalized_dir) / "dim_fixture.parquet")
    return FixtureBuildResult(output, fixtures)


def build_fixture_dimension_from_frames(history: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "fixture_id",
        "gameweek",
        "team_name",
        "was_home",
        "opponent_team",
        "kickoff_time",
        "source_version",
        "raw_snapshot_path",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Historical data missing fixture columns: {', '.join(sorted(missing))}")

    team_lookup = team_map[["season", "source_team_name", "team_uid"]].drop_duplicates()
    rows: list[dict[str, object]] = []
    for (season, fixture_id), frame in history.groupby(["season", "fixture_id"], dropna=False):
        sides = (
            frame[["team_name", "was_home", "opponent_team"]]
            .dropna(subset=["team_name", "was_home"])
            .drop_duplicates()
        )
        home = sides.loc[sides["was_home"] == True]  # noqa: E712
        away = sides.loc[sides["was_home"] == False]  # noqa: E712
        if len(home) != 1 or len(away) != 1:
            raise ValueError(f"Fixture {season}/{fixture_id} does not have one home and one away team.")
        base = frame.iloc[0]
        rows.append(
            {
                "season": season,
                "source_fixture_id": int(fixture_id),
                "fixture_key": f"{season}:{int(fixture_id)}",
                "gameweek": int(base["gameweek"]),
                "home_team_name": home.iloc[0]["team_name"],
                "away_team_name": away.iloc[0]["team_name"],
                "kickoff_time": base["kickoff_time"],
                "source_available_time": match_result_available_time(base["kickoff_time"]),
                "source_available_method": CONSERVATIVE_AVAILABILITY_METHOD,
                "finished": pd.NA,
                "home_goals": _single_numeric(frame.loc[frame["was_home"].astype(bool), "team_h_score"]),
                "away_goals": _single_numeric(frame.loc[~frame["was_home"].astype(bool), "team_a_score"]),
                "source_version": base["source_version"],
                "raw_snapshot_path": base["raw_snapshot_path"],
            }
        )
    fixtures = pd.DataFrame(rows)
    fixtures = fixtures.merge(
        team_lookup.rename(columns={"source_team_name": "home_team_name", "team_uid": "home_team_uid"}),
        on=["season", "home_team_name"],
        how="left",
    ).merge(
        team_lookup.rename(columns={"source_team_name": "away_team_name", "team_uid": "away_team_uid"}),
        on=["season", "away_team_name"],
        how="left",
    )
    if fixtures[["home_team_uid", "away_team_uid"]].isna().any().any():
        raise ValueError("Some fixtures could not map home/away teams to team_uid.")
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"], utc=True, errors="coerce")
    if fixtures["kickoff_time"].isna().any():
        raise ValueError("Some fixtures have missing or unparsable kickoff_time.")
    fixtures["finished"] = fixtures[["home_goals", "away_goals"]].notna().all(axis=1)
    return fixtures[
        [
            "season",
            "source_fixture_id",
            "fixture_key",
            "gameweek",
            "home_team_uid",
            "away_team_uid",
            "home_team_name",
            "away_team_name",
            "kickoff_time",
            "source_available_time",
            "source_available_method",
            "finished",
            "home_goals",
            "away_goals",
            "source_version",
            "raw_snapshot_path",
        ]
    ].sort_values(["season", "gameweek", "source_fixture_id"])


def _single_numeric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(values) == 0:
        return None
    return float(values[0])
