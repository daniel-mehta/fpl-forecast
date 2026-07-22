from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.panel.common import normalize_name, parse_seasons, phase2_dir
from fpl_forecast.panel.teams import TEAM_ALIAS_TEMPLATE, read_team_aliases


TARGET_COLUMNS = {
    "home_goals",
    "away_goals",
    "home_clean_sheet",
    "away_clean_sheet",
    "match_outcome",
}


def load_historical_team_fixtures(
    *,
    seasons: str | list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> pd.DataFrame:
    season_list = parse_seasons(seasons)
    output_dir = phase2_dir(normalized_dir)
    fixtures = pd.read_parquet(output_dir / "dim_fixture.parquet")
    teams = pd.read_parquet(output_dir / "team_season_map.parquet")
    fixtures = fixtures.loc[fixtures["season"].isin(season_list)].copy()
    table = build_team_fixture_table(fixtures, teams)
    validate_team_fixture_table(table)
    return table


def build_team_fixture_table(fixtures: pd.DataFrame, team_season_map: pd.DataFrame) -> pd.DataFrame:
    frame = fixtures.copy()
    home_ids = team_season_map[["season", "team_uid", "source_team_id"]].rename(
        columns={"team_uid": "home_team_uid", "source_team_id": "source_home_team_id"}
    )
    away_ids = team_season_map[["season", "team_uid", "source_team_id"]].rename(
        columns={"team_uid": "away_team_uid", "source_team_id": "source_away_team_id"}
    )
    frame = frame.merge(home_ids, on=["season", "home_team_uid"], how="left")
    frame = frame.merge(away_ids, on=["season", "away_team_uid"], how="left")
    frame["stable_fixture_uid"] = frame["fixture_key"]
    frame["information_cutoff"] = (
        pd.to_datetime(frame["kickoff_time"], utc=True)
        .groupby([frame["season"], frame["gameweek"]])
        .transform("min")
    )
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce").astype("Int64")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce").astype("Int64")
    frame["result_valid"] = (
        frame["finished"].astype("boolean").fillna(False)
        & frame["home_goals"].notna()
        & frame["away_goals"].notna()
    )
    return frame[
        [
            "season",
            "gameweek",
            "stable_fixture_uid",
            "source_fixture_id",
            "home_team_uid",
            "away_team_uid",
            "source_home_team_id",
            "source_away_team_id",
            "home_team_name",
            "away_team_name",
            "kickoff_time",
            "information_cutoff",
            "source_available_time",
            "source_available_method",
            "finished",
            "result_valid",
            "home_goals",
            "away_goals",
            "source_version",
            "raw_snapshot_path",
        ]
    ].sort_values(["season", "gameweek", "stable_fixture_uid"])


def validate_team_fixture_table(frame: pd.DataFrame) -> None:
    required = {
        "season",
        "gameweek",
        "stable_fixture_uid",
        "source_fixture_id",
        "home_team_uid",
        "away_team_uid",
        "kickoff_time",
        "information_cutoff",
        "source_available_time",
        "finished",
        "result_valid",
        "home_goals",
        "away_goals",
        "source_version",
        "raw_snapshot_path",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Team fixture table missing columns: {', '.join(sorted(missing))}")
    if frame.duplicated(["season", "stable_fixture_uid"]).any():
        raise ValueError("Team fixture table has duplicate fixture keys.")
    if frame["home_team_uid"].eq(frame["away_team_uid"]).any():
        raise ValueError("Team fixture table has home team equal to away team.")
    for column in ("kickoff_time", "information_cutoff", "source_available_time"):
        parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Team fixture table has invalid {column}.")
    valid = frame.loc[frame["result_valid"].astype(bool)]
    for column in ("home_goals", "away_goals"):
        values = pd.to_numeric(valid[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"Invalid nonnegative integer goals in {column}.")
    if frame[["source_version", "raw_snapshot_path"]].isna().any().any():
        raise ValueError("Team fixture table lacks source provenance.")


def frozen_prediction_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in TARGET_COLUMNS]


def assert_frozen_predictions_target_free(frame: pd.DataFrame) -> None:
    present = sorted(TARGET_COLUMNS & set(frame.columns))
    if present:
        raise ValueError(f"Frozen predictions contain target columns: {', '.join(present)}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_current_fixture_frame(
    *,
    season: str,
    as_of: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> pd.DataFrame:
    season_dir = Path(normalized_dir) / season
    fixtures_path = season_dir / "current_fixtures.parquet"
    teams_path = season_dir / "current_teams.parquet"
    if not fixtures_path.exists() or not teams_path.exists():
        raise FileNotFoundError(f"Missing official normalized current-season data for {season}.")
    fixtures = pd.read_parquet(fixtures_path)
    teams = pd.read_parquet(teams_path)
    inferred = _season_label_from_dates(pd.to_datetime(fixtures["kickoff_time"], utc=True).dropna())
    if inferred != season:
        raise ValueError(f"Current fixture season mismatch: requested {season}, inferred {inferred}.")

    team_map = _current_team_map(teams, normalized_dir=normalized_dir)
    frame = fixtures.merge(
        team_map.rename(
            columns={
                "team_id": "home_team_id",
                "team_uid": "home_team_uid",
                "team_name": "home_team_name",
            }
        ),
        on="home_team_id",
        how="left",
    ).merge(
        team_map.rename(
            columns={
                "team_id": "away_team_id",
                "team_uid": "away_team_uid",
                "team_name": "away_team_name",
            }
        ),
        on="away_team_id",
        how="left",
    )
    if frame[["home_team_uid", "away_team_uid"]].isna().any().any():
        missing = teams.loc[~teams["team_id"].isin(team_map["team_id"]), "team_name"].tolist()
        raise ValueError(f"Current teams lack stable identity mapping: {', '.join(map(str, missing))}")
    as_of_ts = pd.to_datetime(as_of, utc=True)
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True)
    future = frame.loc[frame["kickoff_time"] > as_of_ts].copy()
    if future.empty:
        raise ValueError(f"No future fixtures for {season} after as_of={as_of_ts.isoformat()}.")
    future["stable_fixture_uid"] = future["fixture_id"].map(lambda value: f"{season}:{int(value)}")
    future["source_fixture_id"] = future["fixture_id"]
    future["source_home_team_id"] = future["home_team_id"]
    future["source_away_team_id"] = future["away_team_id"]
    future["information_cutoff"] = as_of_ts
    future["source_available_time"] = pd.NaT
    future["source_available_method"] = "future_fixture_no_result"
    future["result_valid"] = False
    return future[
        [
            "season",
            "gameweek",
            "stable_fixture_uid",
            "source_fixture_id",
            "home_team_uid",
            "away_team_uid",
            "source_home_team_id",
            "source_away_team_id",
            "home_team_name",
            "away_team_name",
            "kickoff_time",
            "information_cutoff",
            "source_available_time",
            "source_available_method",
            "finished",
            "result_valid",
            "source_version",
            "raw_snapshot_path",
        ]
    ]


def _current_team_map(teams: pd.DataFrame, *, normalized_dir: Path | str) -> pd.DataFrame:
    dim_team = pd.read_parquet(phase2_dir(normalized_dir) / "dim_team.parquet")
    aliases = read_team_aliases(TEAM_ALIAS_TEMPLATE)
    records = teams[["team_id", "team_name"]].copy()
    records["canonical_name"] = records["team_name"].map(
        lambda value: aliases.get(normalize_name(value), value)
    )
    records["normalized_short_name"] = records["canonical_name"].map(normalize_name)
    return records.merge(dim_team[["team_uid", "normalized_short_name"]], on="normalized_short_name", how="inner")


def _season_label_from_dates(values: pd.Series) -> str:
    if values.empty:
        raise ValueError("Cannot infer season from empty current fixture kickoff times.")
    first = values.min()
    start_year = first.year if first.month >= 6 else first.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"
