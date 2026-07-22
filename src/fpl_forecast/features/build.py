from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.panel.common import phase2_dir, write_parquet


REGISTRY_PATH = PROJECT_ROOT / "src" / "fpl_forecast" / "features" / "registry.json"


@dataclass(frozen=True)
class FeatureBuildResult:
    feature_path: Path
    features: pd.DataFrame


def load_feature_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def registered_feature_names(path: Path = REGISTRY_PATH) -> set[str]:
    return {item["feature_name"] for item in load_feature_registry(path)}


def build_features_player_fixture(
    *,
    seasons: list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> FeatureBuildResult:
    output_dir = phase2_dir(normalized_dir)
    fact_path = output_dir / "fact_player_fixture.parquet"
    fixture_path = output_dir / "dim_fixture.parquet"
    if not fact_path.exists():
        raise FileNotFoundError(f"Missing fact table: {fact_path}")
    if not fixture_path.exists():
        raise FileNotFoundError(f"Missing fixture table: {fixture_path}")
    fact = pd.read_parquet(fact_path)
    fixtures = pd.read_parquet(fixture_path)
    fact = fact.loc[fact["season"].isin(seasons)].copy()
    features = build_features_from_frames(fact, fixtures)
    path = write_parquet(features, output_dir / "features_player_fixture.parquet")
    return FeatureBuildResult(path, features)


def build_features_from_frames(fact: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    feature = fact[
        [
            "season",
            "player_uid",
            "fixture_key",
            "gameweek",
            "kickoff_time",
            "source_available_time",
            "information_cutoff",
            "player_team_uid",
            "opponent_team_uid",
            "was_home",
            "entity_type",
            "target_total_points",
        ]
    ].copy()
    feature["kickoff_time"] = pd.to_datetime(feature["kickoff_time"], utc=True)
    feature["information_cutoff"] = pd.to_datetime(feature["information_cutoff"], utc=True)
    feature["home_flag"] = feature["was_home"].astype("boolean").astype("Int64")

    working = fact.sort_values(["season", "player_uid", "kickoff_time", "fixture_key"]).copy()
    working["kickoff_time"] = pd.to_datetime(working["kickoff_time"], utc=True)
    working["source_available_time"] = pd.to_datetime(working["source_available_time"], utc=True)
    working["information_cutoff"] = pd.to_datetime(working["information_cutoff"], utc=True)
    for column in ("minutes", "starts", "total_points", "goals_scored", "assists"):
        if column not in working.columns:
            working[column] = 0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)
    working["appearance"] = (working["minutes"] > 0).astype(int)
    lagged = _player_cutoff_features(working)

    prior = (
        working.assign(appearance=(working["minutes"] > 0).astype(int))
        .groupby(["season", "player_uid"], as_index=False)
        .agg(
            prior_join_minutes=("minutes", "sum"),
            prior_join_appearances=("appearance", "sum"),
        )
    )
    season_order = {season: index for index, season in enumerate(sorted(working["season"].unique()))}
    prior["join_to_season"] = prior["season"].map(
        lambda season: _next_loaded_season(season, season_order)
    )
    prior = prior.dropna(subset=["join_to_season"]).rename(
        columns={
            "season": "prior_source_season",
            "join_to_season": "season",
            "prior_join_minutes": "prior_season_minutes",
            "prior_join_appearances": "prior_season_appearances",
        }
    )
    lagged = lagged.merge(prior, on=["season", "player_uid"], how="left")

    feature = feature.merge(lagged, on=["season", "player_uid", "fixture_key"], how="left")
    feature["prior_source_season"] = feature["prior_source_season"].astype("string")

    feature = _add_team_cutoff_features(feature, fixtures)

    fill_zero = [
        column
        for column in feature.columns
        if column.startswith("prev") or column.startswith("season_to_date") or column.endswith("_sum")
    ]
    for column in fill_zero:
        if column not in {"prev_fixture_minutes"}:
            feature[column] = pd.to_numeric(feature[column], errors="coerce").fillna(0)
    feature["feature_registry_version"] = "registry.json"
    return feature.sort_values(["season", "kickoff_time", "fixture_key", "player_uid"])


def _player_cutoff_features(working: pd.DataFrame) -> pd.DataFrame:
    source_rows: list[pd.DataFrame] = []
    for _, group in working.groupby(["season", "player_uid"], sort=False):
        group = group.sort_values(["source_available_time", "fixture_key"]).copy()
        source = group[["kickoff_time", "source_available_time"]].rename(
            columns={
                "kickoff_time": "player_max_source_kickoff",
                "source_available_time": "player_max_source_available_time",
            }
        )
        source["prev_fixture_minutes"] = group["minutes"].to_numpy()
        for column, output_prefix in (
            ("minutes", "minutes"),
            ("starts", "starts"),
            ("total_points", "total_points"),
            ("goals_scored", "goals_scored"),
            ("assists", "assists"),
        ):
            for window in (3, 5):
                source[f"prev{window}_{output_prefix}_sum"] = (
                    group[column].rolling(window, min_periods=1).sum().to_numpy()
                )
        source["season_to_date_minutes"] = group["minutes"].cumsum().to_numpy()
        source["season_to_date_appearances"] = group["appearance"].cumsum().to_numpy()
        source["season"] = group["season"].to_numpy()
        source["player_uid"] = group["player_uid"].to_numpy()
        source_rows.append(source)

    source_history = pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame()
    targets = working[["season", "player_uid", "fixture_key", "information_cutoff"]].copy()
    merged_rows: list[pd.DataFrame] = []
    for key, target_group in targets.groupby(["season", "player_uid"], sort=False):
        source_group = source_history.loc[
            (source_history["season"] == key[0]) & (source_history["player_uid"] == key[1])
        ].sort_values("player_max_source_available_time")
        target_group = target_group.sort_values("information_cutoff")
        merged = pd.merge_asof(
            target_group,
            source_group,
            left_on="information_cutoff",
            right_on="player_max_source_available_time",
            by=["season", "player_uid"],
            allow_exact_matches=False,
            direction="backward",
        )
        merged_rows.append(merged)
    output = pd.concat(merged_rows, ignore_index=True) if merged_rows else targets
    return output.drop(columns=["information_cutoff"])


def _add_team_cutoff_features(feature: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    team_history = _team_fixture_history(fixtures)
    team_history["team_prev3_goals_for_sum"] = team_history.groupby(
        ["season", "team_uid"], sort=False
    )["goals_for"].transform(lambda series: series.rolling(3, min_periods=1).sum())
    team_history["team_prev3_goals_against_sum"] = team_history.groupby(
        ["season", "team_uid"], sort=False
    )["goals_against"].transform(lambda series: series.rolling(3, min_periods=1).sum())
    team_history = team_history.rename(
        columns={
            "kickoff_time": "team_max_source_kickoff",
            "source_available_time": "team_max_source_available_time",
        }
    )
    team_join = _asof_team_feature(
        feature,
        team_history,
        team_column="player_team_uid",
        history_team_column="team_uid",
        value_columns=["team_prev3_goals_for_sum"],
    )
    opponent_join = _asof_team_feature(
        feature,
        team_history,
        team_column="opponent_team_uid",
        history_team_column="team_uid",
        value_columns=["team_prev3_goals_against_sum"],
    ).rename(
        columns={
            "team_prev3_goals_against_sum": "opponent_prev3_goals_against_sum",
            "team_max_source_kickoff": "opponent_max_source_kickoff",
            "team_max_source_available_time": "opponent_max_source_available_time",
        }
    )
    return feature.merge(
        team_join,
        on=["season", "player_uid", "fixture_key"],
        how="left",
    ).merge(
        opponent_join,
        on=["season", "player_uid", "fixture_key"],
        how="left",
    )


def _asof_team_feature(
    feature: pd.DataFrame,
    team_history: pd.DataFrame,
    *,
    team_column: str,
    history_team_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    base = feature[["season", "player_uid", "fixture_key", "information_cutoff", team_column]].copy()
    base = base.rename(columns={team_column: history_team_column})
    rows: list[pd.DataFrame] = []
    for key, target_group in base.groupby(["season", history_team_column], sort=False):
        source_group = team_history.loc[
            (team_history["season"] == key[0]) & (team_history[history_team_column] == key[1])
        ].sort_values("team_max_source_available_time")
        target_group = target_group.sort_values("information_cutoff")
        merged = pd.merge_asof(
            target_group,
            source_group[
                [
                    "season",
                    history_team_column,
                    "team_max_source_kickoff",
                    "team_max_source_available_time",
                    *value_columns,
                ]
            ],
            left_on="information_cutoff",
            right_on="team_max_source_available_time",
            by=["season", history_team_column],
            allow_exact_matches=False,
            direction="backward",
        )
        rows.append(
            merged[
                [
                    "season",
                    "player_uid",
                    "fixture_key",
                    "team_max_source_kickoff",
                    "team_max_source_available_time",
                    *value_columns,
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _team_fixture_history(fixtures: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = fixtures.copy()
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True)
    if "source_available_time" not in frame.columns:
        frame["source_available_time"] = frame["kickoff_time"] + pd.Timedelta(hours=3)
    frame["source_available_time"] = pd.to_datetime(frame["source_available_time"], utc=True)
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "season": row.season,
                "fixture_key": row.fixture_key,
                "team_uid": row.home_team_uid,
                "kickoff_time": row.kickoff_time,
                "source_available_time": row.source_available_time,
                "goals_for": row.home_goals,
                "goals_against": row.away_goals,
            }
        )
        rows.append(
            {
                "season": row.season,
                "fixture_key": row.fixture_key,
                "team_uid": row.away_team_uid,
                "kickoff_time": row.kickoff_time,
                "source_available_time": row.source_available_time,
                "goals_for": row.away_goals,
                "goals_against": row.home_goals,
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "team_uid", "source_available_time", "fixture_key"])




def _next_loaded_season(season: str, season_order: dict[str, int]) -> str | None:
    index = season_order[season] + 1
    for candidate, candidate_index in season_order.items():
        if candidate_index == index:
            return candidate
    return None
