from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.panel.common import parse_seasons, phase2_dir


TARGET_COLUMNS = {
    "actual_minutes",
    "actual_appearance",
    "actual_start",
    "actual_reached_60",
    "actual_played_90",
    "actual_state",
}


def load_minutes_frame(
    *,
    seasons: str | list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> pd.DataFrame:
    season_list = parse_seasons(seasons)
    output_dir = phase2_dir(normalized_dir)
    fact = pd.read_parquet(output_dir / "fact_player_fixture.parquet")
    features = pd.read_parquet(output_dir / "features_player_fixture.parquet")
    player_map = pd.read_parquet(output_dir / "player_season_map.parquet")
    fact = fact.loc[fact["season"].isin(season_list)].copy()
    frame = build_minutes_frame(fact, features, player_map)
    validate_minutes_frame(frame)
    return frame


def build_minutes_frame(
    fact: pd.DataFrame,
    features: pd.DataFrame,
    player_season_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_fact = {
        "season",
        "gameweek",
        "player_uid",
        "fixture_key",
        "player_name",
        "player_team_uid",
        "opponent_team_uid",
        "fpl_position",
        "entity_type",
        "was_home",
        "kickoff_time",
        "information_cutoff",
        "source_available_time",
        "source_available_method",
        "minutes",
        "starts",
    }
    missing = required_fact.difference(fact.columns)
    if missing:
        raise ValueError(f"Minutes fact data missing columns: {', '.join(sorted(missing))}")

    keep = [
        "season",
        "gameweek",
        "player_uid",
        "fixture_key",
        "player_name",
        "player_team_uid",
        "opponent_team_uid",
        "fpl_position",
        "entity_type",
        "was_home",
        "kickoff_time",
        "information_cutoff",
        "source_available_time",
        "source_available_method",
        "minutes",
        "starts",
        "source_version",
        "raw_snapshot_path",
    ]
    frame = fact[[column for column in keep if column in fact.columns]].copy()
    feature_keep = [
        "season",
        "player_uid",
        "fixture_key",
        "prev_fixture_minutes",
        "prev3_minutes_sum",
        "prev5_minutes_sum",
        "prev3_starts_sum",
        "prev5_starts_sum",
        "season_to_date_minutes",
        "season_to_date_appearances",
        "prior_source_season",
        "prior_season_minutes",
        "prior_season_appearances",
        "player_max_source_available_time",
    ]
    feature_keep = [column for column in feature_keep if column in features.columns]
    frame = frame.merge(
        features[feature_keep],
        on=["season", "player_uid", "fixture_key"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.loc[frame["entity_type"].eq("player")].copy()
    frame["stable_fixture_uid"] = frame["fixture_key"]
    frame["actual_minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0).clip(0, 120)
    starts = pd.to_numeric(frame["starts"], errors="coerce")
    frame["starts_exact_available"] = starts.notna()
    frame["actual_start"] = starts.where(starts.notna()).astype("Int64")
    frame["actual_appearance"] = frame["actual_minutes"].gt(0).astype(int)
    frame["actual_reached_60"] = frame["actual_minutes"].ge(60).astype(int)
    frame["actual_played_90"] = frame["actual_minutes"].ge(90).astype(int)
    frame["actual_state"] = [
        role_duration_state(minutes, start if pd.notna(start) else None)
        for minutes, start in zip(frame["actual_minutes"], frame["actual_start"], strict=False)
    ]
    frame = _add_shifted_history(frame)
    if player_season_map is not None and not player_season_map.empty:
        frame = _add_season_identity_diagnostics(frame, player_season_map)
    else:
        frame["transferred_player"] = False
        frame["position_change"] = False
    frame = _assign_populations(frame)
    return frame.sort_values(["season", "gameweek", "stable_fixture_uid", "player_uid"]).reset_index(
        drop=True
    )


def role_duration_state(minutes: float, starts: int | float | None) -> str:
    played = float(minutes or 0)
    started = False if starts is None or pd.isna(starts) else int(starts) == 1
    if played <= 0:
        return "DNP"
    if not started:
        return "SUB_60_PLUS" if played >= 60 else "SUB_UNDER_60"
    if played < 60:
        return "START_UNDER_60"
    if played < 90:
        return "START_60_TO_89"
    return "START_90"


def validate_minutes_frame(frame: pd.DataFrame) -> None:
    required = {
        "season",
        "gameweek",
        "player_uid",
        "stable_fixture_uid",
        "information_cutoff",
        "source_available_time",
        "max_feature_source_available_time",
        "actual_minutes",
        "actual_state",
        "evaluation_population",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Minutes frame missing columns: {', '.join(sorted(missing))}")
    if frame.duplicated(["season", "stable_fixture_uid", "player_uid"]).any():
        raise ValueError("Minutes frame has duplicate player-fixture keys.")
    minutes = pd.to_numeric(frame["actual_minutes"], errors="coerce")
    if minutes.isna().any() or minutes.lt(0).any() or minutes.gt(120).any():
        raise ValueError("Actual minutes must be numeric and between 0 and 120.")
    starts = pd.to_numeric(frame["actual_start"], errors="coerce")
    bad_starts = starts.notna() & ~starts.isin([0, 1])
    if bad_starts.any():
        raise ValueError("Exact starts must be 0, 1, or null.")
    impossible = starts.eq(1) & minutes.le(0)
    if impossible.any():
        raise ValueError("Start labels imply an appearance but some rows have zero minutes.")
    cutoff = pd.to_datetime(frame["information_cutoff"], utc=True)
    max_feature_time = pd.to_datetime(frame["max_feature_source_available_time"], utc=True)
    leaked = max_feature_time.notna() & ~max_feature_time.lt(cutoff)
    if leaked.any():
        raise ValueError("Minutes features contain source availability at or after the cutoff.")


def frozen_prediction_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in TARGET_COLUMNS]


def assert_frozen_predictions_target_free(frame: pd.DataFrame, forbidden: tuple[str, ...]) -> None:
    present = sorted((TARGET_COLUMNS | set(forbidden)).intersection(frame.columns))
    if present:
        raise ValueError(f"Frozen minutes predictions contain target/post-cutoff columns: {', '.join(present)}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_shifted_history(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["source_available_time"] = pd.to_datetime(output["source_available_time"], utc=True)
    output["information_cutoff"] = pd.to_datetime(output["information_cutoff"], utc=True)
    output["kickoff_time"] = pd.to_datetime(output["kickoff_time"], utc=True)
    for column in ("actual_minutes", "actual_appearance", "actual_start", "actual_reached_60", "actual_played_90"):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0)
    for window in (1, 3, 5, 10):
        output[f"lag{window}_minutes_mean"] = 0.0
        output[f"lag{window}_appearance_rate"] = 0.0
        output[f"lag{window}_start_rate"] = 0.0
    output["lag_source_available_time"] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    output["season_appearance_before"] = 0.0
    output["season_minutes_before"] = 0.0
    output["season_starts_before"] = 0.0
    for _, player_rows in output.groupby("player_uid", sort=False):
        available = player_rows.sort_values(["source_available_time", "kickoff_time", "fixture_key"])
        avail_times = available["source_available_time"].to_numpy(dtype="datetime64[ns]")
        avail_minutes = available["actual_minutes"].to_numpy(dtype=float)
        avail_appear = available["actual_appearance"].to_numpy(dtype=float)
        avail_start = available["actual_start"].to_numpy(dtype=float)
        avail_season = available["season"].astype(str).to_numpy()
        for index, row in player_rows.iterrows():
            cutoff = pd.Timestamp(row["information_cutoff"]).to_datetime64()
            end = int(np.searchsorted(avail_times, cutoff, side="left"))
            for window in (1, 3, 5, 10):
                start = max(0, end - window)
                if end > start:
                    output.loc[index, f"lag{window}_minutes_mean"] = float(avail_minutes[start:end].mean())
                    output.loc[index, f"lag{window}_appearance_rate"] = float(avail_appear[start:end].mean())
                    output.loc[index, f"lag{window}_start_rate"] = float(avail_start[start:end].mean())
            if end > 0:
                output.loc[index, "lag_source_available_time"] = pd.Timestamp(avail_times[end - 1], tz="UTC")
                same_season = avail_season[:end] == str(row["season"])
                output.loc[index, "season_appearance_before"] = float(avail_appear[:end][same_season].sum())
                output.loc[index, "season_minutes_before"] = float(avail_minutes[:end][same_season].sum())
                output.loc[index, "season_starts_before"] = float(avail_start[:end][same_season].sum())
    ordered = output.sort_index()
    ordered["days_since_last_source"] = (
        (ordered["information_cutoff"] - ordered["lag_source_available_time"]).dt.total_seconds() / 86400
    ).clip(lower=0)
    ordered["prior_seen_before"] = ordered["lag_source_available_time"].notna()
    fallback_time = pd.Timestamp("1900-01-01T00:00:00Z")
    ordered["max_feature_source_available_time"] = ordered["lag_source_available_time"].fillna(fallback_time)
    return ordered.sort_index()


def _add_season_identity_diagnostics(frame: pd.DataFrame, player_season_map: pd.DataFrame) -> pd.DataFrame:
    season_map = player_season_map[
        ["season", "player_uid", "source_team", "fpl_position"]
    ].drop_duplicates(["season", "player_uid"])
    season_map = season_map.sort_values(["player_uid", "season"])
    season_map["previous_source_team"] = season_map.groupby("player_uid")["source_team"].shift(1)
    season_map["previous_fpl_position"] = season_map.groupby("player_uid")["fpl_position"].shift(1)
    season_map["transferred_player"] = (
        season_map["previous_source_team"].notna()
        & season_map["source_team"].ne(season_map["previous_source_team"])
    )
    season_map["position_change"] = (
        season_map["previous_fpl_position"].notna()
        & season_map["fpl_position"].ne(season_map["previous_fpl_position"])
    )
    output = frame.merge(
        season_map[["season", "player_uid", "transferred_player", "position_change"]],
        on=["season", "player_uid"],
        how="left",
    )
    output["transferred_player"] = output["transferred_player"].fillna(False).astype(bool)
    output["position_change"] = output["position_change"].fillna(False).astype(bool)
    return output


def _assign_populations(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    prior_minutes = pd.to_numeric(output.get("prior_season_minutes", 0), errors="coerce").fillna(0)
    prior_apps = pd.to_numeric(output.get("prior_season_appearances", 0), errors="coerce").fillna(0)
    prev5 = pd.to_numeric(output["lag5_minutes_mean"], errors="coerce").fillna(0) * 5
    season_apps = pd.to_numeric(output["season_appearance_before"], errors="coerce").fillna(0)
    is_gw1 = pd.to_numeric(output["gameweek"], errors="coerce").eq(1)
    has_prior_history = prior_minutes.gt(0) | prior_apps.gt(0)
    has_inseason_history = prev5.gt(0) | season_apps.gt(0)
    output["pre_deadline_history_active"] = np.where(is_gw1, has_prior_history, has_inseason_history)
    output["cold_start_no_history"] = ~(has_prior_history | has_inseason_history)
    output["actual_appearances_diagnostic"] = output["actual_appearance"].astype(bool)
    output["evaluation_population"] = np.select(
        [
            output["pre_deadline_history_active"].astype(bool),
            output["cold_start_no_history"].astype(bool),
        ],
        ["pre_deadline_history_active", "cold_start_no_history"],
        default="other_history_inactive",
    )
    output["primary_data_quality_population"] = "all_observed_players"
    return output
