from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.panel.common import parse_seasons, phase2_dir
from fpl_forecast.xpoints.config import XPointsConfig


OUTCOME_COLUMNS = {
    "total_points",
    "target_total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "bonus",
    "bps",
}


def load_xpoints_frame(*, seasons: str | list[str], normalized_dir: Path | str = NORMALIZED_DIR) -> pd.DataFrame:
    season_list = parse_seasons(seasons)
    fact = pd.read_parquet(phase2_dir(normalized_dir) / "fact_player_fixture.parquet")
    fact = fact.loc[fact["season"].isin(season_list)].copy()
    fact = fact.loc[fact["entity_type"].eq("player")].copy()
    fact["stable_fixture_uid"] = fact["fixture_key"]
    fact["actual_total_points"] = pd.to_numeric(fact["total_points"], errors="coerce")
    fact["pre_deadline_population"] = _population_from_history(fact)
    return fact.sort_values(["season", "gameweek", "stable_fixture_uid", "player_uid"]).reset_index(drop=True)


def load_minutes_predictions(*, mode: str, config: XPointsConfig) -> pd.DataFrame:
    run_id = config.minutes_runs[mode]
    path = PROJECT_ROOT / "reports" / "minutes_backtests" / run_id / "frozen_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 5 minutes artifact: {path}")
    minutes = pd.read_parquet(path)
    keep_models = {
        "M3_EWMA_MINUTES": "M3",
        "M5_REGULARIZED_STATE_SOFTMAX": "M5",
    }
    minutes = minutes.loc[minutes["model_name"].isin(keep_models)].copy()
    minutes["minutes_variant"] = minutes["model_name"].map(keep_models)
    return minutes


def load_team_predictions(*, mode: str, config: XPointsConfig) -> pd.DataFrame:
    run_id = config.team_runs[mode]
    path = PROJECT_ROOT / "reports" / "team_backtests" / run_id / "frozen_fixture_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 4 team artifact: {path}")
    team = pd.read_parquet(path)
    return team.loc[team["model_name"].eq("T2_REGULARIZED_ATTACK_DEFENCE")].copy()


def load_phase3_reference(*, mode: str, config: XPointsConfig) -> pd.DataFrame:
    run_id = config.phase3_runs[mode]
    path = PROJECT_ROOT / "reports" / "backtests" / run_id / "frozen_fixture_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 3 baseline artifact: {path}")
    phase3 = pd.read_parquet(path)
    baseline = config.reference_phase3_baseline
    phase3 = phase3.loc[phase3["baseline"].eq(baseline)].copy()
    phase3 = phase3.rename(columns={"prediction": "expected_points"})
    phase3["model_name"] = f"X0_PHASE3_{baseline}"
    phase3["stable_fixture_uid"] = phase3["fixture_key"]
    return phase3


def frozen_prediction_columns(frame: pd.DataFrame) -> list[str]:
    forbidden = {column.lower() for column in OUTCOME_COLUMNS}
    return [column for column in frame.columns if column.lower() not in forbidden]


def assert_frozen_target_free(frame: pd.DataFrame, forbidden_columns: tuple[str, ...]) -> None:
    lower = {column.lower(): column for column in frame.columns}
    forbidden = {column.lower() for column in forbidden_columns} | {column.lower() for column in OUTCOME_COLUMNS}
    present = sorted(lower[column] for column in lower if column in forbidden or "xp" == column)
    if present:
        raise ValueError(f"Frozen xPoints predictions contain forbidden columns: {', '.join(present)}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _population_from_history(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values(["player_uid", "source_available_time", "kickoff_time", "fixture_key"]).copy()
    ordered["appearance"] = pd.to_numeric(ordered["minutes"], errors="coerce").fillna(0).gt(0).astype(int)
    ordered["prior_appearance"] = ordered.groupby("player_uid")["appearance"].cumsum() - ordered["appearance"]
    ordered["prior_minutes"] = (
        ordered.groupby("player_uid")["minutes"].cumsum() - pd.to_numeric(ordered["minutes"], errors="coerce").fillna(0)
    )
    population = pd.Series("cold_start_no_history", index=ordered.index)
    population.loc[ordered["prior_appearance"].gt(0) | ordered["prior_minutes"].gt(0)] = (
        "pre_deadline_history_active"
    )
    return population.sort_index()
