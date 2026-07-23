from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.decision.config import DecisionConfig
from fpl_forecast.panel.common import phase2_dir


def load_decision_candidates(
    *,
    mode: str,
    config: DecisionConfig,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> pd.DataFrame:
    run_id = config.xpoints_runs[mode]
    xpoints_path = PROJECT_ROOT / "reports" / "xpoints_backtests" / run_id / "player_gameweek_predictions.parquet"
    frozen_path = PROJECT_ROOT / "reports" / "xpoints_backtests" / run_id / "frozen_player_fixture_predictions.parquet"
    if not xpoints_path.exists() or not frozen_path.exists():
        raise FileNotFoundError(f"Missing Phase 6 xPoints artifacts for {mode}: {run_id}.")
    gameweek = pd.read_parquet(xpoints_path)
    frozen = pd.read_parquet(frozen_path)
    fact = pd.read_parquet(phase2_dir(normalized_dir) / "fact_player_fixture.parquet")
    meta = (
        fact.loc[fact["entity_type"].eq("player")]
        .sort_values(["season", "gameweek", "player_uid", "source_available_time"])
        .groupby(["season", "gameweek", "player_uid"], as_index=False)
        .agg(
            player_name=("player_name", "last"),
            fpl_position=("fpl_position", "last"),
            player_team_uid=("player_team_uid", "last"),
            price_tenths=("price_tenths", "last"),
            actual_points=("total_points", "sum"),
            actual_minutes=("minutes", "sum"),
        )
    )
    probs = (
        frozen.groupby(["season", "gameweek", "player_uid", "model_name"], as_index=False)
        .agg(
            p_appearance=("p_appearance", "max"),
            p_start=("p_start", "max"),
            information_cutoff=("information_cutoff", "min"),
            cold_start_no_history=("cold_start_no_history", "max"),
        )
    )
    frame = gameweek.merge(meta, on=["season", "gameweek", "player_uid"], how="left", validate="many_to_one")
    frame = frame.merge(probs, on=["season", "gameweek", "player_uid", "model_name"], how="left")
    frame["p_appearance"] = pd.to_numeric(frame["p_appearance"], errors="coerce").fillna(
        frame["expected_points"].gt(0).astype(float)
    )
    frame["p_start"] = pd.to_numeric(frame["p_start"], errors="coerce").fillna(frame["p_appearance"])
    if frame[["fpl_position", "player_team_uid", "price_tenths"]].isna().any().any():
        raise ValueError("Decision candidates are missing price, position, or team metadata.")
    if "D0_PRICE_VALUE_BASELINE" in config.comparison_models:
        base_keys = [
            "season",
            "gameweek",
            "player_uid",
            "player_name",
            "fpl_position",
            "player_team_uid",
            "price_tenths",
            "actual_points",
            "actual_minutes",
            "p_appearance",
            "p_start",
            "information_cutoff",
            "cold_start_no_history",
        ]
        baseline = frame.sort_values(["season", "gameweek", "player_uid"]).drop_duplicates(
            ["season", "gameweek", "player_uid"]
        )[base_keys].copy()
        baseline["model_name"] = "D0_PRICE_VALUE_BASELINE"
        position_floor = baseline.groupby(["season", "gameweek", "fpl_position"])["price_tenths"].transform("min")
        baseline["expected_points"] = (baseline["price_tenths"] - position_floor).clip(lower=0) / 10.0
        baseline["points_std"] = 0.0
        baseline["points_p10"] = baseline["expected_points"]
        baseline["points_p25"] = baseline["expected_points"]
        baseline["points_p50"] = baseline["expected_points"]
        baseline["points_p75"] = baseline["expected_points"]
        baseline["points_p90"] = baseline["expected_points"]
        baseline["prob_points_eq_0"] = baseline["expected_points"].eq(0).astype(float)
        baseline["prob_points_ge_1"] = baseline["expected_points"].ge(1).astype(float)
        baseline["prob_points_ge_5"] = baseline["expected_points"].ge(5).astype(float)
        baseline["prob_points_ge_10"] = baseline["expected_points"].ge(10).astype(float)
        for column in frame.columns:
            if column not in baseline.columns:
                baseline[column] = pd.NA
        frame = pd.concat([frame, baseline[frame.columns]], ignore_index=True)
    return frame


def assert_frozen_decisions_target_free(frame: pd.DataFrame, forbidden_columns: tuple[str, ...]) -> None:
    lower = {column.lower(): column for column in frame.columns}
    forbidden = {column.lower() for column in forbidden_columns}
    present = sorted(value for key, value in lower.items() if key in forbidden or key == "xp")
    if present:
        raise ValueError(f"Frozen decision artifact contains target/future columns: {', '.join(present)}")


def candidate_slice(frame: pd.DataFrame, *, season: str, gameweek: int, model_name: str) -> pd.DataFrame:
    subset = frame.loc[
        frame["season"].eq(season) & frame["gameweek"].eq(gameweek) & frame["model_name"].eq(model_name)
    ].copy()
    subset = subset.drop_duplicates("player_uid")
    if subset.empty:
        raise ValueError(f"No decision candidates for {season} GW{gameweek} {model_name}.")
    return subset
