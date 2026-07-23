from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_forecast.xpoints.components import attacking_shares, fit_component_priors, player_component_rates
from fpl_forecast.xpoints.config import XPointsConfig
from fpl_forecast.xpoints.simulation import simulate_component_points


def predict_xpoints_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    minutes_predictions: pd.DataFrame,
    team_predictions: pd.DataFrame,
    phase3_reference: pd.DataFrame,
    config: XPointsConfig,
    fold_index: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    priors = fit_component_priors(train, config)
    base_minutes = _minutes_variant(minutes_predictions, "M3")
    x0, x0_draws = _x0_reference(test, phase3_reference, config)
    x1_base = _component_base(test, base_minutes, team_predictions, priors, config, team_constrained=False)
    x1, x1_draws = _finalize_model(
        x1_base,
        "X1_INDEPENDENT_COMPONENT_RATES_M3",
        config=config,
        seed=config.random_seed + fold_index * 101 + 1,
    )
    x2_frames = []
    x2_draws = []
    for variant, model_name, offset in (
        ("M3", "X2_TEAM_CONSTRAINED_SIM_M3", 2),
        ("M5", "X2_TEAM_CONSTRAINED_SIM_M5", 3),
    ):
        minutes = _minutes_variant(minutes_predictions, variant)
        base = _component_base(test, minutes, team_predictions, priors, config, team_constrained=True)
        pred, draws = _finalize_model(
            base,
            model_name,
            config=config,
            seed=config.random_seed + fold_index * 101 + offset,
        )
        x2_frames.append(pred)
        x2_draws.append(draws)
    predictions = pd.concat([x0, x1, *x2_frames], ignore_index=True)
    draws = np.vstack([x0_draws, x1_draws, *x2_draws])
    conservation = _conservation_diagnostics(pd.concat(x2_frames, ignore_index=True), team_predictions)
    return predictions, draws, conservation


def _x0_reference(
    test: pd.DataFrame,
    phase3_reference: pd.DataFrame,
    config: XPointsConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    id_cols = _id_columns(test)
    merged = test[id_cols].merge(
        phase3_reference[["season", "stable_fixture_uid", "player_uid", "expected_points"]],
        on=["season", "stable_fixture_uid", "player_uid"],
        how="left",
        validate="one_to_one",
    )
    merged["expected_points"] = pd.to_numeric(merged["expected_points"], errors="coerce").fillna(0)
    merged["phase3_reference_expected_points"] = merged["expected_points"]
    draws = np.repeat(
        np.rint(merged["phase3_reference_expected_points"]).to_numpy(dtype=int)[:, None],
        config.draw_count,
        axis=1,
    )
    from fpl_forecast.xpoints.simulation import summarize_draws

    summaries = pd.DataFrame([summarize_draws(row) for row in draws], index=merged.index)
    output = pd.concat([merged.drop(columns=["expected_points"]), summaries], axis=1)
    output["model_name"] = "X0_PHASE3_B5_EB_POINTS_PER90"
    output["phase4_team_model"] = "none"
    output["phase5_minutes_model"] = "none"
    output["component_model"] = "direct_phase3_reference"
    for column in _component_output_columns():
        output[column] = 0.0
    return output, draws.astype(np.int16)


def _component_base(
    test: pd.DataFrame,
    minutes: pd.DataFrame,
    team_predictions: pd.DataFrame,
    priors: dict[str, object],
    config: XPointsConfig,
    *,
    team_constrained: bool,
) -> pd.DataFrame:
    id_cols = _id_columns(test)
    base = test[id_cols].merge(
        minutes,
        on=["season", "stable_fixture_uid", "player_uid"],
        how="left",
        validate="one_to_one",
    )
    rates = player_component_rates(test, priors, config)
    base = base.merge(rates, on=["season", "stable_fixture_uid", "player_uid", "fpl_position", "player_team_uid"])
    base = _attach_team_context(base, team_predictions)
    if team_constrained:
        shares = attacking_shares(rates, base, config)
        base = base.merge(
            shares[["season", "stable_fixture_uid", "player_uid", "goal_share", "assist_share"]],
            on=["season", "stable_fixture_uid", "player_uid"],
            how="left",
        )
        base["expected_goals"] = base["team_expected_goals"] * base["goal_share"].fillna(0)
        base["expected_assists"] = (
            base["team_expected_goals"] * base["assisted_goal_rate"] * base["assist_share"].fillna(0)
        )
    else:
        exposure = pd.to_numeric(base["expected_minutes"], errors="coerce").fillna(0).clip(0, 90) / 90
        base["expected_goals"] = base["goals_scored_per90"] * exposure
        base["expected_assists"] = base["assists_per90"] * exposure
    exposure = pd.to_numeric(base["expected_minutes"], errors="coerce").fillna(0).clip(0, 90) / 90
    base["clean_sheet_probability"] = base["team_clean_sheet_probability"].fillna(0) * base["p_reached_60"].fillna(0)
    base["expected_goals_conceded_deduction_events"] = (
        base["opponent_expected_goals"].fillna(0).clip(lower=0) / 2
    ) * base["p_reached_60"].fillna(0)
    base["expected_saves"] = np.where(base["fpl_position"].eq("GKP"), base["saves_per90"] * exposure, 0)
    base["expected_penalty_saves"] = np.where(
        base["fpl_position"].eq("GKP"), (base["penalties_saved_per90"] * exposure).clip(0, 0.15), 0
    )
    base["expected_penalty_misses"] = (base["penalties_missed_per90"] * exposure).clip(0, 0.15)
    base["expected_yellow_cards"] = (base["yellow_cards_per90"] * exposure).clip(0, 0.5)
    base["expected_red_cards"] = (base["red_cards_per90"] * exposure).clip(0, 0.15)
    base["expected_own_goals"] = (base["own_goals_per90"] * exposure).clip(0, 0.15)
    base["expected_bonus"] = (base["bonus_per90"] * exposure).clip(0, 3)
    base["predicted_bps"] = base["bps_per90"] * exposure
    base["expected_points_appearance"] = base["p_appearance"].fillna(0) + base["p_reached_60"].fillna(0)
    goal_pts = base["fpl_position"].map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}).fillna(0)
    base["expected_points_goals"] = base["expected_goals"] * goal_pts
    base["expected_points_assists"] = base["expected_assists"] * 3
    cs_pts = np.select([base["fpl_position"].isin(["GKP", "DEF"]), base["fpl_position"].eq("MID")], [4, 1], 0)
    base["expected_points_clean_sheets"] = base["clean_sheet_probability"] * cs_pts
    base["expected_points_saves"] = base["expected_saves"] / 3
    base["expected_points_penalties"] = base["expected_penalty_saves"] * 5 - base["expected_penalty_misses"] * 2
    base["expected_points_goals_conceded"] = -base["expected_goals_conceded_deduction_events"] * base[
        "fpl_position"
    ].isin(["GKP", "DEF"]).astype(int)
    base["expected_points_cards"] = -base["expected_yellow_cards"] - 3 * base["expected_red_cards"]
    base["expected_points_own_goals"] = -2 * base["expected_own_goals"]
    base["expected_points_bonus"] = base["expected_bonus"]
    return base


def _finalize_model(
    base: pd.DataFrame,
    model_name: str,
    *,
    config: XPointsConfig,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    summaries, draws = simulate_component_points(base, config=config, seed=seed)
    output = pd.concat([base[_id_columns(base)], summaries], axis=1)
    output["model_name"] = model_name
    output["phase4_team_model"] = "T2_REGULARIZED_ATTACK_DEFENCE" if model_name.startswith("X2") else "none"
    output["phase5_minutes_model"] = "M5_REGULARIZED_STATE_SOFTMAX" if model_name.endswith("M5") else "M3_EWMA_MINUTES"
    output["component_model"] = "team_constrained_component_sim" if model_name.startswith("X2") else "independent_component_rates"
    for column in _component_output_columns():
        output[column] = base[column].to_numpy(dtype=float)
    output["team_expected_goals"] = base["team_expected_goals"].to_numpy(dtype=float)
    output["opponent_expected_goals"] = base["opponent_expected_goals"].to_numpy(dtype=float)
    output["team_clean_sheet_probability"] = base["team_clean_sheet_probability"].to_numpy(dtype=float)
    output["goal_share"] = base.get("goal_share", pd.Series(0, index=base.index)).to_numpy(dtype=float)
    output["assist_share"] = base.get("assist_share", pd.Series(0, index=base.index)).to_numpy(dtype=float)
    return output, draws


def _attach_team_context(base: pd.DataFrame, team_predictions: pd.DataFrame) -> pd.DataFrame:
    team_cols = [
        "season",
        "stable_fixture_uid",
        "home_team_uid",
        "away_team_uid",
        "expected_home_goals",
        "expected_away_goals",
        "home_clean_sheet_probability",
        "away_clean_sheet_probability",
    ]
    output = base.merge(team_predictions[team_cols], on=["season", "stable_fixture_uid"], how="left")
    is_home = output["player_team_uid"].eq(output["home_team_uid"])
    output["team_expected_goals"] = np.where(is_home, output["expected_home_goals"], output["expected_away_goals"])
    output["opponent_expected_goals"] = np.where(is_home, output["expected_away_goals"], output["expected_home_goals"])
    output["team_clean_sheet_probability"] = np.where(
        is_home, output["home_clean_sheet_probability"], output["away_clean_sheet_probability"]
    )
    return output.drop(
        columns=[
            "home_team_uid",
            "away_team_uid",
            "expected_home_goals",
            "expected_away_goals",
            "home_clean_sheet_probability",
            "away_clean_sheet_probability",
        ]
    )


def _minutes_variant(minutes_predictions: pd.DataFrame, variant: str) -> pd.DataFrame:
    frame = minutes_predictions.loc[minutes_predictions["minutes_variant"].eq(variant)].copy()
    return frame.rename(columns={"predicted_minutes": "expected_minutes"})[
        [
            "season",
            "stable_fixture_uid",
            "player_uid",
            "expected_minutes",
            "p_appearance",
            "p_start",
            "p_reached_60",
            "p_played_90",
            "cold_start_no_history",
            "evaluation_population",
        ]
    ]


def _conservation_diagnostics(predictions: pd.DataFrame, team_predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby(["model_name", "season", "stable_fixture_uid", "player_team_uid"], as_index=False)
        .agg(
            allocated_expected_goals=("expected_goals", "sum"),
            allocated_expected_assists=("expected_assists", "sum"),
            goal_share_sum=("goal_share", "sum"),
            assist_share_sum=("assist_share", "sum"),
        )
    )
    team_long = pd.concat(
        [
            team_predictions.rename(
                columns={"home_team_uid": "player_team_uid", "expected_home_goals": "team_expected_goals"}
            )[["season", "stable_fixture_uid", "player_team_uid", "team_expected_goals"]],
            team_predictions.rename(
                columns={"away_team_uid": "player_team_uid", "expected_away_goals": "team_expected_goals"}
            )[["season", "stable_fixture_uid", "player_team_uid", "team_expected_goals"]],
        ],
        ignore_index=True,
    )
    output = grouped.merge(team_long, on=["season", "stable_fixture_uid", "player_team_uid"], how="left")
    output["goal_conservation_error"] = output["allocated_expected_goals"] - output["team_expected_goals"]
    return output


def _id_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "season",
        "gameweek",
        "stable_fixture_uid",
        "fixture_key",
        "player_uid",
        "player_name",
        "fpl_position",
        "player_team_uid",
        "opponent_team_uid",
        "was_home",
        "information_cutoff",
        "pre_deadline_population",
        "fold_id",
        "expected_minutes",
        "p_appearance",
        "p_start",
        "p_reached_60",
        "p_played_90",
        "cold_start_no_history",
    ]
    return [column for column in columns if column in frame.columns]


def _component_output_columns() -> list[str]:
    return [
        "expected_goals",
        "expected_assists",
        "clean_sheet_probability",
        "expected_saves",
        "expected_penalty_saves",
        "expected_penalty_misses",
        "expected_yellow_cards",
        "expected_red_cards",
        "expected_own_goals",
        "expected_bonus",
        "expected_points_appearance",
        "expected_points_goals",
        "expected_points_assists",
        "expected_points_clean_sheets",
        "expected_points_saves",
        "expected_points_penalties",
        "expected_points_goals_conceded",
        "expected_points_cards",
        "expected_points_own_goals",
        "expected_points_bonus",
    ]
