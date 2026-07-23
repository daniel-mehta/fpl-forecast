from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_forecast.minutes_model.config import MinutesModelConfig


START_STATES = ("start_under_60", "start_60_to_89", "start_90")
NONSTART_STATES = ("dnp", "sub_under_60", "sub_60_plus")


def add_lineup_coherence(predictions: pd.DataFrame, config: MinutesModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = predictions.copy()
    output["p_start_raw"] = output["p_start"]
    output["lineup_adjustment_applied"] = False
    adjusted_frames = []
    diagnostics = []
    group_columns = ["model_name", "season", "stable_fixture_uid", "player_team_uid"]
    for key, group in output.groupby(group_columns, dropna=False):
        model_name, season, fixture_uid, team_uid = key
        raw_sum = float(group["p_start_raw"].sum())
        complete = len(group) >= config.lineup_adjustment_min_candidates
        adjusted = group.copy()
        if complete and raw_sum > 1e-9:
            adjusted = _adjust_group_start_probabilities(adjusted, target_starters=11.0)
        diagnostics.append(
            {
                "model_name": model_name,
                "season": season,
                "stable_fixture_uid": fixture_uid,
                "player_team_uid": team_uid,
                "candidate_rows": int(len(group)),
                "raw_start_sum": raw_sum,
                "adjusted_start_sum": float(adjusted["p_start"].sum()),
                "lineup_adjustment_applied": bool(adjusted["lineup_adjustment_applied"].any()),
                "candidate_reconstruction_complete": bool(complete),
            }
        )
        adjusted_frames.append(adjusted)
    return pd.concat(adjusted_frames, ignore_index=True), pd.DataFrame(diagnostics)


def _adjust_group_start_probabilities(group: pd.DataFrame, *, target_starters: float) -> pd.DataFrame:
    output = group.copy()
    old = output["p_start_raw"].clip(1e-6, 1 - 1e-6).to_numpy()
    low, high = -30.0, 30.0
    for _ in range(80):
        mid = (low + high) / 2
        shifted = 1 / (1 + np.exp(-(np.log(old / (1 - old)) + mid)))
        if shifted.sum() > target_starters:
            high = mid
        else:
            low = mid
    new = 1 / (1 + np.exp(-(np.log(old / (1 - old)) + (low + high) / 2)))
    output["p_start"] = new
    old_start = output["p_start_raw"].clip(1e-9, 1 - 1e-9)
    new_start = output["p_start"].clip(1e-9, 1 - 1e-9)
    start_scale = new_start / old_start
    nonstart_scale = (1 - new_start) / (1 - old_start)
    for suffix in START_STATES:
        output[f"prob_state_{suffix}"] *= start_scale
    for suffix in NONSTART_STATES:
        output[f"prob_state_{suffix}"] *= nonstart_scale
    prob_cols = [column for column in output.columns if column.startswith("prob_state_")]
    denom = output[prob_cols].sum(axis=1).replace(0, 1)
    output[prob_cols] = output[prob_cols].div(denom, axis=0)
    output["p_appearance"] = 1 - output["prob_state_dnp"]
    output["p_reached_60"] = (
        output["prob_state_sub_60_plus"]
        + output["prob_state_start_60_to_89"]
        + output["prob_state_start_90"]
    )
    output["p_played_90"] = output["prob_state_start_90"]
    output["predicted_minutes"] = (
        output["prob_state_sub_under_60"] * 22
        + output["prob_state_sub_60_plus"] * 65
        + output["prob_state_start_under_60"] * 45
        + output["prob_state_start_60_to_89"] * 75
        + output["prob_state_start_90"] * 90
    )
    output["lineup_adjustment_applied"] = True
    return output
