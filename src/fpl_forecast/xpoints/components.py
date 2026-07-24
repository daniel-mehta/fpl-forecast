from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_forecast.xpoints.config import XPointsConfig


EVENT_COLUMNS = [
    "goals_scored",
    "assists",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "bonus",
    "bps",
    "defensive_contribution",
]


def fit_component_priors(train: pd.DataFrame, config: XPointsConfig) -> dict[str, object]:
    frame = train.loc[train["entity_type"].eq("player")].copy()
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0)
    frame["appearance"] = frame["minutes"].gt(0).astype(int)
    pos_rates = {}
    for column in EVENT_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        rates = {}
        for position, group in frame.assign(value=values).groupby("fpl_position"):
            minutes = float(group["minutes"].sum())
            rates[position] = 0.0 if minutes <= 0 else float(group["value"].sum() * 90.0 / minutes)
        pos_rates[column] = rates
    player_rates = _player_rates(frame, pos_rates, config)
    team_goal_history = _team_goal_history(frame)
    assisted_rate = _safe_ratio(frame["assists"].sum(), frame["goals_scored"].sum(), config.assisted_goal_prior)
    return {
        "position_rates": pos_rates,
        "player_rates": player_rates,
        "team_goal_history": team_goal_history,
        "assisted_goal_rate": float(np.clip(assisted_rate, 0.25, 0.95)),
    }


def player_component_rates(test: pd.DataFrame, priors: dict[str, object], config: XPointsConfig) -> pd.DataFrame:
    output = test[["season", "stable_fixture_uid", "player_uid", "fpl_position", "player_team_uid"]].copy()
    rates = priors["player_rates"]
    pos_rates = priors["position_rates"]
    for column in EVENT_COLUMNS:
        merged = output[["player_uid", "fpl_position"]].merge(
            rates.get(column, pd.DataFrame()),
            on="player_uid",
            how="left",
        )
        fallback = merged["fpl_position"].map(pos_rates[column]).fillna(0).astype(float)
        output[f"{column}_per90"] = (
            merged["rate_per90"].fillna(fallback).astype(float).clip(lower=0).to_numpy()
        )
        if "transferred_player" in test.columns:
            transferred = test["transferred_player"].astype(bool).to_numpy()
            output.loc[transferred, f"{column}_per90"] = (
                output.loc[transferred, f"{column}_per90"].to_numpy(dtype=float) * 0.72
                + fallback.loc[transferred].to_numpy(dtype=float) * 0.28
            )
        if "cold_start_no_history" in test.columns:
            cold = test["cold_start_no_history"].astype(bool).to_numpy()
            output.loc[cold, f"{column}_per90"] = fallback.loc[cold].to_numpy(dtype=float)
        output[f"{column}_fallback"] = merged["rate_per90"].isna().to_numpy()
    output["assisted_goal_rate"] = float(priors["assisted_goal_rate"])
    return output


def attacking_shares(rates: pd.DataFrame, minutes: pd.DataFrame, config: XPointsConfig) -> pd.DataFrame:
    frame = rates.merge(
        minutes[
            [
                "season",
                "stable_fixture_uid",
                "player_uid",
                "expected_minutes",
                "p_appearance",
                "p_start",
                "cold_start_no_history",
                "transferred_player",
            ]
        ],
        on=["season", "stable_fixture_uid", "player_uid"],
        how="left",
    )
    expected_minutes = pd.to_numeric(frame["expected_minutes"], errors="coerce").fillna(0).clip(0, 90)
    appearance = pd.to_numeric(frame["p_appearance"], errors="coerce").fillna(0).clip(0, 1)
    cold_start = frame.get("cold_start_no_history", pd.Series(False, index=frame.index)).astype(bool)
    frame["active_weight"] = np.maximum(expected_minutes / 90, appearance * 0.10)
    frame.loc[cold_start, "active_weight"] = np.maximum(
        frame.loc[cold_start, "active_weight"],
        frame.loc[cold_start, "p_appearance"].fillna(0).clip(0, 1) * 0.35,
    )
    frame["goal_weight"] = (
        pd.to_numeric(frame["goals_scored_per90"], errors="coerce").fillna(0).clip(lower=0)
        * frame["active_weight"]
    )
    frame["assist_weight"] = (
        pd.to_numeric(frame["assists_per90"], errors="coerce").fillna(0).clip(lower=0)
        * frame["active_weight"]
    )
    groups = ["season", "stable_fixture_uid", "player_team_uid"]
    for column in ("goal_weight", "assist_weight"):
        frame[column] = frame[column].where(frame[column].gt(0), _position_prior_weight(frame, column, config))
        zero_groups = frame.groupby(groups)[column].transform("sum").le(0)
        frame.loc[zero_groups, column] = frame.loc[zero_groups, "fpl_position"].map(
            config.position_goal_prior_per90 if column == "goal_weight" else config.position_assist_prior_per90
        ).fillna(0.01)
    frame["goal_share"] = frame["goal_weight"] / frame.groupby(groups)["goal_weight"].transform("sum").replace(0, np.nan)
    frame["assist_share"] = frame["assist_weight"] / frame.groupby(groups)["assist_weight"].transform("sum").replace(0, np.nan)
    frame["goal_share"] = frame["goal_share"].fillna(0)
    frame["assist_share"] = frame["assist_share"].fillna(0)
    for share_col, prefix in (("goal_share", "goal"), ("assist_share", "assist")):
        share_sum_sq = frame.groupby(groups)[share_col].transform(lambda values: float((values.astype(float) ** 2).sum()))
        frame[f"{prefix}_share_hhi"] = share_sum_sq
        frame[f"effective_{prefix}_attackers"] = np.where(share_sum_sq > 0, 1.0 / share_sum_sq, 0.0)
        weight_col = f"{prefix}_weight"
        frame[f"nonzero_{prefix}_weight_players"] = frame.groupby(groups)[weight_col].transform(
            lambda values: int(pd.to_numeric(values, errors="coerce").fillna(0).gt(0).sum())
        )
    frame["attacking_share_hhi"] = frame[["goal_share_hhi", "assist_share_hhi"]].max(axis=1)
    return frame


def award_bonus_from_bps(frame: pd.DataFrame) -> pd.Series:
    eligible = frame.loc[frame["eligible"].astype(bool)].copy()
    bonus = pd.Series(0, index=frame.index, dtype="int64")
    if eligible.empty:
        return bonus
    bps = pd.to_numeric(eligible["predicted_bps"], errors="coerce").fillna(-9999)
    unique_scores = sorted(bps.unique(), reverse=True)
    awarded_slots = 0
    for score in unique_scores:
        tied = eligible.index[bps.eq(score)]
        if awarded_slots == 0:
            points = 3
        elif awarded_slots == 1:
            points = 2
        elif awarded_slots == 2:
            points = 1
        else:
            break
        bonus.loc[tied] = points
        awarded_slots += len(tied)
        if awarded_slots >= 3:
            break
    return bonus


def _player_rates(
    frame: pd.DataFrame,
    pos_rates: dict[str, dict[str, float]],
    config: XPointsConfig,
) -> dict[str, pd.DataFrame]:
    rows = {}
    minutes = frame.groupby("player_uid")["minutes"].sum()
    positions = (
        frame.sort_values(["season", "gameweek"])
        .groupby("player_uid")["fpl_position"]
        .agg(lambda values: values.dropna().astype(str).iloc[-1] if len(values.dropna()) else "MID")
    )
    for column in EVENT_COLUMNS:
        totals = pd.to_numeric(frame[column], errors="coerce").fillna(0).groupby(frame["player_uid"]).sum()
        table = pd.DataFrame({"player_uid": totals.index, "event_total": totals.values})
        table["minutes"] = table["player_uid"].map(minutes).fillna(0)
        table["fpl_position"] = table["player_uid"].map(positions).fillna("MID")
        table["position_prior_per90"] = table["fpl_position"].map(pos_rates[column]).fillna(
            _safe_ratio(frame[column].sum() * 90, frame["minutes"].sum(), 0.0)
        )
        table["raw_rate_per90"] = np.where(table["minutes"] > 0, table["event_total"] * 90 / table["minutes"], 0)
        table["rate_per90"] = (
            (table["event_total"] * 90) + (table["position_prior_per90"] * config.component_shrink_minutes)
        ) / (table["minutes"] + config.component_shrink_minutes)
        table["shrink_minutes"] = config.component_shrink_minutes
        rows[column] = table[
            [
                "player_uid",
                "rate_per90",
                "raw_rate_per90",
                "position_prior_per90",
                "shrink_minutes",
                "minutes",
            ]
        ]
    return rows


def _team_goal_history(frame: pd.DataFrame) -> pd.DataFrame:
    goals = (
        frame.groupby(["season", "fixture_key", "player_team_uid"], as_index=False)["goals_scored"]
        .sum()
        .rename(columns={"fixture_key": "stable_fixture_uid", "goals_scored": "allocated_actual_team_goals"})
    )
    return goals


def _safe_ratio(numerator: float, denominator: float, fallback: float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return fallback
    return float(numerator) / denominator


def _position_prior_weight(frame: pd.DataFrame, column: str, config: XPointsConfig) -> pd.Series:
    priors = config.position_goal_prior_per90 if column == "goal_weight" else config.position_assist_prior_per90
    active = pd.to_numeric(frame["active_weight"], errors="coerce").fillna(0).clip(lower=0)
    return frame["fpl_position"].map(priors).fillna(0.01).astype(float) * active
