from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_forecast.xpoints.rules import POSITIONS, ScoringRules, load_rules


COMPONENT_COLUMNS = [
    "minutes",
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
    "defensive_contribution",
]


def score_frame(frame: pd.DataFrame, *, rules: ScoringRules | None = None) -> pd.Series:
    rules = rules or load_rules()
    _validate_positions(frame)
    pos = frame["fpl_position"].astype(str)
    minutes = _num(frame, "minutes")
    points = (minutes > 0).astype(int) * rules.appearance
    points += (minutes >= 60).astype(int) * rules.appearance_60
    points += _num(frame, "goals_scored") * pos.map(rules.goal_points).fillna(0).astype(int)
    points += _num(frame, "assists") * rules.assist
    clean = _num(frame, "clean_sheets")
    points += np.where(pos.isin(["GKP", "DEF"]) & (minutes >= 60), clean * rules.clean_sheet_gkp_def, 0)
    points += np.where(pos.eq("MID") & (minutes >= 60), clean * rules.clean_sheet_mid, 0)
    points += (_num(frame, "saves") // rules.save_point_per).astype(int)
    points += _num(frame, "penalties_saved") * rules.penalty_save
    points += _num(frame, "penalties_missed") * rules.penalty_miss
    points += np.where(
        pos.isin(["GKP", "DEF"]),
        (_num(frame, "goals_conceded") // rules.goals_conceded_per).astype(int)
        * rules.goals_conceded_gkp_def,
        0,
    )
    points += _num(frame, "yellow_cards") * rules.yellow_card
    points += _num(frame, "red_cards") * rules.red_card
    points += _num(frame, "own_goals") * rules.own_goal
    points += _num(frame, "bonus")
    thresholds = pos.map(rules.defensive_contribution_thresholds)
    defensive_contribution = _num(frame, "defensive_contribution")
    points += np.where(
        thresholds.notna() & defensive_contribution.ge(thresholds.fillna(10**9).astype(int)),
        rules.defensive_contribution_points,
        0,
    )
    return pd.Series(points, index=frame.index).astype(int)


def component_points(frame: pd.DataFrame, *, rules: ScoringRules | None = None) -> pd.DataFrame:
    rules = rules or load_rules()
    pos = frame["fpl_position"].astype(str)
    minutes = _num(frame, "minutes")
    output = pd.DataFrame(index=frame.index)
    output["points_appearance"] = (minutes > 0).astype(int) + (minutes >= 60).astype(int)
    output["points_goals"] = _num(frame, "goals_scored") * pos.map(rules.goal_points).fillna(0).astype(int)
    output["points_assists"] = _num(frame, "assists") * rules.assist
    clean = _num(frame, "clean_sheets")
    output["points_clean_sheets"] = np.where(pos.isin(["GKP", "DEF"]) & (minutes >= 60), clean * 4, 0)
    output["points_clean_sheets"] += np.where(pos.eq("MID") & (minutes >= 60), clean, 0)
    output["points_saves"] = (_num(frame, "saves") // rules.save_point_per).astype(int)
    output["points_penalties"] = _num(frame, "penalties_saved") * 5 - _num(frame, "penalties_missed") * 2
    output["points_goals_conceded"] = np.where(
        pos.isin(["GKP", "DEF"]), -(_num(frame, "goals_conceded") // 2).astype(int), 0
    )
    output["points_cards"] = -_num(frame, "yellow_cards") - 3 * _num(frame, "red_cards")
    output["points_own_goals"] = -2 * _num(frame, "own_goals")
    output["points_bonus"] = _num(frame, "bonus")
    thresholds = pos.map(rules.defensive_contribution_thresholds)
    defensive_contribution = _num(frame, "defensive_contribution")
    output["points_defensive_contribution"] = np.where(
        thresholds.notna() & defensive_contribution.ge(thresholds.fillna(10**9).astype(int)),
        rules.defensive_contribution_points,
        0,
    )
    return output


def reconstruction_audit(frame: pd.DataFrame, *, rules: ScoringRules | None = None) -> dict[str, pd.DataFrame]:
    players = frame.loc[frame["entity_type"].eq("player")].copy()
    players["reconstructed_points"] = score_frame(players, rules=rules)
    players["actual_total_points"] = _num(players, "total_points")
    players["point_difference"] = players["reconstructed_points"] - players["actual_total_points"]
    by_season_position = (
        players.groupby(["season", "fpl_position"], as_index=False)
        .agg(
            rows=("point_difference", "size"),
            exact_matches=("point_difference", lambda values: int(values.eq(0).sum())),
            mean_abs_difference=("point_difference", lambda values: float(values.abs().mean())),
            max_abs_difference=("point_difference", lambda values: int(values.abs().max())),
        )
        .sort_values(["season", "fpl_position"])
    )
    by_season_position["exact_match_pct"] = (
        by_season_position["exact_matches"] / by_season_position["rows"]
    )
    diff_counts = (
        players["point_difference"].value_counts().rename_axis("point_difference").reset_index(name="rows")
    ).sort_values("point_difference")
    component_frame = players[["season", "fpl_position"]].copy()
    for column in COMPONENT_COLUMNS:
        component_frame[column] = players[column] if column in players.columns else 0
    component_nulls = (
        component_frame[["season", "fpl_position", *COMPONENT_COLUMNS]]
        .groupby(["season", "fpl_position"], as_index=False)
        .agg({column: lambda values: int(values.isna().sum()) for column in COMPONENT_COLUMNS})
    )
    mismatches = players.loc[players["point_difference"].ne(0), _mismatch_columns(players)].head(50)
    return {
        "by_season_position": by_season_position,
        "difference_counts": diff_counts,
        "component_nulls": component_nulls,
        "mismatches": mismatches,
    }


def _mismatch_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "season",
        "gameweek",
        "player_uid",
        "player_name",
        "fpl_position",
        "minutes",
        "total_points",
        "reconstructed_points",
        "point_difference",
        *COMPONENT_COLUMNS[1:],
    ]
    return [column for column in columns if column in frame.columns]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype="int64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)


def _validate_positions(frame: pd.DataFrame) -> None:
    invalid = sorted(set(frame["fpl_position"].dropna().astype(str)).difference(POSITIONS))
    if invalid:
        raise ValueError(f"Invalid standard FPL position(s): {', '.join(invalid)}")
