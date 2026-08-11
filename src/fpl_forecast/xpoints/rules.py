from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fpl_forecast.ingest.season import parse_season_label


POSITIONS = ("GKP", "DEF", "MID", "FWD")
GOALKEEPER_GOAL_POINTS_CHANGE_SEASON = 2024
OUTFIELD_GOAL_POINTS = {"DEF": 6, "MID": 5, "FWD": 4}


@dataclass(frozen=True)
class ScoringRules:
    version: str = "fpl_standard_2022_2025"
    appearance: int = 1
    appearance_60: int = 1
    assist: int = 3
    clean_sheet_gkp_def: int = 4
    clean_sheet_mid: int = 1
    save_point_per: int = 3
    penalty_save: int = 5
    penalty_miss: int = -2
    goals_conceded_per: int = 2
    goals_conceded_gkp_def: int = -1
    yellow_card: int = -1
    red_card: int = -3
    own_goal: int = -2
    defensive_contribution_points: int = 2
    defensive_contribution_thresholds: dict[str, int] = field(
        default_factory=lambda: {"DEF": 10, "MID": 12, "FWD": 12}
    )


def load_rules(version: str = "fpl_standard_2022_2026") -> ScoringRules:
    if version not in {"fpl_standard_2022_2025", "fpl_standard_2022_2026"}:
        raise ValueError(f"Unknown FPL scoring rules version: {version}")
    return ScoringRules(version=version)


def goal_points_for_position(*, season: str, position: str) -> int:
    """Return the goal value under the rules in force for ``season``."""
    start_year, _ = parse_season_label(season)
    if position == "GKP":
        return 10 if start_year >= GOALKEEPER_GOAL_POINTS_CHANGE_SEASON else 6
    if position in OUTFIELD_GOAL_POINTS:
        return OUTFIELD_GOAL_POINTS[position]
    raise ValueError(f"Invalid standard FPL position: {position!r}.")


def goal_point_values(seasons: pd.Series, positions: pd.Series) -> pd.Series:
    """Vectorized row scoring that fails closed on missing or malformed seasons."""
    if not seasons.index.equals(positions.index):
        raise ValueError("Season and position rows must share the same index.")
    values = [
        goal_points_for_position(season=str(season), position=str(position))
        for season, position in zip(seasons, positions, strict=True)
    ]
    return pd.Series(values, index=seasons.index, dtype="int64")
