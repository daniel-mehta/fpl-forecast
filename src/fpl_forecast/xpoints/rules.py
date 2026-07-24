from __future__ import annotations

from dataclasses import dataclass, field


POSITIONS = ("GKP", "DEF", "MID", "FWD")


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

    @property
    def goal_points(self) -> dict[str, int]:
        return {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}


def load_rules(version: str = "fpl_standard_2022_2026") -> ScoringRules:
    if version not in {"fpl_standard_2022_2025", "fpl_standard_2022_2026"}:
        raise ValueError(f"Unknown FPL scoring rules version: {version}")
    return ScoringRules(version=version)
