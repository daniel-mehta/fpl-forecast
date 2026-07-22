from __future__ import annotations

import pandas as pd


ALL_OBSERVED = "all_observed_standard_players"
PRE_DEADLINE_HISTORY_ACTIVE = "pre_deadline_history_active"
COLD_START_NO_HISTORY = "cold_start_no_history"
ACTUAL_APPEARANCES_DIAGNOSTIC = "actual_appearances_diagnostic"


def assign_pre_deadline_populations(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    gameweek = pd.to_numeric(output["gameweek"], errors="coerce")
    prev5_minutes = pd.to_numeric(output.get("prev5_minutes_sum"), errors="coerce").fillna(0)
    season_to_date_appearances = pd.to_numeric(
        output.get("season_to_date_appearances"),
        errors="coerce",
    ).fillna(0)
    prior_minutes = pd.to_numeric(output.get("prior_season_minutes"), errors="coerce").fillna(0)
    prior_appearances = pd.to_numeric(
        output.get("prior_season_appearances"),
        errors="coerce",
    ).fillna(0)

    gw1_active = (gameweek == 1) & ((prior_minutes > 0) | (prior_appearances > 0))
    in_season_active = (gameweek != 1) & (
        (prev5_minutes > 0) | (season_to_date_appearances > 0)
    )
    any_history = (
        (prev5_minutes > 0)
        | (season_to_date_appearances > 0)
        | (prior_minutes > 0)
        | (prior_appearances > 0)
    )
    output["pre_deadline_population"] = "pre_deadline_inactive_with_history"
    output.loc[gw1_active | in_season_active, "pre_deadline_population"] = (
        PRE_DEADLINE_HISTORY_ACTIVE
    )
    output.loc[~any_history, "pre_deadline_population"] = COLD_START_NO_HISTORY
    output["is_pre_deadline_history_active"] = output["pre_deadline_population"].eq(
        PRE_DEADLINE_HISTORY_ACTIVE
    )
    output["is_cold_start_no_history"] = output["pre_deadline_population"].eq(COLD_START_NO_HISTORY)
    return output


def population_coverage(scored_player_gameweek: pd.DataFrame) -> pd.DataFrame:
    base = scored_player_gameweek.drop_duplicates(["season", "gameweek", "player_uid"]).copy()
    rows: list[dict[str, object]] = []
    for (season, gameweek), group in base.groupby(["season", "gameweek"], sort=True):
        total = len(group)
        active = int(group["pre_deadline_population"].eq(PRE_DEADLINE_HISTORY_ACTIVE).sum())
        cold = int(group["pre_deadline_population"].eq(COLD_START_NO_HISTORY).sum())
        actual = int(pd.to_numeric(group["minutes"], errors="coerce").fillna(0).gt(0).sum())
        rows.append(
            {
                "season": season,
                "gameweek": int(gameweek),
                "all_observed_rows": total,
                "pre_deadline_history_active_rows": active,
                "cold_start_no_history_rows": cold,
                "actual_appearances_rows": actual,
                "pre_deadline_history_active_share": active / total if total else 0.0,
                "cold_start_no_history_share": cold / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)
