from __future__ import annotations

import pandas as pd


def decision_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in scored.groupby("model_name"):
        rows.append(
            {
                "model_name": model_name,
                "decisions": int(len(group)),
                "mean_expected_score": float(group["objective"].mean()),
                "mean_realized_points": float(group["realized_points"].mean()),
                "mean_captain_bonus": float(group["captain_bonus_points"].mean()),
                "mean_autosub_count": float(group["autosub_count"].mean()),
                "feasible_solution_rate": float(
                    group["solver_status"].isin(["optimal", "heuristic_feasible"]).mean()
                ),
                "exact_optimal_solution_rate": float(group["solver_status"].eq("optimal").mean()),
                "mean_bank_tenths": float(group["bank_tenths"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_realized_points", ascending=False)


def compare_models(scored: pd.DataFrame, *, left: str, right: str) -> pd.DataFrame:
    keys = ["season", "gameweek"]
    left_frame = scored.loc[scored["model_name"].eq(left)][keys + ["realized_points", "captain"]]
    right_frame = scored.loc[scored["model_name"].eq(right)][keys + ["realized_points", "captain"]]
    merged = left_frame.merge(right_frame, on=keys, suffixes=("_left", "_right"))
    if merged.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "left_model": left,
                "right_model": right,
                "matched_decisions": int(len(merged)),
                "mean_realized_difference": float(
                    (merged["realized_points_left"] - merged["realized_points_right"]).mean()
                ),
                "captain_agreement": float(merged["captain_left"].eq(merged["captain_right"]).mean()),
            }
        ]
    )
