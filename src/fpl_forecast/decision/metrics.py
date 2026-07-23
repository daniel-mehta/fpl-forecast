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
                "mean_mean_only_objective": float(group["mean_only_objective"].mean()),
                "mean_appearance_aware_advantage": float(
                    (group["objective"] - group["mean_only_objective"]).mean()
                ),
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
    left_frame = scored.loc[scored["model_name"].eq(left)][keys + ["realized_points", "captain", "lineup"]]
    right_frame = scored.loc[scored["model_name"].eq(right)][keys + ["realized_points", "captain", "lineup"]]
    merged = left_frame.merge(right_frame, on=keys, suffixes=("_left", "_right"))
    if merged.empty:
        return pd.DataFrame()
    differences = merged["realized_points_left"] - merged["realized_points_right"]
    bootstrap = _paired_gameweek_bootstrap(differences)
    return pd.DataFrame(
        [
            {
                "left_model": left,
                "right_model": right,
                "matched_decisions": int(len(merged)),
                "mean_realized_difference": float(differences.mean()),
                "bootstrap_ci_low": bootstrap[0],
                "bootstrap_ci_high": bootstrap[1],
                "captain_agreement": float(merged["captain_left"].eq(merged["captain_right"]).mean()),
                "mean_lineup_overlap": float(
                    merged.apply(
                        lambda row: len(set(str(row["lineup_left"]).split(",")) & set(str(row["lineup_right"]).split(",")))
                        / 11,
                        axis=1,
                    ).mean()
                ),
            }
        ]
    )


def selected_player_calibration(squads: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    frame = squads.copy()
    frame["expected_points"] = pd.to_numeric(frame["expected_points"], errors="coerce")
    frame["actual_points"] = pd.to_numeric(frame["actual_points"], errors="coerce")
    frame["prediction_bin"] = frame.groupby("model_name")["expected_points"].transform(
        lambda values: pd.qcut(values.rank(method="first"), q=bins, labels=False, duplicates="drop")
    )
    rows = []
    for (model_name, prediction_bin), group in frame.groupby(["model_name", "prediction_bin"], dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "prediction_bin": int(prediction_bin) if pd.notna(prediction_bin) else -1,
                "selected_players": int(len(group)),
                "mean_prediction": float(group["expected_points"].mean()),
                "mean_actual_points": float(group["actual_points"].mean()),
                "bias": float((group["expected_points"] - group["actual_points"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "prediction_bin"])


def _paired_gameweek_bootstrap(differences: pd.Series, samples: int = 1000, seed: int = 7) -> tuple[float, float]:
    if differences.empty:
        return (float("nan"), float("nan"))
    sampled = []
    values = differences.to_numpy(dtype=float)
    generator = pd.Series(range(samples)).sample(frac=1, random_state=seed).index
    for sample_index in generator:
        random_state = seed + int(sample_index)
        sample = pd.Series(values).sample(n=len(values), replace=True, random_state=random_state)
        sampled.append(float(sample.mean()))
    quantiles = pd.Series(sampled).quantile([0.025, 0.975])
    return (float(quantiles.iloc[0]), float(quantiles.iloc[1]))
