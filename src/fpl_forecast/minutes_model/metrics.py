from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl_forecast.minutes_model.config import MinutesModelConfig


@dataclass(frozen=True)
class MinutesMetricTables:
    overall: pd.DataFrame
    by_population: pd.DataFrame
    by_season: pd.DataFrame
    binary: pd.DataFrame
    state: pd.DataFrame
    calibration: pd.DataFrame
    ranking: pd.DataFrame
    bootstrap: pd.DataFrame


def score_minutes_predictions(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    outcome_columns = [
        "season",
        "stable_fixture_uid",
        "player_uid",
        "actual_minutes",
        "actual_appearance",
        "actual_start",
        "actual_reached_60",
        "actual_played_90",
        "actual_state",
        "actual_appearances_diagnostic",
    ]
    scored = predictions.merge(
        outcomes[outcome_columns],
        on=["season", "stable_fixture_uid", "player_uid"],
        how="left",
        validate="many_to_one",
    )
    if scored["actual_minutes"].isna().any():
        raise ValueError("Scored minutes predictions have missing joined outcomes.")
    scored["minutes_error"] = scored["predicted_minutes"] - scored["actual_minutes"]
    scored["state_probability_actual"] = scored.apply(_actual_state_probability, axis=1).clip(1e-12, 1)
    scored["state_log_loss"] = -np.log(scored["state_probability_actual"])
    return scored


def minutes_metric_tables(scored: pd.DataFrame, config: MinutesModelConfig) -> MinutesMetricTables:
    return MinutesMetricTables(
        overall=_minutes_metrics(scored, ["model_name"]),
        by_population=_minutes_metrics(scored, ["evaluation_population", "model_name"]),
        by_season=_minutes_metrics(scored, ["season", "model_name"]),
        binary=_binary_metrics(scored, ["model_name"], config),
        state=_state_metrics(scored, ["model_name"], config),
        calibration=_minutes_calibration(scored, config),
        ranking=_ranking_metrics(scored),
        bootstrap=_bootstrap_minutes(scored, config),
    )


def population_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["season", "gameweek", "evaluation_population"]
    if "model_name" in frame.columns:
        dedup = frame.drop_duplicates(["season", "gameweek", "stable_fixture_uid", "player_uid"])
    else:
        dedup = frame
    return (
        dedup.groupby(columns, as_index=False)
        .agg(rows=("player_uid", "size"), actual_appearances=("actual_appearance", "sum"))
        .sort_values(columns)
    )


def _minutes_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        errors = pd.to_numeric(group["minutes_error"], errors="coerce")
        rows.append(
            {
                **dict(zip(groups, key_values, strict=False)),
                "rows": int(len(group)),
                "mae": float(errors.abs().mean()),
                "rmse": float(math.sqrt(float((errors**2).mean()))),
                "bias": float(errors.mean()),
                "median_absolute_error": float(errors.abs().median()),
                "spearman": _spearman(group["predicted_minutes"], group["actual_minutes"]),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _binary_metrics(frame: pd.DataFrame, groups: list[str], config: MinutesModelConfig) -> pd.DataFrame:
    targets = {
        "appearance": ("p_appearance", "actual_appearance"),
        "start": ("p_start", "actual_start"),
        "reached_60": ("p_reached_60", "actual_reached_60"),
        "played_90": ("p_played_90", "actual_played_90"),
    }
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        for target_name, (pred_col, actual_col) in targets.items():
            actual = pd.to_numeric(group[actual_col], errors="coerce")
            valid = actual.notna()
            if not valid.any():
                continue
            predicted = group.loc[valid, pred_col].clip(1e-12, 1 - 1e-12)
            actual = actual.loc[valid]
            rows.append(
                {
                    **dict(zip(groups, key_values, strict=False)),
                    "target": target_name,
                    "rows": int(valid.sum()),
                    "brier": float(((predicted - actual) ** 2).mean()),
                    "log_loss": float(
                        (-(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))).mean()
                    ),
                    "predicted_rate": float(predicted.mean()),
                    "observed_rate": float(actual.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values([*groups, "target"]).reset_index(drop=True)


def _state_metrics(frame: pd.DataFrame, groups: list[str], config: MinutesModelConfig) -> pd.DataFrame:
    rows = []
    prob_cols = [f"prob_state_{state.lower()}" for state in config.state_order]
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        actual = pd.Categorical(group["actual_state"], categories=config.state_order).codes
        probs = group[prob_cols].clip(1e-12, 1).to_numpy()
        onehot = np.eye(len(config.state_order))[actual]
        rows.append(
            {
                **dict(zip(groups, key_values, strict=False)),
                "rows": int(len(group)),
                "multiclass_log_loss": float(-(onehot * np.log(probs)).sum(axis=1).mean()),
                "multiclass_brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
                "state_accuracy": float((probs.argmax(axis=1) == actual).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _minutes_calibration(frame: pd.DataFrame, config: MinutesModelConfig) -> pd.DataFrame:
    rows = []
    bins = list(config.calibration_bins)
    for model_name, group in frame.groupby("model_name"):
        bucket = pd.cut(
            group["predicted_minutes"],
            bins=bins,
            include_lowest=True,
            right=True,
            labels=False,
        )
        summary = (
            group.assign(prediction_bin=bucket)
            .dropna(subset=["prediction_bin"])
            .groupby("prediction_bin", as_index=False)
            .agg(
                rows=("actual_minutes", "size"),
                mean_prediction=("predicted_minutes", "mean"),
                mean_actual_minutes=("actual_minutes", "mean"),
            )
        )
        summary["model_name"] = model_name
        summary["bin_lower"] = summary["prediction_bin"].map(lambda value: float(bins[int(value)]))
        summary["bin_upper"] = summary["prediction_bin"].map(lambda value: float(bins[int(value) + 1]))
        summary["calibration_gap_actual_minus_prediction"] = (
            summary["mean_actual_minutes"] - summary["mean_prediction"]
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _ranking_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["season", "gameweek", "model_name"]
    for key, group in frame.groupby(groups, dropna=False):
        season, gameweek, model_name = key
        for population in ("all_observed_players", "pre_deadline_history_active"):
            subset = group
            if population == "pre_deadline_history_active":
                subset = group.loc[group["evaluation_population"].eq(population)]
            for k in (15, 30):
                top = subset.sort_values("predicted_minutes", ascending=False).head(k)
                rows.append(
                    {
                        "season": season,
                        "gameweek": int(gameweek),
                        "model_name": model_name,
                        "population": population,
                        "k": k,
                        "candidate_rows": int(len(subset)),
                        "topk_actual_minutes": float(top["actual_minutes"].sum()) if len(top) else 0.0,
                        "topk_appearances": int(top["actual_appearance"].sum()) if len(top) else 0,
                        "topk_starts": int(pd.to_numeric(top["actual_start"], errors="coerce").fillna(0).sum())
                        if len(top)
                        else 0,
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap_minutes(scored: pd.DataFrame, config: MinutesModelConfig) -> pd.DataFrame:
    reference = config.reference_model
    models = [model for model in sorted(scored["model_name"].unique()) if model != reference]
    block_metrics = (
        scored.assign(abs_error=scored["minutes_error"].abs())
        .groupby(["season", "gameweek", "model_name"], as_index=False)
        .agg(mae=("abs_error", "mean"))
    )
    block_metrics["block"] = block_metrics["season"].astype(str) + "_GW" + block_metrics["gameweek"].astype(str)
    blocks = sorted(block_metrics["block"].unique())
    if not blocks or reference not in set(scored["model_name"]):
        return pd.DataFrame()
    pivot = block_metrics.pivot(index="block", columns="model_name", values="mae").reindex(blocks)
    rng = np.random.default_rng(config.bootstrap_seed)
    rows = []
    for model_name in models:
        valid = pivot[[model_name, reference]].dropna()
        if valid.empty:
            continue
        diff_by_block = (valid[model_name] - valid[reference]).to_numpy(dtype=float)
        differences = []
        for _ in range(config.bootstrap_samples):
            sampled = rng.integers(0, len(diff_by_block), len(diff_by_block))
            differences.append(float(diff_by_block[sampled].mean()))
        rows.append(
            {
                "model_name": model_name,
                "reference_model": reference,
                "mean_mae_difference": float(np.mean(differences)),
                "ci_lower": float(np.quantile(differences, 0.025)),
                "ci_upper": float(np.quantile(differences, 0.975)),
                "evaluated_gameweeks": int(len(diff_by_block)),
            }
        )
    return pd.DataFrame(rows)


def _actual_state_probability(row: pd.Series) -> float:
    column = f"prob_state_{str(row['actual_state']).lower()}"
    return float(row[column])


def _spearman(predicted: pd.Series, actual: pd.Series) -> float:
    if len(predicted) < 2:
        return float("nan")
    if predicted.nunique(dropna=True) < 2 or actual.nunique(dropna=True) < 2:
        return float("nan")
    value = predicted.rank(method="average").corr(actual.rank(method="average"))
    return float(value) if pd.notna(value) else float("nan")
