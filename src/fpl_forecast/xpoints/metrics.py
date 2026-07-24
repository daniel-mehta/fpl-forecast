from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl_forecast.xpoints.config import XPointsConfig


@dataclass(frozen=True)
class XPointsMetricTables:
    overall: pd.DataFrame
    by_population: pd.DataFrame
    by_position: pd.DataFrame
    by_season: pd.DataFrame
    calibration: pd.DataFrame
    ranking: pd.DataFrame
    distribution: pd.DataFrame
    components: pd.DataFrame
    bootstrap: pd.DataFrame


def score_predictions(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    outcome_cols = [
        "season",
        "stable_fixture_uid",
        "player_uid",
        "total_points",
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
        "bps",
        "defensive_contribution",
        "minutes",
    ]
    scored = predictions.merge(
        outcomes[outcome_cols],
        on=["season", "stable_fixture_uid", "player_uid"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_actual"),
    )
    if scored["total_points"].isna().any():
        raise ValueError("Scored xPoints predictions have missing target outcomes.")
    scored["actual_total_points"] = pd.to_numeric(scored["total_points"], errors="coerce")
    scored["point_error"] = scored["expected_points"] - scored["actual_total_points"]
    scored["actual_appearance"] = pd.to_numeric(scored["minutes"], errors="coerce").fillna(0).gt(0).astype(int)
    return scored


def metric_tables(scored: pd.DataFrame, config: XPointsConfig) -> XPointsMetricTables:
    return XPointsMetricTables(
        overall=_point_metrics(scored, ["model_name"]),
        by_population=_point_metrics(scored, ["pre_deadline_population", "model_name"]),
        by_position=_point_metrics(scored, ["fpl_position", "model_name"]),
        by_season=_point_metrics(scored, ["season", "model_name"]),
        calibration=_calibration(scored),
        ranking=_ranking(scored),
        distribution=_distribution_metrics(scored),
        components=_component_metrics(scored),
        bootstrap=_bootstrap(scored, config),
    )


def _point_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        errors = group["point_error"]
        rows.append(
            {
                **dict(zip(groups, key_values, strict=False)),
                "rows": int(len(group)),
                "mae": float(errors.abs().mean()),
                "rmse": float(math.sqrt(float((errors**2).mean()))),
                "bias": float(errors.mean()),
                "mean_prediction": float(group["expected_points"].mean()),
                "mean_actual": float(group["actual_total_points"].mean()),
                "spearman": _spearman(group["expected_points"], group["actual_total_points"]),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _calibration(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = [-10, 0, 1, 2, 3, 4, 5, 7, 10, 40]
    for model_name, group in frame.groupby("model_name"):
        bucket = pd.cut(group["expected_points"], bins=bins, include_lowest=True, labels=False)
        summary = (
            group.assign(prediction_bin=bucket)
            .dropna(subset=["prediction_bin"])
            .groupby("prediction_bin", as_index=False)
            .agg(
                rows=("actual_total_points", "size"),
                mean_prediction=("expected_points", "mean"),
                mean_actual=("actual_total_points", "mean"),
            )
        )
        summary["model_name"] = model_name
        summary["bin_lower"] = summary["prediction_bin"].map(lambda value: float(bins[int(value)]))
        summary["bin_upper"] = summary["prediction_bin"].map(lambda value: float(bins[int(value) + 1]))
        summary["actual_minus_prediction"] = summary["mean_actual"] - summary["mean_prediction"]
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _ranking(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(["season", "gameweek", "model_name"], dropna=False):
        season, gameweek, model_name = key
        for population in ("all_observed_players", "pre_deadline_history_active"):
            subset = group
            if population == "pre_deadline_history_active":
                subset = group.loc[group["pre_deadline_population"].eq(population)]
            for k in (15, 30):
                top = subset.sort_values("expected_points", ascending=False).head(k)
                rows.append(
                    {
                        "season": season,
                        "gameweek": int(gameweek),
                        "model_name": model_name,
                        "population": population,
                        "k": k,
                        "candidate_rows": int(len(subset)),
                        "topk_actual_points": float(top["actual_total_points"].sum()) if len(top) else 0,
                        "topk_actual_appearances": int(top["actual_appearance"].sum()) if len(top) else 0,
                    }
                )
    return pd.DataFrame(rows)


def _distribution_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in frame.groupby("model_name"):
        actual = group["actual_total_points"]
        rows.append(
            {
                "model_name": model_name,
                "rows": int(len(group)),
                "prob_ge_1_brier": float(((group["prob_points_ge_1"] - actual.ge(1).astype(int)) ** 2).mean()),
                "prob_ge_5_brier": float(((group["prob_points_ge_5"] - actual.ge(5).astype(int)) ** 2).mean()),
                "prob_ge_10_brier": float(((group["prob_points_ge_10"] - actual.ge(10).astype(int)) ** 2).mean()),
                "zero_rate_predicted": float(group["prob_points_eq_0"].mean()),
                "zero_rate_actual": float(actual.eq(0).mean()),
                "central_50_coverage": float(actual.between(group["points_p25"], group["points_p75"]).mean()),
                "central_80_coverage": float(actual.between(group["points_p10"], group["points_p90"]).mean()),
                "sharpness_p90_p10": float((group["points_p90"] - group["points_p10"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def _component_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    component_pairs = {
        "goals": ("expected_goals", "goals_scored"),
        "assists": ("expected_assists", "assists"),
        "saves": ("expected_saves", "saves"),
        "bonus": ("expected_bonus", "bonus"),
        "yellow_cards": ("expected_yellow_cards", "yellow_cards"),
        "red_cards": ("expected_red_cards", "red_cards"),
        "own_goals": ("expected_own_goals", "own_goals"),
        "defensive_contribution": ("expected_defensive_contribution", "defensive_contribution"),
    }
    rows = []
    for model_name, group in frame.groupby("model_name"):
        for component, (pred_col, actual_col) in component_pairs.items():
            predicted = pd.to_numeric(group[pred_col], errors="coerce").fillna(0)
            actual = pd.to_numeric(group[actual_col], errors="coerce").fillna(0)
            rows.append(
                {
                    "model_name": model_name,
                    "component": component,
                    "rows": int(len(group)),
                    "mean_prediction": float(predicted.mean()),
                    "mean_actual": float(actual.mean()),
                    "mae": float((predicted - actual).abs().mean()),
                    "event_brier": float((((predicted.clip(0, 1)) - actual.gt(0).astype(int)) ** 2).mean())
                    if component != "bonus"
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap(scored: pd.DataFrame, config: XPointsConfig) -> pd.DataFrame:
    reference = f"X0_PHASE3_{config.reference_phase3_baseline}"
    default = config.default_model
    block = (
        scored.assign(abs_error=scored["point_error"].abs())
        .groupby(["season", "gameweek", "model_name"], as_index=False)
        .agg(mae=("abs_error", "mean"))
    )
    pivot = block.pivot(index=["season", "gameweek"], columns="model_name", values="mae")
    if reference not in pivot.columns or default not in pivot.columns:
        return pd.DataFrame()
    diffs = (pivot[default] - pivot[reference]).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(config.bootstrap_seed)
    samples = []
    for _ in range(config.bootstrap_samples):
        samples.append(float(diffs[rng.integers(0, len(diffs), len(diffs))].mean()))
    return pd.DataFrame(
        [
            {
                "model_name": default,
                "reference_model": reference,
                "mean_mae_difference": float(np.mean(samples)),
                "ci_lower": float(np.quantile(samples, 0.025)),
                "ci_upper": float(np.quantile(samples, 0.975)),
                "evaluated_gameweeks": int(len(diffs)),
            }
        ]
    )


def _spearman(predicted: pd.Series, actual: pd.Series) -> float:
    if len(predicted) < 2 or predicted.nunique(dropna=True) < 2 or actual.nunique(dropna=True) < 2:
        return float("nan")
    value = predicted.rank().corr(actual.rank())
    return float(value) if pd.notna(value) else float("nan")
