from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from fpl_forecast.backtest.populations import (
    ACTUAL_APPEARANCES_DIAGNOSTIC,
    ALL_OBSERVED,
    COLD_START_NO_HISTORY,
    PRE_DEADLINE_HISTORY_ACTIVE,
)


@dataclass(frozen=True)
class MetricTables:
    overall: pd.DataFrame
    by_season: pd.DataFrame
    by_gameweek: pd.DataFrame
    by_position: pd.DataFrame
    by_population: pd.DataFrame
    calibration: pd.DataFrame
    ranking: pd.DataFrame
    ranking_by_population: pd.DataFrame
    bootstrap: pd.DataFrame


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.copy()
    scored["target_total_points"] = pd.to_numeric(scored["target_total_points"], errors="coerce")
    scored["prediction"] = pd.to_numeric(scored["prediction"], errors="coerce")
    scored["error"] = scored["prediction"] - scored["target_total_points"]
    scored["absolute_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"] ** 2
    if "pre_deadline_population" not in scored.columns:
        scored["pre_deadline_population"] = "unknown_population"
    scored["population"] = ALL_OBSERVED
    return scored


def aggregate_player_gameweek(scored: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "gameweek", "player_uid", "baseline"]
    keep = ["player_name", "fpl_position", "information_cutoff", "population", "pre_deadline_population"]
    grouped = (
        scored.groupby(keys, as_index=False)
        .agg(
            prediction=("prediction", "sum"),
            target_total_points=("target_total_points", "sum"),
            fixture_count=("fixture_key", "nunique"),
            minutes=("minutes", "sum"),
            **{column: (column, "first") for column in keep},
        )
    )
    return score_predictions(grouped)


def metric_tables(
    scored_fixture: pd.DataFrame,
    scored_player_gameweek: pd.DataFrame,
    *,
    reference_baseline: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> MetricTables:
    return MetricTables(
        overall=_metric_group(scored_player_gameweek, ["baseline"]),
        by_season=_metric_group(scored_player_gameweek, ["season", "baseline"]),
        by_gameweek=_metric_group(scored_player_gameweek, ["season", "gameweek", "baseline"]),
        by_position=_metric_group(scored_player_gameweek, ["fpl_position", "baseline"]),
        by_population=_population_metrics(scored_player_gameweek),
        calibration=calibration_table(scored_player_gameweek),
        ranking=ranking_table(scored_player_gameweek),
        ranking_by_population=ranking_by_population_table(scored_player_gameweek),
        bootstrap=block_bootstrap_differences(
            scored_player_gameweek,
            reference_baseline=reference_baseline,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    )


def _metric_group(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key_values, strict=False))
        errors = pd.to_numeric(group["error"], errors="coerce")
        absolute = pd.to_numeric(group["absolute_error"], errors="coerce")
        squared = pd.to_numeric(group["squared_error"], errors="coerce")
        record.update(
            {
                "rows": int(len(group)),
                "mae": float(absolute.mean()),
                "rmse": float(math.sqrt(float(squared.mean()))),
                "bias": float(errors.mean()),
                "median_ae": float(absolute.median()),
                "spearman": _spearman(group["prediction"], group["target_total_points"]),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _population_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    all_rows = frame.assign(population=ALL_OBSERVED)
    active = frame.loc[frame["pre_deadline_population"].eq(PRE_DEADLINE_HISTORY_ACTIVE)].assign(
        population=PRE_DEADLINE_HISTORY_ACTIVE
    )
    cold = frame.loc[frame["pre_deadline_population"].eq(COLD_START_NO_HISTORY)].assign(
        population=COLD_START_NO_HISTORY
    )
    appearances = frame.loc[pd.to_numeric(frame["minutes"], errors="coerce").fillna(0) > 0].assign(
        population=ACTUAL_APPEARANCES_DIAGNOSTIC
    )
    return _metric_group(
        pd.concat([all_rows, active, cold, appearances], ignore_index=True),
        ["population", "baseline"],
    )


def _spearman(prediction: pd.Series, actual: pd.Series) -> float | None:
    pred_rank = pd.to_numeric(prediction, errors="coerce").rank(method="average")
    actual_rank = pd.to_numeric(actual, errors="coerce").rank(method="average")
    if pred_rank.nunique(dropna=True) <= 1 or actual_rank.nunique(dropna=True) <= 1:
        return None
    value = pred_rank.corr(actual_rank)
    return None if pd.isna(value) else float(value)


def calibration_table(frame: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for baseline, group in frame.groupby("baseline"):
        ranked = group.copy()
        ranked["prediction_bin"] = pd.qcut(
            ranked["prediction"].rank(method="first"),
            q=min(bins, len(ranked)),
            labels=False,
            duplicates="drop",
        )
        summary = (
            ranked.groupby("prediction_bin", as_index=False)
            .agg(
                rows=("player_uid", "size"),
                prediction_mean=("prediction", "mean"),
                actual_mean=("target_total_points", "mean"),
            )
            .assign(baseline=baseline)
        )
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[
        ["baseline", "prediction_bin", "rows", "prediction_mean", "actual_mean"]
    ]


def ranking_table(frame: pd.DataFrame, ks: tuple[int, ...] = (15, 30)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (season, gameweek, baseline), group in frame.groupby(["season", "gameweek", "baseline"]):
        actual_order = group.sort_values("target_total_points", ascending=False)
        predicted_order = group.sort_values("prediction", ascending=False)
        for k in ks:
            predicted_top = set(predicted_order.head(k)["player_uid"])
            actual_top = set(actual_order.head(k)["player_uid"])
            rows.append(
                {
                    "season": season,
                    "gameweek": int(gameweek),
                    "baseline": baseline,
                    "k": k,
                    "top_k_actual_points": float(
                        predicted_order.head(k)["target_total_points"].sum()
                    ),
                    "top_k_overlap": len(predicted_top & actual_top) / max(1, len(actual_top)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby(["baseline", "k"], as_index=False)
        .agg(
            evaluated_gameweeks=("gameweek", "size"),
            mean_top_k_actual_points=("top_k_actual_points", "mean"),
            mean_top_k_overlap=("top_k_overlap", "mean"),
        )
    )


def ranking_by_population_table(frame: pd.DataFrame, ks: tuple[int, ...] = (15, 30)) -> pd.DataFrame:
    rows = []
    populations = [
        (ALL_OBSERVED, frame),
        (PRE_DEADLINE_HISTORY_ACTIVE, frame.loc[frame["pre_deadline_population"].eq(PRE_DEADLINE_HISTORY_ACTIVE)]),
        (COLD_START_NO_HISTORY, frame.loc[frame["pre_deadline_population"].eq(COLD_START_NO_HISTORY)]),
        (
            ACTUAL_APPEARANCES_DIAGNOSTIC,
            frame.loc[pd.to_numeric(frame["minutes"], errors="coerce").fillna(0) > 0],
        ),
    ]
    for population, group in populations:
        if group.empty:
            continue
        ranked = ranking_table(group, ks=ks)
        if ranked.empty:
            continue
        ranked["population"] = population
        rows.append(ranked)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[
        [
            "population",
            "baseline",
            "k",
            "evaluated_gameweeks",
            "mean_top_k_actual_points",
            "mean_top_k_overlap",
        ]
    ]


def block_bootstrap_differences(
    frame: pd.DataFrame,
    *,
    reference_baseline: str,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    pivot = (
        frame.groupby(["season", "gameweek", "baseline"], as_index=False)["absolute_error"]
        .mean()
        .pivot(index=["season", "gameweek"], columns="baseline", values="absolute_error")
        .dropna()
    )
    if pivot.empty or reference_baseline not in pivot.columns:
        return pd.DataFrame()
    rng = pd.Series(range(len(pivot))).sample
    rows: list[dict[str, object]] = []
    for baseline in pivot.columns:
        if baseline == reference_baseline:
            continue
        diffs = pivot[baseline] - pivot[reference_baseline]
        sampled_values: list[float] = []
        for sample_index in range(samples):
            sample = rng(n=len(diffs), replace=True, random_state=seed + sample_index)
            sampled_values.append(float(diffs.iloc[sample.to_numpy()].mean()))
        series = pd.Series(sampled_values)
        rows.append(
            {
                "baseline": baseline,
                "reference_baseline": reference_baseline,
                "evaluated_gameweeks": int(len(diffs)),
                "mean_mae_difference": float(diffs.mean()),
                "ci_lower": float(series.quantile(0.025)),
                "ci_upper": float(series.quantile(0.975)),
                "bootstrap_samples": int(samples),
                "bootstrap_seed": int(seed),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "baseline",
                "reference_baseline",
                "evaluated_gameweeks",
                "mean_mae_difference",
                "ci_lower",
                "ci_upper",
                "bootstrap_samples",
                "bootstrap_seed",
            ]
        )
    return pd.DataFrame(rows).sort_values("baseline").reset_index(drop=True)
