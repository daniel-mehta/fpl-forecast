from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl_forecast.team_model.config import TeamModelConfig
from fpl_forecast.team_model.probabilities import (
    dixon_coles_joint_distribution,
    dixon_coles_score_probability,
)


@dataclass(frozen=True)
class TeamMetricTables:
    expected_goals: pd.DataFrame
    expected_goals_by_season: pd.DataFrame
    expected_goals_by_gameweek: pd.DataFrame
    expected_goals_by_side: pd.DataFrame
    expected_goals_by_fallback: pd.DataFrame
    clean_sheet: pd.DataFrame
    clean_sheet_calibration: pd.DataFrame
    outcome: pd.DataFrame
    joint_score: pd.DataFrame
    joint_score_by_season: pd.DataFrame
    low_score: pd.DataFrame
    low_score_by_season: pd.DataFrame
    bootstrap: pd.DataFrame
    bootstrap_t3_vs_t2: pd.DataFrame


def score_fixture_predictions(predictions: pd.DataFrame, outcomes: pd.DataFrame, config: TeamModelConfig) -> pd.DataFrame:
    scored = predictions.merge(
        outcomes[
            [
                "season",
                "stable_fixture_uid",
                "home_goals",
                "away_goals",
                "result_valid",
            ]
        ],
        on=["season", "stable_fixture_uid"],
        how="left",
        validate="many_to_one",
    )
    if scored["home_goals"].isna().any() or scored["away_goals"].isna().any():
        raise ValueError("Scored team predictions have missing joined outcomes.")
    scored["home_goal_error"] = scored["expected_home_goals"] - scored["home_goals"]
    scored["away_goal_error"] = scored["expected_away_goals"] - scored["away_goals"]
    scored["home_poisson_nll"] = _poisson_nll(scored["home_goals"], scored["expected_home_goals"])
    scored["away_poisson_nll"] = _poisson_nll(scored["away_goals"], scored["expected_away_goals"])
    scored["home_clean_sheet_actual"] = (scored["away_goals"] == 0).astype(int)
    scored["away_clean_sheet_actual"] = (scored["home_goals"] == 0).astype(int)
    scored["match_outcome_actual"] = np.select(
        [scored["home_goals"] > scored["away_goals"], scored["home_goals"] == scored["away_goals"]],
        ["H", "D"],
        default="A",
    )
    scored["match_outcome_prediction"] = scored[
        ["home_win_probability", "draw_probability", "away_win_probability"]
    ].idxmax(axis=1).map(
        {
            "home_win_probability": "H",
            "draw_probability": "D",
            "away_win_probability": "A",
        }
    )
    scored["draw_actual"] = scored["match_outcome_actual"].eq("D").astype(int)
    scored["draw_error"] = scored["draw_probability"] - scored["draw_actual"]
    scored["scoreline_actual"] = scored["home_goals"].astype(str) + "-" + scored["away_goals"].astype(str)
    scored["joint_score_probability_actual"] = scored.apply(_actual_score_probability, axis=1).clip(
        config.log_loss_epsilon,
        1.0,
    )
    scored["joint_score_nll"] = -np.log(scored["joint_score_probability_actual"])
    predicted_score = scored.apply(lambda row: _most_likely_scoreline(row, config), axis=1)
    scored["scoreline_prediction"] = predicted_score
    scored["exact_score_hit"] = scored["scoreline_prediction"].eq(scored["scoreline_actual"]).astype(int)
    return scored


def team_metric_tables(scored: pd.DataFrame, config: TeamModelConfig) -> TeamMetricTables:
    return TeamMetricTables(
        expected_goals=_goal_metrics(scored, ["model_name"]),
        expected_goals_by_season=_goal_metrics(scored, ["season", "model_name"]),
        expected_goals_by_gameweek=_goal_metrics(scored, ["season", "gameweek", "model_name"]),
        expected_goals_by_side=_goal_side_metrics(scored),
        expected_goals_by_fallback=_goal_metrics(
            scored.assign(
                fallback_involved=scored["home_unseen_or_promoted_flag"]
                | scored["away_unseen_or_promoted_flag"]
                | scored["home_low_history_flag"]
                | scored["away_low_history_flag"]
            ),
            ["fallback_involved", "model_name"],
        ),
        clean_sheet=_clean_sheet_metrics(scored, ["model_name"], config),
        clean_sheet_calibration=_clean_sheet_calibration(scored),
        outcome=_outcome_metrics(scored, ["model_name"], config),
        joint_score=_joint_score_metrics(scored, ["model_name"]),
        joint_score_by_season=_joint_score_metrics(scored, ["season", "model_name"]),
        low_score=_low_score_metrics(scored, ["model_name"]),
        low_score_by_season=_low_score_metrics(scored, ["season", "model_name"]),
        bootstrap=_block_bootstrap(scored, reference_model=config.reference_model, config=config),
        bootstrap_t3_vs_t2=_block_bootstrap_t3_vs_t2(scored, config=config),
    )


def _goal_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        errors = pd.concat([group["home_goal_error"], group["away_goal_error"]], ignore_index=True)
        nll = pd.concat([group["home_poisson_nll"], group["away_poisson_nll"]], ignore_index=True)
        rows.append(
            {
                **dict(zip(groups, key_values, strict=False)),
                "fixtures": int(len(group)),
                "goal_mae": float(errors.abs().mean()),
                "goal_rmse": float(math.sqrt(float((errors**2).mean()))),
                "goal_bias": float(errors.mean()),
                "poisson_nll": float(nll.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _goal_side_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in frame.groupby("model_name"):
        for side, error_column, nll_column in (
            ("home", "home_goal_error", "home_poisson_nll"),
            ("away", "away_goal_error", "away_poisson_nll"),
        ):
            errors = group[error_column]
            rows.append(
                {
                    "model_name": model_name,
                    "side": side,
                    "fixtures": int(len(group)),
                    "goal_mae": float(errors.abs().mean()),
                    "goal_rmse": float(math.sqrt(float((errors**2).mean()))),
                    "goal_bias": float(errors.mean()),
                    "poisson_nll": float(group[nll_column].mean()),
                }
            )
    return pd.DataFrame(rows)


def _clean_sheet_metrics(frame: pd.DataFrame, groups: list[str], config: TeamModelConfig) -> pd.DataFrame:
    rows = []
    eps = config.log_loss_epsilon
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        actual = pd.concat([group["home_clean_sheet_actual"], group["away_clean_sheet_actual"]], ignore_index=True)
        predicted = pd.concat(
            [group["home_clean_sheet_probability"], group["away_clean_sheet_probability"]],
            ignore_index=True,
        ).clip(eps, 1 - eps)
        rows.append(
            {
                **dict(zip(groups, key_values, strict=False)),
                "team_fixture_rows": int(len(actual)),
                "brier": float(((predicted - actual) ** 2).mean()),
                "log_loss": float((-(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))).mean()),
                "predicted_rate": float(predicted.mean()),
                "observed_rate": float(actual.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _clean_sheet_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in frame.groupby("model_name"):
        stacked = pd.DataFrame(
            {
                "probability": pd.concat(
                    [group["home_clean_sheet_probability"], group["away_clean_sheet_probability"]],
                    ignore_index=True,
                ),
                "actual": pd.concat(
                    [group["home_clean_sheet_actual"], group["away_clean_sheet_actual"]],
                    ignore_index=True,
                ),
            }
        )
        bin_edges = np.linspace(0.0, 1.0, 11)
        stacked["probability_bin"] = pd.cut(
            stacked["probability"],
            bins=bin_edges,
            include_lowest=True,
            right=True,
            labels=False,
        )
        summary = (
            stacked.groupby("probability_bin", as_index=False)
            .agg(
                rows=("actual", "size"),
                predicted_rate=("probability", "mean"),
                observed_rate=("actual", "mean"),
            )
            .assign(model_name=model_name)
        )
        summary["bin_lower"] = summary["probability_bin"].map(lambda value: float(bin_edges[int(value)]))
        summary["bin_upper"] = summary["probability_bin"].map(lambda value: float(bin_edges[int(value) + 1]))
        summary["calibration_gap"] = summary["observed_rate"] - summary["predicted_rate"]
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)[
        [
            "model_name",
            "probability_bin",
            "bin_lower",
            "bin_upper",
            "rows",
            "predicted_rate",
            "observed_rate",
            "calibration_gap",
        ]
    ]


def _outcome_metrics(frame: pd.DataFrame, groups: list[str], config: TeamModelConfig) -> pd.DataFrame:
    rows = []
    eps = config.log_loss_epsilon
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        probs = group[["home_win_probability", "draw_probability", "away_win_probability"]].clip(eps, 1 - eps)
        actual = pd.get_dummies(group["match_outcome_actual"]).reindex(columns=["H", "D", "A"], fill_value=0)
        selected = np.select(
            [group["match_outcome_actual"].eq("H"), group["match_outcome_actual"].eq("D")],
            [probs["home_win_probability"], probs["draw_probability"]],
            default=probs["away_win_probability"],
        )
        rows.append(
            {
                **dict(zip(groups, key_values, strict=False)),
                "fixtures": int(len(group)),
                "multiclass_log_loss": float(-np.log(selected).mean()),
                "multiclass_brier": float(((probs.to_numpy() - actual.to_numpy()) ** 2).sum(axis=1).mean()),
                "accuracy": float(group["match_outcome_prediction"].eq(group["match_outcome_actual"]).mean()),
                "draw_brier": float(((group["draw_probability"] - group["draw_actual"]) ** 2).mean()),
                "draw_predicted_rate": float(group["draw_probability"].mean()),
                "draw_observed_rate": float(group["draw_actual"].mean()),
                "exact_score_accuracy": float(group["exact_score_hit"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _joint_score_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(groups, key_values, strict=False)),
                "fixtures": int(len(group)),
                "joint_score_nll": float(group["joint_score_nll"].mean()),
                "exact_score_accuracy": float(group["exact_score_hit"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _low_score_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    low_scores = {
        "0-0": "joint_prob_0_0",
        "0-1": "joint_prob_0_1",
        "1-0": "joint_prob_1_0",
        "1-1": "joint_prob_1_1",
    }
    for key, group in frame.groupby(groups, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(groups, key_values, strict=False))
        for scoreline, probability_column in low_scores.items():
            predicted = group[probability_column]
            observed = group["scoreline_actual"].eq(scoreline).astype(int)
            rows.append(
                {
                    **base,
                    "scoreline": scoreline,
                    "fixtures": int(len(group)),
                    "predicted_frequency": float(predicted.mean()),
                    "observed_frequency": float(observed.mean()),
                    "calibration_gap": float(observed.mean() - predicted.mean()),
                }
            )
        predicted = group[list(low_scores.values())].sum(axis=1)
        observed = group["scoreline_actual"].isin(low_scores).astype(int)
        rows.append(
            {
                **base,
                "scoreline": "low_score_corner",
                "fixtures": int(len(group)),
                "predicted_frequency": float(predicted.mean()),
                "observed_frequency": float(observed.mean()),
                "calibration_gap": float(observed.mean() - predicted.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(groups + ["scoreline"]).reset_index(drop=True)


def _block_bootstrap(scored: pd.DataFrame, *, reference_model: str, config: TeamModelConfig) -> pd.DataFrame:
    pivot = (
        _goal_metrics(scored, ["season", "gameweek", "model_name"])
        .pivot(index=["season", "gameweek"], columns="model_name", values="goal_mae")
        .dropna()
    )
    if pivot.empty or reference_model not in pivot.columns:
        return pd.DataFrame()
    sampler = pd.Series(range(len(pivot))).sample
    rows = []
    for model_name in pivot.columns:
        if model_name == reference_model:
            continue
        diffs = pivot[model_name] - pivot[reference_model]
        sampled = [
            float(diffs.iloc[sampler(n=len(diffs), replace=True, random_state=config.bootstrap_seed + i).to_numpy()].mean())
            for i in range(config.bootstrap_samples)
        ]
        series = pd.Series(sampled)
        rows.append(
            {
                "model_name": model_name,
                "reference_model": reference_model,
                "evaluated_gameweeks": int(len(diffs)),
                "mean_goal_mae_difference": float(diffs.mean()),
                "ci_lower": float(series.quantile(0.025)),
                "ci_upper": float(series.quantile(0.975)),
                "bootstrap_samples": config.bootstrap_samples,
                "bootstrap_seed": config.bootstrap_seed,
            }
        )
    return pd.DataFrame(rows)


def _block_bootstrap_t3_vs_t2(scored: pd.DataFrame, *, config: TeamModelConfig) -> pd.DataFrame:
    metrics = _block_metric_values(scored)
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    sampler = pd.Series(range(len(metrics))).sample
    for metric in ("joint_score_nll", "multiclass_log_loss", "draw_brier"):
        t3 = metrics[f"T3_DIXON_COLES:{metric}"]
        t2 = metrics[f"T2_REGULARIZED_ATTACK_DEFENCE:{metric}"]
        diffs = t3 - t2
        sampled = [
            float(diffs.iloc[sampler(n=len(diffs), replace=True, random_state=config.bootstrap_seed + i).to_numpy()].mean())
            for i in range(config.bootstrap_samples)
        ]
        series = pd.Series(sampled)
        rows.append(
            {
                "comparison": "T3_DIXON_COLES_minus_T2_REGULARIZED_ATTACK_DEFENCE",
                "metric": metric,
                "evaluated_gameweeks": int(len(diffs)),
                "mean_difference": float(diffs.mean()),
                "ci_lower": float(series.quantile(0.025)),
                "ci_upper": float(series.quantile(0.975)),
                "bootstrap_samples": config.bootstrap_samples,
                "bootstrap_seed": config.bootstrap_seed,
            }
        )
    return pd.DataFrame(rows)


def _block_metric_values(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, gameweek, model_name), group in scored.groupby(["season", "gameweek", "model_name"]):
        probs = group[["home_win_probability", "draw_probability", "away_win_probability"]].clip(1e-12, 1 - 1e-12)
        selected = np.select(
            [group["match_outcome_actual"].eq("H"), group["match_outcome_actual"].eq("D")],
            [probs["home_win_probability"], probs["draw_probability"]],
            default=probs["away_win_probability"],
        )
        rows.append(
            {
                "season": season,
                "gameweek": gameweek,
                "model_name": model_name,
                "joint_score_nll": float(group["joint_score_nll"].mean()),
                "multiclass_log_loss": float(-np.log(selected).mean()),
                "draw_brier": float(((group["draw_probability"] - group["draw_actual"]) ** 2).mean()),
            }
        )
    pivot = pd.DataFrame(rows).pivot(index=["season", "gameweek"], columns="model_name")
    required = [
        ("joint_score_nll", "T2_REGULARIZED_ATTACK_DEFENCE"),
        ("joint_score_nll", "T3_DIXON_COLES"),
        ("multiclass_log_loss", "T2_REGULARIZED_ATTACK_DEFENCE"),
        ("multiclass_log_loss", "T3_DIXON_COLES"),
        ("draw_brier", "T2_REGULARIZED_ATTACK_DEFENCE"),
        ("draw_brier", "T3_DIXON_COLES"),
    ]
    if pivot.empty or any(column not in pivot.columns for column in required):
        return pd.DataFrame()
    flattened = pivot[required].dropna()
    flattened.columns = [f"{model_name}:{metric}" for metric, model_name in flattened.columns]
    return flattened


def _poisson_nll(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    y = pd.to_numeric(actual, errors="coerce").astype(float)
    lam = pd.to_numeric(predicted, errors="coerce").astype(float).clip(lower=1e-12)
    return lam - y * np.log(lam) + y.map(lambda value: math.lgamma(value + 1))


def _actual_score_probability(row: pd.Series) -> float:
    return dixon_coles_score_probability(
        int(row["home_goals"]),
        int(row["away_goals"]),
        float(row["expected_home_goals"]),
        float(row["expected_away_goals"]),
        float(row.get("dixon_coles_rho", 0.0) or 0.0),
    )


def _most_likely_scoreline(row: pd.Series, config: TeamModelConfig) -> str:
    joint = dixon_coles_joint_distribution(
        float(row["expected_home_goals"]),
        float(row["expected_away_goals"]),
        float(row.get("dixon_coles_rho", 0.0) or 0.0),
        max_goals=config.probability_max_goals,
    )
    finite = joint[:-1, :-1]
    home_goals, away_goals = np.unravel_index(np.argmax(finite), finite.shape)
    return f"{int(home_goals)}-{int(away_goals)}"
