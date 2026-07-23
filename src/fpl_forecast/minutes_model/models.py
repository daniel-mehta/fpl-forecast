from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl_forecast.minutes_model.config import MinutesModelConfig


MODEL_NAMES = (
    "M0_ZERO_MINUTES",
    "M1_RECENT_MEAN_P3",
    "M2_RECENT_MEAN_P5",
    "M3_EWMA_MINUTES",
    "M4_PREVIOUS_SEASON_ROLE_GW1",
    "M5_REGULARIZED_STATE_SOFTMAX",
    "M6_NONLINEAR_RECENCY_ENSEMBLE",
)

NUMERIC_FEATURES = (
    "was_home",
    "gameweek",
    "lag1_minutes_mean",
    "lag3_minutes_mean",
    "lag5_minutes_mean",
    "lag10_minutes_mean",
    "lag3_appearance_rate",
    "lag5_appearance_rate",
    "lag3_start_rate",
    "lag5_start_rate",
    "season_minutes_before",
    "season_appearance_before",
    "season_starts_before",
    "prior_season_minutes",
    "prior_season_appearances",
    "days_since_last_source",
)
CATEGORICAL_FEATURES = ("fpl_position",)


@dataclass(frozen=True)
class ModelDiagnostic:
    model_name: str
    status: str
    training_rows: int
    details: dict[str, object]


def fit_predict_minutes_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: MinutesModelConfig,
    models: list[str] | None = None,
) -> tuple[pd.DataFrame, list[ModelDiagnostic]]:
    if train.empty:
        raise ValueError("Cannot fit minutes models with no eligible training rows.")
    selected = list(MODEL_NAMES if models is None else models)
    unknown = sorted(set(selected).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown minutes model(s): {', '.join(unknown)}")
    predictions = []
    diagnostics = []
    for model_name in selected:
        pred, diagnostic = _predict_model(model_name, train, test, config)
        predictions.append(pred)
        diagnostics.append(diagnostic)
    return pd.concat(predictions, ignore_index=True), diagnostics


def _predict_model(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: MinutesModelConfig,
) -> tuple[pd.DataFrame, ModelDiagnostic]:
    if model_name == "M0_ZERO_MINUTES":
        probs = np.zeros((len(test), len(config.state_order)))
        probs[:, config.state_order.index("DNP")] = 1.0
        return _prediction_frame(test, model_name, probs, config), ModelDiagnostic(
            model_name, "deterministic_zero", len(train), {}
        )
    if model_name in {"M1_RECENT_MEAN_P3", "M2_RECENT_MEAN_P5", "M3_EWMA_MINUTES"}:
        minutes = _baseline_minutes(model_name, test, config)
        probs = _minutes_to_state_probs(minutes, config)
        return _prediction_frame(test, model_name, probs, config), ModelDiagnostic(
            model_name, "history_mean_baseline", len(train), {"formula": _baseline_formula(model_name)}
        )
    if model_name == "M4_PREVIOUS_SEASON_ROLE_GW1":
        minutes = _previous_season_minutes(test, config)
        probs = _minutes_to_state_probs(minutes, config)
        return _prediction_frame(test, model_name, probs, config), ModelDiagnostic(
            model_name,
            "previous_season_role_gw1",
            len(train),
            {"formula": "shrunk prior-season minutes per appearance times prior appearance probability"},
        )
    if model_name == "M5_REGULARIZED_STATE_SOFTMAX":
        probs, details = _softmax_state_model(train, test, config)
        return _prediction_frame(test, model_name, probs, config), ModelDiagnostic(
            model_name, "regularized_multinomial_softmax", len(train), details
        )
    probs, details = _nonlinear_recency_ensemble(train, test, config)
    return _prediction_frame(test, model_name, probs, config), ModelDiagnostic(
        model_name, "nonlinear_shrunk_bucket_ensemble", len(train), details
    )


def _prediction_frame(
    test: pd.DataFrame,
    model_name: str,
    state_probs: np.ndarray,
    config: MinutesModelConfig,
) -> pd.DataFrame:
    state_probs = _normalize_probs(state_probs)
    output = test[_identity_columns(test)].copy()
    output["model_name"] = model_name
    for index, state in enumerate(config.state_order):
        output[f"prob_state_{state.lower()}"] = state_probs[:, index]
    minutes = np.array([config.state_minutes[state] for state in config.state_order])
    output["predicted_minutes"] = state_probs @ minutes
    output["p_appearance"] = 1.0 - output["prob_state_dnp"]
    output["p_start"] = (
        output["prob_state_start_under_60"]
        + output["prob_state_start_60_to_89"]
        + output["prob_state_start_90"]
    )
    output["p_reached_60"] = (
        output["prob_state_sub_60_plus"]
        + output["prob_state_start_60_to_89"]
        + output["prob_state_start_90"]
    )
    output["p_played_90"] = output["prob_state_start_90"]
    output["prediction_source"] = "phase5_minutes_model"
    return output


def _identity_columns(test: pd.DataFrame) -> list[str]:
    columns = [
        "season",
        "gameweek",
        "player_uid",
        "stable_fixture_uid",
        "fixture_key",
        "player_name",
        "player_team_uid",
        "opponent_team_uid",
        "fpl_position",
        "was_home",
        "kickoff_time",
        "information_cutoff",
        "max_feature_source_available_time",
        "evaluation_population",
        "primary_data_quality_population",
        "pre_deadline_history_active",
        "cold_start_no_history",
        "transferred_player",
        "position_change",
        "starts_exact_available",
    ]
    return [column for column in columns if column in test.columns]


def _baseline_minutes(model_name: str, test: pd.DataFrame, config: MinutesModelConfig) -> np.ndarray:
    if model_name == "M1_RECENT_MEAN_P3":
        return pd.to_numeric(test["lag3_minutes_mean"], errors="coerce").fillna(0).clip(0, 90).to_numpy()
    if model_name == "M2_RECENT_MEAN_P5":
        return pd.to_numeric(test["lag5_minutes_mean"], errors="coerce").fillna(0).clip(0, 90).to_numpy()
    prev = pd.to_numeric(test["lag1_minutes_mean"], errors="coerce").fillna(0)
    recent = pd.to_numeric(test["lag3_minutes_mean"], errors="coerce").fillna(prev)
    longer = pd.to_numeric(test["lag10_minutes_mean"], errors="coerce").fillna(recent)
    alpha = config.ewma_alpha
    return (alpha * prev + (1 - alpha) * (0.65 * recent + 0.35 * longer)).clip(0, 90).to_numpy()


def _baseline_formula(model_name: str) -> str:
    formulas = {
        "M1_RECENT_MEAN_P3": "mean actual minutes from the previous three available fixtures",
        "M2_RECENT_MEAN_P5": "mean actual minutes from the previous five available fixtures",
        "M3_EWMA_MINUTES": "alpha * previous fixture minutes plus recency-weighted recent means",
    }
    return formulas[model_name]


def _previous_season_minutes(test: pd.DataFrame, config: MinutesModelConfig) -> np.ndarray:
    prior_minutes = pd.to_numeric(test.get("prior_season_minutes", 0), errors="coerce").fillna(0)
    prior_apps = pd.to_numeric(test.get("prior_season_appearances", 0), errors="coerce").fillna(0)
    role_rate = prior_minutes / prior_apps.replace(0, np.nan)
    role_rate = role_rate.fillna(0).clip(0, 90)
    app_prob = prior_apps / (prior_apps + config.shrink_matches)
    return (role_rate * app_prob).clip(0, 90).to_numpy()


def _minutes_to_state_probs(minutes: np.ndarray | pd.Series, config: MinutesModelConfig) -> np.ndarray:
    values = np.asarray(minutes, dtype=float)
    p90 = np.clip(values / max(config.state_minutes["START_90"], 1e-9), 0, 1)
    probs = np.zeros((len(values), len(config.state_order)))
    probs[:, config.state_order.index("DNP")] = 1 - p90
    probs[:, config.state_order.index("START_90")] = p90
    return probs


def _softmax_state_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: MinutesModelConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    train = _recent_training_sample(train, config)
    x_train, x_test = _design_matrices(train, test)
    y = pd.Categorical(train["actual_state"], categories=config.state_order).codes
    if (y < 0).any():
        raise ValueError("Training rows contain unknown minutes state labels.")
    y_onehot = np.eye(len(config.state_order))[y]
    weights = np.zeros((x_train.shape[1], len(config.state_order)))
    class_counts = y_onehot.sum(axis=0) + 1.0
    weights[0, :] = np.log(class_counts / class_counts.sum())
    lr = config.softmax_learning_rate
    l2 = config.softmax_l2
    for _ in range(config.softmax_iterations):
        probs = _softmax(x_train @ weights)
        grad = x_train.T @ (probs - y_onehot) / len(x_train)
        grad[1:, :] += l2 * weights[1:, :]
        weights -= lr * grad
    train_probs = _softmax(x_train @ weights)
    loss = float(-(y_onehot * np.log(np.clip(train_probs, 1e-12, 1))).sum(axis=1).mean())
    return _softmax(x_test @ weights), {
        "training_log_loss": loss,
        "features": x_train.shape[1],
        "sampled_training_rows": len(train),
    }


def _nonlinear_recency_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: MinutesModelConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    train = _recent_training_sample(train, config)
    global_probs = _state_distribution(train, config)
    lookups = []
    for columns in (
        ["fpl_position", "recency_bucket"],
        ["fpl_position", "start_bucket"],
        ["history_bucket"],
    ):
        lookups.append(_bucket_lookup(train, columns, global_probs, config))
    test_buckets = _with_buckets(test)
    probs = []
    for _, row in test_buckets.iterrows():
        row_probs = [global_probs]
        for columns, table in lookups:
            key = tuple(row[column] for column in columns)
            if key in table:
                row_probs.append(table[key])
        probs.append(np.mean(np.vstack(row_probs), axis=0))
    return _normalize_probs(np.vstack(probs)), {"ensembles": len(lookups), "min_leaf": config.ensemble_min_leaf}


def _recent_training_sample(train: pd.DataFrame, config: MinutesModelConfig) -> pd.DataFrame:
    if len(train) <= config.max_model_training_rows:
        return train
    return train.sort_values("source_available_time").tail(config.max_model_training_rows).copy()


def _design_matrices(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_num = _numeric_matrix(train)
    test_num = _numeric_matrix(test)
    means = np.nanmean(train_num, axis=0)
    stds = np.nanstd(train_num, axis=0)
    stds[stds < 1e-6] = 1.0
    train_num = (np.nan_to_num(train_num, nan=means) - means) / stds
    test_num = (np.nan_to_num(test_num, nan=means) - means) / stds
    train_cat, test_cat = _categorical_matrix(train, test)
    return (
        np.hstack([np.ones((len(train), 1)), train_num, train_cat]),
        np.hstack([np.ones((len(test), 1)), test_num, test_cat]),
    )


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for column in NUMERIC_FEATURES:
        if column in frame.columns:
            values.append(pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype=float))
        else:
            values.append(np.zeros(len(frame)))
    return np.vstack(values).T if values else np.empty((len(frame), 0))


def _categorical_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    test_parts = []
    for column in CATEGORICAL_FEATURES:
        categories = sorted(set(train.get(column, pd.Series(dtype=str)).dropna().astype(str)))
        for category in categories:
            train_parts.append(train[column].astype(str).eq(category).to_numpy(dtype=float))
            test_parts.append(test[column].astype(str).eq(category).to_numpy(dtype=float))
    if not train_parts:
        return np.empty((len(train), 0)), np.empty((len(test), 0))
    return np.vstack(train_parts).T, np.vstack(test_parts).T


def _with_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["recency_bucket"] = pd.cut(
        pd.to_numeric(output["lag5_minutes_mean"], errors="coerce").fillna(0),
        bins=[-0.1, 0, 30, 60, 120],
        labels=["none", "low", "medium", "high"],
    ).astype(str)
    output["start_bucket"] = pd.cut(
        pd.to_numeric(output["lag5_start_rate"], errors="coerce").fillna(0),
        bins=[-0.1, 0, 0.34, 0.67, 1.0],
        labels=["none", "low", "medium", "high"],
    ).astype(str)
    history = pd.to_numeric(output.get("prior_season_appearances", 0), errors="coerce").fillna(0)
    output["history_bucket"] = pd.cut(
        history,
        bins=[-0.1, 0, 5, 20, 100],
        labels=["none", "thin", "regular", "heavy"],
    ).astype(str)
    return output


def _bucket_lookup(
    train: pd.DataFrame,
    columns: list[str],
    global_probs: np.ndarray,
    config: MinutesModelConfig,
) -> tuple[list[str], dict[tuple[object, ...], np.ndarray]]:
    bucketed = _with_buckets(train)
    table = {}
    for key, group in bucketed.groupby(columns, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        weight = len(group) / (len(group) + config.shrink_matches)
        if len(group) >= config.ensemble_min_leaf:
            table[key_tuple] = weight * _state_distribution(group, config) + (1 - weight) * global_probs
    return columns, table


def _state_distribution(frame: pd.DataFrame, config: MinutesModelConfig) -> np.ndarray:
    counts = (
        frame["actual_state"].value_counts().reindex(config.state_order).fillna(0).to_numpy(dtype=float).copy()
    )
    counts += 1.0
    return counts / counts.sum()


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 0, 1)
    denom = probs.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return probs / denom
