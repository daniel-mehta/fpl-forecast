from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl_forecast.team_model.config import TeamModelConfig


MODEL_NAMES = (
    "T0_LEAGUE_HOME_AWAY",
    "T1_SHRUNK_ROLLING_TEAM_RATE",
    "T2_REGULARIZED_ATTACK_DEFENCE",
)


@dataclass(frozen=True)
class ModelFit:
    model_name: str
    parameters: dict[str, object]
    ratings: pd.DataFrame
    converged: bool = True
    status: str = "ok"


def fit_predict_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    config: TeamModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    if train.empty:
        raise ValueError("Cannot fit team models with no eligible training fixtures.")
    fits = [
        fit_t0(train, config=config),
        fit_t1(train, cutoff=cutoff, config=config),
        fit_t2(train, cutoff=cutoff, config=config),
    ]
    predictions = []
    diagnostics = []
    ratings = []
    for fit in fits:
        predicted = predict_with_fit(fit, test, config=config)
        predictions.append(predicted)
        ratings.append(fit.ratings.assign(model_name=fit.model_name, cutoff=cutoff.isoformat()))
        diagnostics.append(
            {
                "model_name": fit.model_name,
                "converged": fit.converged,
                "status": fit.status,
                "training_fixtures": int(len(train)),
                "iterations": fit.parameters.get("iterations"),
                "objective": fit.parameters.get("objective"),
                "max_abs_gradient": fit.parameters.get("max_abs_gradient"),
                "home_lambda_clipped": int(predicted["home_lambda_clipped_flag"].sum()),
                "away_lambda_clipped": int(predicted["away_lambda_clipped_flag"].sum()),
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.concat(ratings, ignore_index=True), diagnostics


def fit_t0(train: pd.DataFrame, *, config: TeamModelConfig) -> ModelFit:
    valid = train.loc[train["result_valid"].astype(bool)].copy()
    if len(valid) < config.t0_min_training_fixtures:
        raise ValueError("T0 has insufficient training fixtures.")
    home_mean = _safe_mean(valid["home_goals"], fallback=1.3)
    away_mean = _safe_mean(valid["away_goals"], fallback=1.1)
    ratings = _neutral_ratings(valid, config=config)
    return ModelFit(
        "T0_LEAGUE_HOME_AWAY",
        {"home_mean": home_mean, "away_mean": away_mean},
        ratings,
    )


def fit_t1(train: pd.DataFrame, *, cutoff: pd.Timestamp, config: TeamModelConfig) -> ModelFit:
    valid = train.loc[train["result_valid"].astype(bool)].copy()
    home_mean = _safe_mean(valid["home_goals"], fallback=1.3)
    away_mean = _safe_mean(valid["away_goals"], fallback=1.1)
    params: dict[str, object] = {
        "home_mean": home_mean,
        "away_mean": away_mean,
        "team_rates": {},
    }
    rows = []
    all_teams = sorted(set(valid["home_team_uid"]) | set(valid["away_team_uid"]))
    for team in all_teams:
        home_recent = (
            valid.loc[valid["home_team_uid"].eq(team)]
            .sort_values("source_available_time")
            .tail(config.t1_recent_window)
        )
        away_recent = (
            valid.loc[valid["away_team_uid"].eq(team)]
            .sort_values("source_available_time")
            .tail(config.t1_recent_window)
        )
        home_attack = _shrunk_rate(home_recent["home_goals"], home_mean, config.t1_shrink_matches)
        away_attack = _shrunk_rate(away_recent["away_goals"], away_mean, config.t1_shrink_matches)
        home_defence = _shrunk_rate(home_recent["away_goals"], away_mean, config.t1_shrink_matches)
        away_defence = _shrunk_rate(away_recent["home_goals"], home_mean, config.t1_shrink_matches)
        count = int(len(home_recent) + len(away_recent))
        params["team_rates"][team] = {
            "home_attack": home_attack,
            "away_attack": away_attack,
            "home_defence_concede": home_defence,
            "away_defence_concede": away_defence,
            "training_fixture_count": count,
        }
        rows.append(
            {
                "team_uid": team,
                "attack_effect": math.log(max((home_attack + away_attack) / max(home_mean + away_mean, 1e-9), 1e-9)),
                "attack_multiplier": ((home_attack + away_attack) / 2) / max((home_mean + away_mean) / 2, 1e-9),
                "defence_effect": math.log(max((home_defence + away_defence) / max(home_mean + away_mean, 1e-9), 1e-9)),
                "defence_concession_multiplier": ((home_defence + away_defence) / 2)
                / max((home_mean + away_mean) / 2, 1e-9),
                "training_fixture_count": count,
                "fallback_flag": count < config.t2_low_history_threshold,
            }
        )
    return ModelFit("T1_SHRUNK_ROLLING_TEAM_RATE", params, pd.DataFrame(rows))


def fit_t2(train: pd.DataFrame, *, cutoff: pd.Timestamp, config: TeamModelConfig) -> ModelFit:
    valid = train.loc[train["result_valid"].astype(bool)].copy()
    weights = _recency_weights(valid, cutoff=cutoff, half_life_days=config.recency_half_life_days)
    base_home = _weighted_mean(valid["home_goals"], weights, fallback=1.3)
    base_away = _weighted_mean(valid["away_goals"], weights, fallback=1.1)
    all_teams = sorted(set(valid["home_team_uid"]) | set(valid["away_team_uid"]))
    fit = _fit_t2_poisson_parameters(valid, weights, all_teams, base_home, base_away, config)
    attack = fit["attack"]
    defence = fit["defence"]
    rows = []
    for team in all_teams:
        team_mask = valid["home_team_uid"].eq(team) | valid["away_team_uid"].eq(team)
        count = int(team_mask.sum())
        rows.append(
            {
                "team_uid": team,
                "attack_effect": attack[team],
                "attack_multiplier": math.exp(attack[team]),
                "defence_effect": defence[team],
                "defence_concession_multiplier": math.exp(defence[team]),
                "training_fixture_count": count,
                "positive_weight_fixture_count": int((team_mask & weights.gt(0).to_numpy()).sum()),
                "effective_sample_size": _effective_sample_size(weights.loc[team_mask]),
                "fallback_flag": count < config.t2_low_history_threshold,
            }
        )
    return ModelFit(
        "T2_REGULARIZED_ATTACK_DEFENCE",
        {
            "intercept": fit["intercept"],
            "home_advantage": fit["home_advantage"],
            "attack": attack,
            "defence": defence,
            "solver": "deterministic_newton_line_search",
            "recency_weight_source": "source_available_time",
            "loss": (
                "sum_i w_i * (exp(eta_i) - y_i * eta_i + log(y_i!)) "
                "+ 0.5 * t2_ridge_penalty * (sum_t attack_t^2 + sum_t defence_t^2)"
            ),
            "penalized_parameters": "attack,defence",
            "unpenalized_parameters": "intercept,home_advantage",
            "identifiability_constraint": "sum(attack)=0 and sum(defence)=0",
            "recency_half_life_days": config.recency_half_life_days,
            "ridge_penalty": config.t2_ridge_penalty,
            "iterations": fit["iterations"],
            "objective": fit["objective"],
            "max_abs_gradient": fit["max_abs_gradient"],
        },
        pd.DataFrame(rows),
        converged=True,
        status=f"converged iterations={fit['iterations']} max_abs_gradient={fit['max_abs_gradient']:.3g}",
    )


def predict_with_fit(fit: ModelFit, test: pd.DataFrame, *, config: TeamModelConfig) -> pd.DataFrame:
    rows = []
    for row in test.itertuples(index=False):
        home_team = row.home_team_uid
        away_team = row.away_team_uid
        if fit.model_name == "T0_LEAGUE_HOME_AWAY":
            home_lambda = float(fit.parameters["home_mean"])
            away_lambda = float(fit.parameters["away_mean"])
        elif fit.model_name == "T1_SHRUNK_ROLLING_TEAM_RATE":
            rates = fit.parameters["team_rates"]
            home_rates = rates.get(home_team, {})
            away_rates = rates.get(away_team, {})
            home_lambda = (
                home_rates.get("home_attack", fit.parameters["home_mean"])
                + away_rates.get("away_defence_concede", fit.parameters["home_mean"])
            ) / 2
            away_lambda = (
                away_rates.get("away_attack", fit.parameters["away_mean"])
                + home_rates.get("home_defence_concede", fit.parameters["away_mean"])
            ) / 2
        else:
            attack = fit.parameters["attack"]
            defence = fit.parameters["defence"]
            intercept = float(fit.parameters["intercept"])
            home_advantage = float(fit.parameters["home_advantage"])
            home_lambda = math.exp(
                intercept + home_advantage + attack.get(home_team, 0.0) + defence.get(away_team, 0.0)
            )
            away_lambda = math.exp(intercept + attack.get(away_team, 0.0) + defence.get(home_team, 0.0))
        home_count = _team_count(fit.ratings, home_team)
        away_count = _team_count(fit.ratings, away_team)
        rows.append(
            {
                **row._asdict(),
                "model_name": fit.model_name,
                "expected_home_goals": _bound_lambda(home_lambda, config),
                "expected_away_goals": _bound_lambda(away_lambda, config),
                "home_lambda_clipped_flag": not (
                    config.min_expected_goals <= home_lambda <= config.max_expected_goals
                ),
                "away_lambda_clipped_flag": not (
                    config.min_expected_goals <= away_lambda <= config.max_expected_goals
                ),
                "home_unseen_or_promoted_flag": home_count == 0,
                "away_unseen_or_promoted_flag": away_count == 0,
                "home_low_history_flag": 0 < home_count < config.t2_low_history_threshold,
                "away_low_history_flag": 0 < away_count < config.t2_low_history_threshold,
                "fit_status": fit.status,
                "fit_converged": fit.converged,
            }
        )
    return pd.DataFrame(rows)


def _safe_mean(series: pd.Series, *, fallback: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return fallback if values.empty else float(values.mean())


def _weighted_mean(series: pd.Series, weights: pd.Series, *, fallback: float) -> float:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    total = float(weights.sum())
    return fallback if total <= 0 else float((values * weights).sum() / total)


def _shrunk_rate(series: pd.Series, prior: float, shrink_matches: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values.sum() + shrink_matches * prior) / (len(values) + shrink_matches))


def _recency_weights(frame: pd.DataFrame, *, cutoff: pd.Timestamp, half_life_days: int) -> pd.Series:
    available = pd.to_datetime(frame["source_available_time"], utc=True)
    age_days = (cutoff - available).dt.total_seconds() / 86400
    age_days = age_days.clip(lower=0)
    return 0.5 ** (age_days / float(half_life_days))


def _center_effects(effects: dict[str, float]) -> dict[str, float]:
    if not effects:
        return effects
    mean = float(np.mean(list(effects.values())))
    return {team: value - mean for team, value in effects.items()}


def _fit_t2_poisson_parameters(
    valid: pd.DataFrame,
    weights: pd.Series,
    teams: list[str],
    base_home: float,
    base_away: float,
    config: TeamModelConfig,
) -> dict[str, object]:
    if len(teams) < 2:
        raise ValueError("T2 requires at least two teams.")
    team_index = {team: idx for idx, team in enumerate(teams)}
    free_team_count = len(teams) - 1
    param_count = 2 + 2 * free_team_count
    design = np.zeros((len(valid) * 2, param_count), dtype=float)
    goals = np.zeros(len(valid) * 2, dtype=float)
    obs_weights = np.zeros(len(valid) * 2, dtype=float)
    for fixture_idx, row in enumerate(valid.itertuples(index=False)):
        home_row = fixture_idx * 2
        away_row = home_row + 1
        _fill_t2_design_row(
            design[home_row],
            team_index[str(row.home_team_uid)],
            team_index[str(row.away_team_uid)],
            free_team_count,
            home=True,
        )
        _fill_t2_design_row(
            design[away_row],
            team_index[str(row.away_team_uid)],
            team_index[str(row.home_team_uid)],
            free_team_count,
            home=False,
        )
        goals[home_row] = float(row.home_goals)
        goals[away_row] = float(row.away_goals)
        weight = float(weights.iloc[fixture_idx])
        obs_weights[home_row] = weight
        obs_weights[away_row] = weight

    params = np.zeros(param_count, dtype=float)
    params[0] = math.log(math.sqrt(max(base_home * base_away, 1e-9)))
    params[1] = math.log(max(base_home, 1e-9)) - params[0]
    ridge = float(config.t2_ridge_penalty)
    objective, gradient, hessian = _t2_objective_gradient_hessian(
        design,
        goals,
        obs_weights,
        params,
        ridge,
        free_team_count,
    )
    max_abs_gradient = float(np.abs(gradient).max())
    iterations = 0
    converged = max_abs_gradient <= config.t2_convergence_tolerance
    for iteration in range(1, config.t2_max_iterations + 1):
        if converged:
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        accepted = False
        for power in range(12):
            step_size = 0.5**power
            candidate = params - step_size * step
            candidate_objective, candidate_gradient, candidate_hessian = _t2_objective_gradient_hessian(
                design,
                goals,
                obs_weights,
                candidate,
                ridge,
                free_team_count,
            )
            if math.isfinite(candidate_objective) and candidate_objective < objective:
                params = candidate
                objective = candidate_objective
                gradient = candidate_gradient
                hessian = candidate_hessian
                max_abs_gradient = float(np.abs(gradient).max())
                iterations = iteration
                accepted = True
                converged = max_abs_gradient <= config.t2_convergence_tolerance
                break
        if not accepted:
            raise RuntimeError("T2 penalized Poisson fit failed: line search could not reduce objective.")
    if not converged:
        raise RuntimeError(
            "T2 penalized Poisson fit failed to converge "
            f"after {config.t2_max_iterations} iterations; max_abs_gradient={max_abs_gradient:.6g}."
        )
    return {
        "intercept": float(params[0]),
        "home_advantage": float(params[1]),
        "attack": _constrained_effects(params[2 : 2 + free_team_count], teams),
        "defence": _constrained_effects(params[2 + free_team_count :], teams),
        "iterations": iterations,
        "objective": float(objective),
        "max_abs_gradient": max_abs_gradient,
    }


def _fill_t2_design_row(
    row: np.ndarray,
    attacking_team_index: int,
    defending_team_index: int,
    free_team_count: int,
    *,
    home: bool,
) -> None:
    row[0] = 1.0
    row[1] = 1.0 if home else 0.0
    _set_constrained_effect(row, 2, attacking_team_index, free_team_count)
    _set_constrained_effect(row, 2 + free_team_count, defending_team_index, free_team_count)


def _set_constrained_effect(
    row: np.ndarray,
    offset: int,
    team_index: int,
    free_team_count: int,
) -> None:
    if team_index < free_team_count:
        row[offset + team_index] = 1.0
    else:
        row[offset : offset + free_team_count] = -1.0


def _t2_objective_gradient_hessian(
    design: np.ndarray,
    goals: np.ndarray,
    weights: np.ndarray,
    params: np.ndarray,
    ridge: float,
    free_team_count: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    eta = np.clip(design @ params, -20, 20)
    expected = np.exp(eta)
    constant = np.array([math.lgamma(value + 1) for value in goals])
    objective = float(np.sum(weights * (expected - goals * eta + constant)))
    gradient = design.T @ (weights * (expected - goals))
    hessian = design.T @ (design * (weights * expected)[:, None])
    penalty, penalty_gradient, penalty_hessian = _t2_penalty(params, ridge, free_team_count)
    return objective + penalty, gradient + penalty_gradient, hessian + penalty_hessian


def _t2_penalty(
    params: np.ndarray,
    ridge: float,
    free_team_count: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    penalty_gradient = np.zeros_like(params)
    penalty_hessian = np.zeros((len(params), len(params)), dtype=float)
    if free_team_count <= 0 or ridge <= 0:
        return 0.0, penalty_gradient, penalty_hessian
    total_penalty = 0.0
    for offset in (2, 2 + free_team_count):
        values = params[offset : offset + free_team_count]
        total = float(values.sum())
        total_penalty += 0.5 * ridge * float((values**2).sum() + total**2)
        penalty_gradient[offset : offset + free_team_count] = ridge * (values + total)
        penalty_hessian[offset : offset + free_team_count, offset : offset + free_team_count] = ridge * (
            np.eye(free_team_count) + np.ones((free_team_count, free_team_count))
        )
    return total_penalty, penalty_gradient, penalty_hessian


def _constrained_effects(values: np.ndarray, teams: list[str]) -> dict[str, float]:
    effects = [float(value) for value in values]
    effects.append(-float(np.sum(values)))
    return dict(zip(teams, effects, strict=True))


def _effective_sample_size(weights: pd.Series) -> float:
    total = float(weights.sum())
    squared = float((weights**2).sum())
    if squared <= 0:
        return 0.0
    return float((total**2) / squared)


def _bound_lambda(value: float, config: TeamModelConfig) -> float:
    if not math.isfinite(value):
        return config.min_expected_goals
    return float(min(max(value, config.min_expected_goals), config.max_expected_goals))


def _neutral_ratings(train: pd.DataFrame, *, config: TeamModelConfig) -> pd.DataFrame:
    teams = sorted(set(train["home_team_uid"]) | set(train["away_team_uid"]))
    return pd.DataFrame(
        [
            {
                "team_uid": team,
                "attack_effect": 0.0,
                "attack_multiplier": 1.0,
                "defence_effect": 0.0,
                "defence_concession_multiplier": 1.0,
                "training_fixture_count": int(
                    train["home_team_uid"].eq(team).sum() + train["away_team_uid"].eq(team).sum()
                ),
                "fallback_flag": False,
            }
            for team in teams
        ]
    )


def _team_count(ratings: pd.DataFrame, team_uid: str) -> int:
    matched = ratings.loc[ratings["team_uid"].eq(team_uid), "training_fixture_count"]
    if matched.empty:
        return 0
    return int(matched.iloc[0])
