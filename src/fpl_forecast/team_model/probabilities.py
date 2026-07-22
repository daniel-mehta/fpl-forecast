from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fpl_forecast.team_model.config import TeamModelConfig


def poisson_goal_distribution(lam: float, *, max_goals: int) -> list[float]:
    lam = max(float(lam), 0.0)
    probs = [math.exp(-lam)]
    for goals in range(1, max_goals + 1):
        probs.append(probs[-1] * lam / goals)
    tail = max(0.0, 1.0 - sum(probs))
    return [*probs, tail]


def add_probability_columns(predictions: pd.DataFrame, config: TeamModelConfig) -> pd.DataFrame:
    rows = []
    for row in predictions.itertuples(index=False):
        record = row._asdict()
        home_lambda = record["expected_home_goals"]
        away_lambda = record["expected_away_goals"]
        rho = float(record.get("dixon_coles_rho", 0.0) or 0.0)
        home_dist = poisson_goal_distribution(
            home_lambda,
            max_goals=config.probability_max_goals,
        )
        away_dist = poisson_goal_distribution(
            away_lambda,
            max_goals=config.probability_max_goals,
        )
        for goals, probability in enumerate(home_dist[:-1]):
            record[f"home_goals_prob_{goals}"] = probability
        record[f"home_goals_prob_{config.probability_max_goals + 1}_plus"] = home_dist[-1]
        for goals, probability in enumerate(away_dist[:-1]):
            record[f"away_goals_prob_{goals}"] = probability
        record[f"away_goals_prob_{config.probability_max_goals + 1}_plus"] = away_dist[-1]
        record["home_clean_sheet_probability"] = away_dist[0]
        record["away_clean_sheet_probability"] = home_dist[0]
        if record["model_name"] == "T3_DIXON_COLES":
            joint = dixon_coles_joint_distribution(
                home_lambda,
                away_lambda,
                rho,
                max_goals=config.probability_max_goals,
            )
            record["joint_probability_model"] = "dixon_coles"
            record["joint_prob_0_0"] = joint[0, 0]
            record["joint_prob_0_1"] = joint[0, 1]
            record["joint_prob_1_0"] = joint[1, 0]
            record["joint_prob_1_1"] = joint[1, 1]
            record["joint_low_score_corner_probability"] = joint[0, 0] + joint[0, 1] + joint[1, 0] + joint[1, 1]
            record["joint_tail_probability"] = joint[-1, :].sum() + joint[:-1, -1].sum()
            home_win, draw, away_win = dixon_coles_outcome_probabilities(
                home_lambda,
                away_lambda,
                rho,
                max_goals=config.outcome_grid_max_goals,
            )
        else:
            record["joint_probability_model"] = "independent_poisson"
            record["joint_prob_0_0"] = home_dist[0] * away_dist[0]
            record["joint_prob_0_1"] = home_dist[0] * away_dist[1]
            record["joint_prob_1_0"] = home_dist[1] * away_dist[0]
            record["joint_prob_1_1"] = home_dist[1] * away_dist[1]
            record["joint_low_score_corner_probability"] = (
                record["joint_prob_0_0"]
                + record["joint_prob_0_1"]
                + record["joint_prob_1_0"]
                + record["joint_prob_1_1"]
            )
            record["joint_tail_probability"] = home_dist[-1] + away_dist[-1] - home_dist[-1] * away_dist[-1]
            home_win, draw, away_win = outcome_probabilities(
                home_lambda,
                away_lambda,
                max_goals=config.outcome_grid_max_goals,
            )
        record["home_win_probability"] = home_win
        record["draw_probability"] = draw
        record["away_win_probability"] = away_win
        rows.append(record)
    output = pd.DataFrame(rows)
    validate_probability_frame(output, config)
    return output


def outcome_probabilities(home_lambda: float, away_lambda: float, *, max_goals: int) -> tuple[float, float, float]:
    home = np.array(poisson_goal_distribution(home_lambda, max_goals=max_goals)[:-1])
    away = np.array(poisson_goal_distribution(away_lambda, max_goals=max_goals)[:-1])
    matrix = np.outer(home, away)
    total = matrix.sum()
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    matrix = matrix / total
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    norm = home_win + draw + away_win
    return home_win / norm, draw / norm, away_win / norm


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_lambda: float,
    away_lambda: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        tau = 1 - home_lambda * away_lambda * rho
    elif home_goals == 0 and away_goals == 1:
        tau = 1 + home_lambda * rho
    elif home_goals == 1 and away_goals == 0:
        tau = 1 + away_lambda * rho
    elif home_goals == 1 and away_goals == 1:
        tau = 1 - rho
    else:
        tau = 1.0
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError("Invalid Dixon-Coles tau; rho is outside the valid range for these lambdas.")
    return float(tau)


def dixon_coles_score_probability(
    home_goals: int,
    away_goals: int,
    home_lambda: float,
    away_lambda: float,
    rho: float,
) -> float:
    base = _poisson_probability(home_goals, home_lambda) * _poisson_probability(away_goals, away_lambda)
    return float(base * dixon_coles_tau(home_goals, away_goals, home_lambda, away_lambda, rho))


def dixon_coles_joint_distribution(
    home_lambda: float,
    away_lambda: float,
    rho: float,
    *,
    max_goals: int,
) -> np.ndarray:
    home = np.array(poisson_goal_distribution(home_lambda, max_goals=max_goals))
    away = np.array(poisson_goal_distribution(away_lambda, max_goals=max_goals))
    matrix = np.outer(home, away)
    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
        matrix[home_goals, away_goals] *= dixon_coles_tau(
            home_goals,
            away_goals,
            home_lambda,
            away_lambda,
            rho,
        )
    if (matrix < -1e-12).any() or (matrix > 1 + 1e-12).any():
        raise ValueError("Invalid Dixon-Coles joint probabilities.")
    if not np.isclose(matrix.sum(), 1.0, atol=1e-9):
        raise ValueError("Dixon-Coles joint distribution does not sum to one.")
    return matrix


def dixon_coles_outcome_probabilities(
    home_lambda: float,
    away_lambda: float,
    rho: float,
    *,
    max_goals: int,
) -> tuple[float, float, float]:
    home_win, draw, away_win = outcome_probabilities(home_lambda, away_lambda, max_goals=max_goals)
    low_deltas = {
        (0, 0): _dc_delta(0, 0, home_lambda, away_lambda, rho),
        (0, 1): _dc_delta(0, 1, home_lambda, away_lambda, rho),
        (1, 0): _dc_delta(1, 0, home_lambda, away_lambda, rho),
        (1, 1): _dc_delta(1, 1, home_lambda, away_lambda, rho),
    }
    draw += low_deltas[(0, 0)] + low_deltas[(1, 1)]
    away_win += low_deltas[(0, 1)]
    home_win += low_deltas[(1, 0)]
    total = home_win + draw + away_win
    if total <= 0:
        raise ValueError("Invalid Dixon-Coles outcome probabilities.")
    probabilities = (home_win / total, draw / total, away_win / total)
    if min(probabilities) < -1e-12 or max(probabilities) > 1 + 1e-12:
        raise ValueError("Dixon-Coles outcome probabilities outside [0, 1].")
    return tuple(float(max(0.0, min(1.0, value))) for value in probabilities)


def validate_probability_frame(frame: pd.DataFrame, config: TeamModelConfig) -> None:
    probability_columns = [column for column in frame.columns if column.endswith("_probability")]
    goal_columns = [
        column
        for column in frame.columns
        if "_goals_prob_" in column
    ]
    joint_columns = [
        column
        for column in frame.columns
        if column.startswith("joint_prob_")
    ]
    for column in [*probability_columns, *goal_columns, *joint_columns]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any() or ((values < -1e-12) | (values > 1 + 1e-12)).any():
            raise ValueError(f"Invalid probability values in {column}.")
    for prefix in ("home", "away"):
        columns = [column for column in frame.columns if column.startswith(f"{prefix}_goals_prob_")]
        sums = frame[columns].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise ValueError(f"{prefix} goal probabilities do not sum to one.")
    outcome_sum = frame[
        ["home_win_probability", "draw_probability", "away_win_probability"]
    ].sum(axis=1)
    if not np.allclose(outcome_sum, 1.0, atol=1e-9):
        raise ValueError("Match-outcome probabilities do not sum to one.")


def _poisson_probability(goals: int, lam: float) -> float:
    return math.exp(-lam) * lam**goals / math.factorial(goals)


def _dc_delta(home_goals: int, away_goals: int, home_lambda: float, away_lambda: float, rho: float) -> float:
    independent = _poisson_probability(home_goals, home_lambda) * _poisson_probability(away_goals, away_lambda)
    corrected = independent * dixon_coles_tau(home_goals, away_goals, home_lambda, away_lambda, rho)
    return corrected - independent
