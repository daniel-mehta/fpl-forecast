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
        home_dist = poisson_goal_distribution(
            record["expected_home_goals"],
            max_goals=config.probability_max_goals,
        )
        away_dist = poisson_goal_distribution(
            record["expected_away_goals"],
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
        home_win, draw, away_win = outcome_probabilities(
            record["expected_home_goals"],
            record["expected_away_goals"],
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


def validate_probability_frame(frame: pd.DataFrame, config: TeamModelConfig) -> None:
    probability_columns = [column for column in frame.columns if column.endswith("_probability")]
    goal_columns = [
        column
        for column in frame.columns
        if "_goals_prob_" in column
    ]
    for column in [*probability_columns, *goal_columns]:
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
