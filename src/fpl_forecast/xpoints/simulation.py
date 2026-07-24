from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_forecast.xpoints.config import XPointsConfig


POINT_COMPONENTS = [
    "points_appearance",
    "points_goals",
    "points_assists",
    "points_clean_sheets",
    "points_saves",
    "points_penalties",
    "points_goals_conceded",
    "points_cards",
    "points_own_goals",
    "points_defensive_contribution",
    "points_bonus",
]


def summarize_draws(draws: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    values = draws.astype(float)
    return {
        f"{prefix}expected_points": float(values.mean()),
        f"{prefix}points_std": float(values.std(ddof=0)),
        f"{prefix}points_p10": float(np.quantile(values, 0.10)),
        f"{prefix}points_p25": float(np.quantile(values, 0.25)),
        f"{prefix}points_p50": float(np.quantile(values, 0.50)),
        f"{prefix}points_p75": float(np.quantile(values, 0.75)),
        f"{prefix}points_p90": float(np.quantile(values, 0.90)),
        f"{prefix}prob_points_eq_0": float((values == 0).mean()),
        f"{prefix}prob_points_ge_1": float((values >= 1).mean()),
        f"{prefix}prob_points_ge_5": float((values >= 5).mean()),
        f"{prefix}prob_points_ge_10": float((values >= 10).mean()),
    }


def simulate_component_points(
    frame: pd.DataFrame,
    *,
    config: XPointsConfig,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws = np.zeros((len(frame), config.draw_count), dtype=np.int16)
    summaries = []
    for idx, row in enumerate(frame.itertuples(index=False)):
        p_app = _clip(row.p_appearance)
        p60 = min(_clip(row.p_reached_60), p_app)
        app = rng.binomial(1, p_app, config.draw_count)
        reached_60 = rng.binomial(1, 0 if p_app <= 0 else p60 / p_app, config.draw_count) * app
        goals = rng.poisson(max(row.expected_goals, 0), config.draw_count) * app
        assists = rng.poisson(max(row.expected_assists, 0), config.draw_count) * app
        clean = rng.binomial(1, _clip(row.clean_sheet_probability), config.draw_count) * reached_60
        saves = rng.poisson(max(row.expected_saves, 0), config.draw_count) * app
        pen_saved = rng.binomial(1, _clip(row.expected_penalty_saves), config.draw_count) * app
        pen_missed = rng.binomial(1, _clip(row.expected_penalty_misses), config.draw_count) * app
        yellow = rng.binomial(1, _clip(row.expected_yellow_cards), config.draw_count) * app
        red = rng.binomial(1, _clip(row.expected_red_cards), config.draw_count) * app
        own = rng.binomial(1, _clip(row.expected_own_goals), config.draw_count) * app
        expected_defensive_contribution = getattr(row, "expected_defensive_contribution", 0)
        defensive_contribution = rng.poisson(max(expected_defensive_contribution, 0), config.draw_count) * app
        defensive_threshold = int(getattr(row, "defensive_contribution_threshold", 10**9) or 10**9)
        defensive_points = (defensive_contribution >= defensive_threshold).astype(int) * int(
            getattr(row, "defensive_contribution_points", 0) or 0
        )
        bonus = np.minimum(rng.poisson(max(row.expected_bonus, 0), config.draw_count), 3) * app
        gc_deduct = rng.poisson(max(row.expected_goals_conceded_deduction_events, 0), config.draw_count) * reached_60
        pos = row.fpl_position
        points = np.zeros(config.draw_count, dtype=int)
        component_draws = {
            "expected_points_appearance": app + reached_60,
            "expected_points_goals": goals * {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}.get(pos, 0),
            "expected_points_assists": assists * 3,
            "expected_points_clean_sheets": clean * (4 if pos in {"GKP", "DEF"} else 1 if pos == "MID" else 0),
            "expected_points_saves": saves // 3,
            "expected_points_penalties": pen_saved * 5 - pen_missed * 2,
            "expected_points_goals_conceded": -gc_deduct * (1 if pos in {"GKP", "DEF"} else 0),
            "expected_points_cards": -(yellow + red * 3),
            "expected_points_own_goals": -(own * 2),
            "expected_points_defensive_contribution": defensive_points,
            "expected_points_bonus": bonus,
        }
        for component_points in component_draws.values():
            points += component_points
        draws[idx, :] = points.astype(np.int16)
        summary = summarize_draws(draws[idx, :])
        for component, component_points in component_draws.items():
            summary[component] = float(component_points.mean())
        summary["component_points_sum"] = float(sum(summary[component] for component in component_draws))
        summary["component_reconciliation_error"] = float(summary["component_points_sum"] - summary["expected_points"])
        summaries.append(summary)
    return pd.DataFrame(summaries, index=frame.index), draws


def aggregate_gameweek_draws(
    fixture_predictions: pd.DataFrame,
    draws: np.ndarray,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    rows = []
    draw_df = pd.DataFrame(draws, index=fixture_predictions.index)
    for key, group in fixture_predictions.groupby(key_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        summed = draw_df.loc[group.index].sum(axis=0).to_numpy()
        rows.append({**dict(zip(key_columns, key_values, strict=False)), **summarize_draws(summed)})
    return pd.DataFrame(rows)


def coherent_goal_allocation(
    team_goals: int,
    active_players: pd.DataFrame,
    *,
    scorer_weight_column: str = "goal_weight",
    assist_weight_column: str = "assist_weight",
    no_assist_probability: float = 0.28,
    seed: int = 1,
) -> pd.DataFrame:
    if team_goals < 0:
        raise ValueError("team_goals must be nonnegative.")
    rng = np.random.default_rng(seed)
    output = active_players.copy()
    output["simulated_goals"] = 0
    output["simulated_assists"] = 0
    if team_goals == 0 or output.empty:
        return output
    scorer_probs = _probabilities(output[scorer_weight_column])
    for _ in range(team_goals):
        scorer_pos = int(rng.choice(np.arange(len(output)), p=scorer_probs))
        scorer_index = output.index[scorer_pos]
        output.loc[scorer_index, "simulated_goals"] += 1
        assister = output.drop(index=scorer_index)
        if assister.empty or rng.random() < no_assist_probability:
            continue
        assist_probs = _probabilities(assister[assist_weight_column])
        assist_pos = int(rng.choice(np.arange(len(assister)), p=assist_probs))
        output.loc[assister.index[assist_pos], "simulated_assists"] += 1
    return output


def _probabilities(values: pd.Series) -> np.ndarray:
    weights = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        return np.ones(len(weights)) / len(weights)
    return weights / weights.sum()


def _clip(value: float) -> float:
    return float(np.clip(float(value or 0), 0, 1))
