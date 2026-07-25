from __future__ import annotations

import hashlib

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

EXPECTED_COMPONENTS = [
    "expected_points_appearance",
    "expected_points_goals",
    "expected_points_assists",
    "expected_points_clean_sheets",
    "expected_points_saves",
    "expected_points_penalties",
    "expected_points_goals_conceded",
    "expected_points_cards",
    "expected_points_own_goals",
    "expected_points_defensive_contribution",
    "expected_points_bonus",
]

CONDITIONAL_SUMMARY_COLUMNS = [
    "expected_points_unconditional",
    "raw_expected_points_given_appearance",
    "expected_points_given_appearance",
    "appearance_draw_count",
    "simulation_draw_count",
    "simulation_appearance_probability",
    "conditional_points_prior",
    "conditional_points_prior_strength",
    "conditional_estimate_reliability",
    "conditional_estimate_source",
    "conditional_coherence_error",
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
    seed_namespace: str = "xpoints",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Hybrid xPoints calculation.

    Component means are analytic, unconditional expectations from ``frame``.
    Integer draws provide fixture-coherent distributions only.  Draws never
    replace the analytic mean or the full-precision conditional value.
    """
    if frame.empty:
        return pd.DataFrame(index=frame.index), np.empty((0, config.draw_count), dtype=np.int16)
    ordered = _canonical_order(frame)
    if "player_team_uid" not in ordered:
        ordered["player_team_uid"] = [f"standalone_team_{value}" for value in range(len(ordered))]
    if "team_expected_goals" not in ordered:
        expected_goals = ordered.get("expected_goals", pd.Series(0.0, index=ordered.index))
        ordered["team_expected_goals"] = pd.to_numeric(expected_goals, errors="coerce").fillna(0.0)
    if "goal_share" not in ordered:
        ordered["goal_share"] = 1.0
    if "assist_share" not in ordered:
        ordered["assist_share"] = 1.0
    if "assisted_goal_rate" not in ordered:
        ordered["assisted_goal_rate"] = 0.72
    draws = np.zeros((len(ordered), config.draw_count), dtype=np.int16)
    appearances = np.zeros((len(ordered), config.draw_count), dtype=bool)
    components = {column: np.zeros_like(draws) for column in EXPECTED_COMPONENTS}

    fixture_key = "stable_fixture_uid" if "stable_fixture_uid" in ordered else "_fixture_key"
    if fixture_key == "_fixture_key":
        ordered[fixture_key] = [f"row_{value}" for value in range(len(ordered))]
    for fixture_id, fixture in ordered.groupby(fixture_key, sort=True, dropna=False):
        positions = fixture.index.to_numpy(dtype=int)
        _simulate_fixture(
            fixture,
            positions=positions,
            draws=draws,
            appearances=appearances,
            components=components,
            config=config,
            master_seed=seed,
            seed_namespace=seed_namespace,
            fixture_id=str(fixture_id),
        )

    summaries: list[dict[str, float | int | str]] = []
    analytic_components = (
        ordered.reindex(columns=EXPECTED_COMPONENTS, fill_value=0.0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    analytic_total = analytic_components.sum(axis=1)
    for idx in range(len(ordered)):
        summary: dict[str, float | int | str] = summarize_draws(draws[idx])
        summary["simulated_expected_points"] = float(summary["expected_points"])
        summary["expected_points"] = float(analytic_total.iloc[idx])
        for column in EXPECTED_COMPONENTS:
            summary[column] = float(analytic_components.iloc[idx][column])
        summary["component_points_sum"] = float(analytic_total.iloc[idx])
        summary["component_reconciliation_error"] = 0.0
        p_app = _clip(ordered.iloc[idx].get("p_appearance", 0.0))
        conditional = float(analytic_total.iloc[idx] / p_app) if p_app > 0 else 0.0
        appearance_count = int(appearances[idx].sum())
        summary.update(
            {
                "expected_points_unconditional": float(analytic_total.iloc[idx]),
                "raw_expected_points_given_appearance": conditional,
                "expected_points_given_appearance": conditional,
                "appearance_draw_count": appearance_count,
                "simulation_draw_count": int(config.draw_count),
                "simulation_appearance_probability": appearance_count / float(config.draw_count),
                "conditional_points_prior": 0.0,
                "conditional_points_prior_strength": 0.0,
                "conditional_estimate_reliability": 1.0 if p_app > 0 else 0.0,
                "conditional_estimate_source": "analytic_unconditional_divided_by_frozen_appearance_probability",
                "conditional_coherence_error": 0.0,
            }
        )
        summaries.append(summary)
    result = pd.DataFrame(summaries)
    inverse = np.argsort(ordered["_input_position"].to_numpy(dtype=int))
    result = result.iloc[inverse].set_axis(frame.index)
    return result, draws[inverse]


def _simulate_fixture(
    fixture: pd.DataFrame,
    *,
    positions: np.ndarray,
    draws: np.ndarray,
    appearances: np.ndarray,
    components: dict[str, np.ndarray],
    config: XPointsConfig,
    master_seed: int,
    seed_namespace: str,
    fixture_id: str,
) -> None:
    n_draws = config.draw_count
    season = str(fixture["season"].iloc[0]) if "season" in fixture else "unknown_season"
    gameweek = str(fixture["gameweek"].iloc[0]) if "gameweek" in fixture else "unknown_gameweek"
    for local, (_, row) in enumerate(fixture.iterrows()):
        target = positions[local]
        player_id = str(row.get("player_uid", target))
        rng = np.random.default_rng(
            stable_seed(
                master_seed,
                config.simulation_version,
                seed_namespace,
                season,
                gameweek,
                fixture_id,
                player_id,
                "player",
            )
        )
        p_app = _clip(row.get("p_appearance", 0.0))
        p60 = min(_clip(row.get("p_reached_60", 0.0)), p_app)
        app = rng.random(n_draws) < p_app
        reached = app & (rng.random(n_draws) < (p60 / p_app if p_app > 0 else 0.0))
        appearances[target] = app
        components["expected_points_appearance"][target] = app.astype(np.int16) + reached.astype(np.int16)
        conditional_saves = _conditional_rate(row.get("expected_saves", 0.0), p_app)
        saves = rng.poisson(conditional_saves, n_draws) * app
        components["expected_points_saves"][target] = saves // 3
        components["expected_points_penalties"][target] = (
            5 * (rng.random(n_draws) < _conditional_probability(row.get("expected_penalty_saves", 0.0), p_app))
            - 2 * (rng.random(n_draws) < _conditional_probability(row.get("expected_penalty_misses", 0.0), p_app))
        ) * app
        yellow = rng.random(n_draws) < _conditional_probability(row.get("expected_yellow_cards", 0.0), p_app)
        red = rng.random(n_draws) < _conditional_probability(row.get("expected_red_cards", 0.0), p_app)
        own = rng.random(n_draws) < _conditional_probability(row.get("expected_own_goals", 0.0), p_app)
        components["expected_points_cards"][target] = -(yellow.astype(np.int16) + 3 * red.astype(np.int16)) * app
        components["expected_points_own_goals"][target] = -2 * own.astype(np.int16) * app
        dc = rng.poisson(_conditional_rate(row.get("expected_defensive_contribution", 0.0), p_app), n_draws) * app
        threshold = int(row.get("defensive_contribution_threshold", 10**9) or 10**9)
        dc_points = int(row.get("defensive_contribution_points", 0) or 0)
        components["expected_points_defensive_contribution"][target] = (dc >= threshold) * dc_points
        conditional_bonus = min(_conditional_rate(row.get("expected_bonus", 0.0), p_app), 3.0)
        bonus_floor = int(np.floor(conditional_bonus))
        bonus = (
            bonus_floor + (rng.random(n_draws) < (conditional_bonus - bonus_floor)).astype(np.int16)
        ) * app
        components["expected_points_bonus"][target] = bonus

    teams = sorted(str(value) for value in fixture["player_team_uid"].dropna().unique())
    team_goals: dict[str, np.ndarray] = {}
    for team in teams:
        team_rows = fixture.loc[fixture["player_team_uid"].astype(str).eq(team)]
        lam = float(pd.to_numeric(team_rows["team_expected_goals"], errors="coerce").dropna().iloc[0])
        rng = np.random.default_rng(
            stable_seed(
                master_seed,
                config.simulation_version,
                seed_namespace,
                season,
                gameweek,
                fixture_id,
                team,
                "scoreline",
            )
        )
        team_goals[team] = rng.poisson(max(lam, 0.0), n_draws)

    for team in teams:
        team_rows = fixture.loc[fixture["player_team_uid"].astype(str).eq(team)]
        team_positions = team_rows.index.to_numpy(dtype=int)
        opponent = next((value for value in teams if value != team), None)
        conceded = team_goals.get(opponent, np.zeros(n_draws, dtype=int))
        team_apps = appearances[team_positions]
        scorer_shares = _probabilities(
            team_rows.get("goal_share", pd.Series(0.0, index=team_rows.index))
        )
        assist_shares = _probabilities(
            team_rows.get("assist_share", pd.Series(0.0, index=team_rows.index))
        )
        scorer_weights = _calibrated_active_weights(scorer_shares, team_apps)
        assist_weights = _calibrated_active_weights(assist_shares, team_apps)
        assisted_rate = _clip(team_rows.get("assisted_goal_rate", pd.Series([0.72])).iloc[0])
        rng = np.random.default_rng(
            stable_seed(
                master_seed,
                config.simulation_version,
                seed_namespace,
                season,
                gameweek,
                fixture_id,
                team,
                "allocation",
            )
        )
        for draw_index, goal_count in enumerate(team_goals[team]):
            if not goal_count:
                continue
            active = team_apps[:, draw_index]
            if not active.any():
                # This event has negligible probability for a full selectable team.
                # Condition the allocation on at least one participant so every team
                # goal has a footballer scorer without assigning events to a DNP.
                fallback_local = int(np.argmax(scorer_weights))
                fallback_target = team_positions[fallback_local]
                if _clip(fixture.loc[fallback_target].get("p_appearance", 0.0)) == 0:
                    continue
                active[fallback_local] = True
                appearances[fallback_target, draw_index] = True
                components["expected_points_appearance"][
                    fallback_target, draw_index
                ] = 1
            scorer_probs = _active_probabilities(scorer_weights, active)
            for _ in range(int(goal_count)):
                scorer_local = int(rng.choice(len(team_positions), p=scorer_probs))
                scorer_target = team_positions[scorer_local]
                goal_points = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}.get(
                    str(fixture.loc[scorer_target, "fpl_position"]),
                    0,
                )
                components["expected_points_goals"][scorer_target, draw_index] += goal_points
                eligible = active.copy()
                eligible[scorer_local] = False
                if eligible.any() and rng.random() < assisted_rate:
                    assist_local = int(rng.choice(len(team_positions), p=_active_probabilities(assist_weights, eligible)))
                    components["expected_points_assists"][team_positions[assist_local], draw_index] += 3
        for target in team_positions:
            position = str(fixture.loc[target, "fpl_position"])
            reached = components["expected_points_appearance"][target] == 2
            clean_points = 4 if position in {"GKP", "DEF"} else 1 if position == "MID" else 0
            components["expected_points_clean_sheets"][target] = (conceded == 0) * reached * clean_points
            if position in {"GKP", "DEF"}:
                components["expected_points_goals_conceded"][target] = -(conceded // 2) * reached

    for component_draws in components.values():
        draws[positions] += component_draws[positions].astype(np.int16)


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
        row = {**dict(zip(key_columns, key_values, strict=False)), **summarize_draws(summed)}
        analytic = (
            float(pd.to_numeric(group["expected_points"], errors="coerce").sum())
            if "expected_points" in group
            else float(row["expected_points"])
        )
        row["simulated_expected_points"] = row["expected_points"]
        row["expected_points"] = analytic
        p_any = (
            1.0
            - float(
                np.prod(
                    1.0
                    - pd.to_numeric(group["p_appearance"], errors="coerce")
                    .fillna(0.0)
                    .clip(0, 1)
                )
            )
            if "p_appearance" in group
            else float((summed != 0).mean())
        )
        conditional = analytic / p_any if p_any > 0 else 0.0
        row.update(
            {
                "expected_points_unconditional": analytic,
                "raw_expected_points_given_appearance": conditional,
                "expected_points_given_appearance": conditional,
                "simulation_draw_count": draws.shape[1],
                "conditional_estimate_source": "analytic_gameweek_sum_divided_by_any_fixture_appearance_probability",
                "conditional_coherence_error": 0.0,
            }
        )
        rows.append(row)
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


def stable_seed(master_seed: int, *identifiers: object) -> int:
    material = "\x1f".join([str(master_seed), *(str(value) for value in identifiers)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _canonical_order(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.reset_index(drop=False).rename(columns={"index": "_source_index"})
    output["_input_position"] = np.arange(len(output))
    keys = [
        column
        for column in ("season", "gameweek", "stable_fixture_uid", "player_team_uid", "player_uid")
        if column in output
    ]
    return output.sort_values(keys, kind="stable").reset_index(drop=True)


def _weights(values: pd.Series) -> np.ndarray:
    weights = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    return weights if weights.sum() > 0 else np.ones(len(weights), dtype=float)


def _active_probabilities(weights: np.ndarray, active: np.ndarray) -> np.ndarray:
    eligible = np.where(active, weights, 0.0)
    if eligible.sum() <= 0:
        eligible = active.astype(float)
    return eligible / eligible.sum()


def _calibrated_active_weights(
    target_shares: np.ndarray,
    active_matrix: np.ndarray,
    *,
    iterations: int = 40,
) -> np.ndarray:
    """Calibrate conditional-on-active choice weights to unconditional shares."""
    active = active_matrix.astype(float)
    valid_draws = active.sum(axis=0) > 0
    if not valid_draws.any():
        return np.ones(len(target_shares), dtype=float)
    active = active[:, valid_draws]
    weights = np.divide(
        target_shares,
        active.mean(axis=1),
        out=np.ones_like(target_shares),
        where=active.mean(axis=1) > 0,
    )
    weights = np.clip(weights, 1e-12, None)
    for _ in range(iterations):
        weighted = active * weights[:, None]
        probabilities = weighted / weighted.sum(axis=0, keepdims=True)
        current = probabilities.mean(axis=1)
        ratio = np.divide(
            target_shares,
            current,
            out=np.ones_like(target_shares),
            where=current > 0,
        )
        weights *= ratio
        weights = np.clip(weights, 1e-12, 1e12)
    return weights


def _probabilities(values: pd.Series) -> np.ndarray:
    weights = _weights(values)
    return weights / weights.sum()


def _conditional_rate(unconditional: object, p_appearance: float) -> float:
    value = max(float(unconditional or 0.0), 0.0)
    return value / p_appearance if p_appearance > 0 else 0.0


def _conditional_probability(unconditional: object, p_appearance: float) -> float:
    return _clip(_conditional_rate(unconditional, p_appearance))


def _clip(value: object) -> float:
    return float(np.clip(float(value or 0), 0, 1))
