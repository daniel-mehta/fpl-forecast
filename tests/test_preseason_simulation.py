from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fpl_forecast.xpoints.config import load_xpoints_config
from fpl_forecast.xpoints.scoring import score_frame
from fpl_forecast.xpoints.simulation import EXPECTED_COMPONENTS, simulate_component_points, stable_seed


def _base_row(
    player_uid: str,
    *,
    position: str = "MID",
    team: str = "home",
    p_appearance: float = 1.0,
    p_reached_60: float = 1.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "season": "2026-27",
        "gameweek": 1,
        "stable_fixture_uid": "fixture_1",
        "player_uid": player_uid,
        "player_team_uid": team,
        "fpl_position": position,
        "p_appearance": p_appearance,
        "p_reached_60": p_reached_60,
        "team_expected_goals": 0.0,
        "goal_share": 1.0,
        "assist_share": 1.0,
        "assisted_goal_rate": 0.0,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "expected_saves": 0.0,
        "expected_penalty_saves": 0.0,
        "expected_penalty_misses": 0.0,
        "expected_yellow_cards": 0.0,
        "expected_red_cards": 0.0,
        "expected_own_goals": 0.0,
        "expected_defensive_contribution": 0.0,
        "defensive_contribution_threshold": 10**9,
        "defensive_contribution_points": 0,
        "expected_bonus": 0.0,
    }
    row.update({column: 0.0 for column in EXPECTED_COMPONENTS})
    row["expected_points_appearance"] = p_appearance + p_reached_60
    return row


@pytest.mark.parametrize(
    ("label", "p_appearance", "p_reached_60", "expected"),
    [
        ("certain_dnp", 0.0, 0.0, 0.0),
        ("certain_30_minute_substitute", 1.0, 0.0, 1.0),
        ("certain_59_minute_appearance", 1.0, 0.0, 1.0),
        ("certain_60_minute_appearance", 1.0, 1.0, 2.0),
        ("certain_90_minute_appearance", 1.0, 1.0, 2.0),
        ("mixed_dnp_sub_start", 0.7, 0.4, 1.1),
    ],
)
def test_hand_calculated_appearance_expectations(label, p_appearance, p_reached_60, expected):
    _ = label
    config = replace(load_xpoints_config(), draw_count=500)
    frame = pd.DataFrame(
        [_base_row("player", p_appearance=p_appearance, p_reached_60=p_reached_60)]
    )
    summary, draws = simulate_component_points(frame, config=config, seed=99)

    assert summary.iloc[0]["expected_points"] == pytest.approx(expected)
    assert summary.iloc[0]["expected_points_appearance"] == pytest.approx(expected)
    if p_appearance == 0:
        assert np.count_nonzero(draws) == 0


def test_unconditional_and_conditional_points_apply_appearance_once():
    config = replace(load_xpoints_config(), draw_count=1000)
    row = _base_row("player", p_appearance=0.25, p_reached_60=0.0)
    row["expected_points_goals"] = 1.0
    summary, _ = simulate_component_points(pd.DataFrame([row]), config=config, seed=7)

    assert summary.iloc[0]["expected_points_unconditional"] == pytest.approx(1.25)
    assert summary.iloc[0]["expected_points_given_appearance"] == pytest.approx(5.0)
    assert summary.iloc[0]["conditional_coherence_error"] == 0.0


def test_one_expected_goal_is_allocated_to_one_certain_participant():
    config = replace(load_xpoints_config(), draw_count=10_000)
    row = _base_row("striker", position="FWD")
    row["team_expected_goals"] = 1.0
    row["expected_goals"] = 1.0
    row["expected_points_goals"] = 4.0
    summary, draws = simulate_component_points(pd.DataFrame([row]), config=config, seed=123)

    simulated_goal_mean = ((draws[0] - 2) / 4).mean()
    assert summary.iloc[0]["expected_points"] == pytest.approx(6.0)
    assert simulated_goal_mean == pytest.approx(1.0, abs=0.03)


def test_goal_allocation_uses_known_two_player_shares_and_is_row_order_invariant():
    config = replace(load_xpoints_config(), draw_count=10_000)
    first = _base_row("a", position="FWD")
    second = _base_row("b", position="FWD")
    for row, share in ((first, 0.75), (second, 0.25)):
        row["team_expected_goals"] = 1.0
        row["goal_share"] = share
        row["expected_goals"] = share
        row["expected_points_goals"] = share * 4
    frame = pd.DataFrame([first, second])

    summary, draws = simulate_component_points(frame, config=config, seed=321)
    reversed_summary, reversed_draws = simulate_component_points(
        frame.iloc[::-1].reset_index(drop=True),
        config=config,
        seed=321,
    )

    goal_means = ((draws - 2) / 4).mean(axis=1)
    assert goal_means[0] == pytest.approx(0.75, abs=0.03)
    assert goal_means[1] == pytest.approx(0.25, abs=0.03)
    pd.testing.assert_frame_equal(
        summary.reset_index(drop=True),
        reversed_summary.iloc[::-1].reset_index(drop=True),
    )
    assert np.array_equal(draws, reversed_draws[::-1])


def test_scoring_boundaries_and_negative_components_are_hand_calculated():
    frame = pd.DataFrame(
        [
            {"fpl_position": "GKP", "minutes": 60, "clean_sheets": 1, "goals_conceded": 2, "saves": 2},
            {"fpl_position": "GKP", "minutes": 60, "clean_sheets": 0, "goals_conceded": 2, "saves": 3},
            {"fpl_position": "DEF", "minutes": 59, "clean_sheets": 1, "goals_conceded": 2},
            {"fpl_position": "MID", "minutes": 60, "clean_sheets": 1, "yellow_cards": 1},
            {"fpl_position": "FWD", "minutes": 90, "red_cards": 1, "own_goals": 1},
        ]
    )
    assert score_frame(frame).tolist() == [5, 2, 0, 2, -3]


def test_defensive_contribution_threshold_boundaries_are_position_specific():
    frame = pd.DataFrame(
        [
            {"fpl_position": "DEF", "minutes": 90, "defensive_contribution": 9},
            {"fpl_position": "DEF", "minutes": 90, "defensive_contribution": 10},
            {"fpl_position": "MID", "minutes": 90, "defensive_contribution": 11},
            {"fpl_position": "MID", "minutes": 90, "defensive_contribution": 12},
        ]
    )
    assert score_frame(frame).tolist() == [2, 4, 2, 4]


def test_component_expectations_reconcile_exactly_to_analytic_total():
    config = replace(load_xpoints_config(), draw_count=500)
    row = _base_row("player")
    for index, column in enumerate(EXPECTED_COMPONENTS, start=1):
        row[column] = index / 10
    summary, _ = simulate_component_points(pd.DataFrame([row]), config=config, seed=10)

    expected = sum(index / 10 for index in range(1, len(EXPECTED_COMPONENTS) + 1))
    assert summary.iloc[0]["expected_points"] == pytest.approx(expected)
    assert summary.iloc[0]["component_points_sum"] == pytest.approx(expected)
    assert summary.iloc[0]["component_reconciliation_error"] == 0.0


def test_stable_seed_is_deterministic_and_identifier_sensitive():
    assert stable_seed(6060, "fixture", "player") == stable_seed(6060, "fixture", "player")
    assert stable_seed(6060, "fixture", "player") != stable_seed(6060, "fixture", "other")


def test_shared_fixture_scoreline_conserves_goals_and_clean_sheets():
    config = replace(load_xpoints_config(), draw_count=5000)
    home = _base_row("home_forward", position="FWD", team="home", p_reached_60=0.0)
    away = _base_row("away_forward", position="FWD", team="away", p_reached_60=0.0)
    home.update({"team_expected_goals": 1.2, "expected_goals": 1.2, "expected_points_goals": 4.8})
    away.update({"team_expected_goals": 0.8, "expected_goals": 0.8, "expected_points_goals": 3.2})
    _, draws = simulate_component_points(pd.DataFrame([home, away]), config=config, seed=42)
    home_goals = (draws[0] - 1) // 4
    away_goals = (draws[1] - 1) // 4

    expected_home = np.random.default_rng(
        stable_seed(
            42,
            config.simulation_version,
            "xpoints",
            "2026-27",
            "1",
            "fixture_1",
            "home",
            "scoreline",
        )
    ).poisson(1.2, config.draw_count)
    expected_away = np.random.default_rng(
        stable_seed(
            42,
            config.simulation_version,
            "xpoints",
            "2026-27",
            "1",
            "fixture_1",
            "away",
            "scoreline",
        )
    ).poisson(0.8, config.draw_count)
    assert np.array_equal(home_goals, expected_home)
    assert np.array_equal(away_goals, expected_away)


@pytest.mark.slow
def test_distribution_converges_at_production_draw_count():
    frame = pd.DataFrame(
        [
            {
                **_base_row("home", position="MID", team="home", p_appearance=0.9, p_reached_60=0.7),
                "team_expected_goals": 1.4,
                "expected_goals": 0.7,
                "expected_points_goals": 3.5,
                "goal_share": 1.0,
            },
            {
                **_base_row("away", position="FWD", team="away", p_appearance=0.8, p_reached_60=0.5),
                "team_expected_goals": 1.0,
                "expected_goals": 0.5,
                "expected_points_goals": 2.0,
                "goal_share": 1.0,
            },
        ]
    )
    config = load_xpoints_config()
    production, _ = simulate_component_points(frame, config=config, seed=6060)
    reference, _ = simulate_component_points(
        frame,
        config=replace(config, draw_count=20_000),
        seed=6060,
    )

    assert (
        production["prob_points_ge_5"] - reference["prob_points_ge_5"]
    ).abs().quantile(0.95) <= config.probability_p95_abs_tolerance
    assert production["component_reconciliation_error"].abs().max() <= (
        config.component_reconciliation_tolerance
    )
