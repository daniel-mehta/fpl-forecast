from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_forecast.xpoints.components import award_bonus_from_bps
from fpl_forecast.xpoints.config import load_xpoints_config
from fpl_forecast.xpoints.data import assert_frozen_target_free
from fpl_forecast.xpoints.scoring import reconstruction_audit, score_frame
from fpl_forecast.xpoints.simulation import aggregate_gameweek_draws, coherent_goal_allocation, simulate_component_points


def test_scoring_engine_hand_calculates_all_components():
    frame = pd.DataFrame(
        [
            {
                "fpl_position": "DEF",
                "minutes": 90,
                "goals_scored": 1,
                "assists": 1,
                "clean_sheets": 1,
                "goals_conceded": 0,
                "saves": 0,
                "penalties_saved": 0,
                "penalties_missed": 1,
                "yellow_cards": 1,
                "red_cards": 0,
                "own_goals": 1,
                "bonus": 3,
            },
            {
                "fpl_position": "GKP",
                "minutes": 59,
                "goals_scored": 0,
                "assists": 0,
                "clean_sheets": 1,
                "goals_conceded": 4,
                "saves": 7,
                "penalties_saved": 1,
                "penalties_missed": 0,
                "yellow_cards": 0,
                "red_cards": 1,
                "own_goals": 0,
                "bonus": 0,
            },
        ]
    )

    points = score_frame(frame)

    assert points.tolist() == [13, 3]


def test_scoring_reconstruction_audit_exact_on_real_like_rows():
    frame = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "fpl_position": "MID",
                "entity_type": "player",
                "minutes": 60,
                "goals_scored": 1,
                "assists": 0,
                "clean_sheets": 1,
                "goals_conceded": 0,
                "saves": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "own_goals": 0,
                "bonus": 2,
                "total_points": 10,
            }
        ]
    )

    audit = reconstruction_audit(frame)

    assert audit["by_season_position"].iloc[0]["exact_match_pct"] == 1.0
    assert audit["difference_counts"].iloc[0]["point_difference"] == 0


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([40, 30, 20], [3, 2, 1]),
        ([40, 40, 20], [3, 3, 1]),
        ([40, 30, 30], [3, 2, 2]),
        ([40, 30, 20, 20], [3, 2, 1, 1]),
        ([40, 40, 40, 20], [3, 3, 3, 0]),
    ],
)
def test_bonus_tie_rules(scores, expected):
    frame = pd.DataFrame({"predicted_bps": scores, "eligible": [True] * len(scores)})

    assert award_bonus_from_bps(frame).tolist() == expected


def test_bonus_excludes_dnp_or_ineligible_players():
    frame = pd.DataFrame({"predicted_bps": [50, 40], "eligible": [False, True]})

    assert award_bonus_from_bps(frame).tolist() == [0, 3]


def test_coherent_goal_allocation_conserves_goals_and_prevents_self_assist():
    players = pd.DataFrame(
        {
            "player_uid": ["a", "b", "c"],
            "goal_weight": [10.0, 1.0, 1.0],
            "assist_weight": [0.0, 5.0, 5.0],
        }
    )

    allocated = coherent_goal_allocation(3, players, no_assist_probability=0.0, seed=7)

    assert allocated["simulated_goals"].sum() == 3
    assert allocated["simulated_assists"].sum() == 3
    assert (allocated["simulated_goals"] + allocated["simulated_assists"]).ge(0).all()


def test_simulation_outputs_are_deterministic_finite_and_monotonic():
    config = load_xpoints_config()
    frame = pd.DataFrame(
        [
            {
                "fpl_position": "FWD",
                "p_appearance": 0.9,
                "p_reached_60": 0.7,
                "expected_goals": 0.4,
                "expected_assists": 0.2,
                "clean_sheet_probability": 0.1,
                "expected_saves": 0,
                "expected_penalty_saves": 0,
                "expected_penalty_misses": 0.01,
                "expected_yellow_cards": 0.1,
                "expected_red_cards": 0.01,
                "expected_own_goals": 0.01,
                "expected_bonus": 0.2,
                "expected_goals_conceded_deduction_events": 0,
            }
        ]
    )

    first, first_draws = simulate_component_points(frame, config=config, seed=1)
    second, second_draws = simulate_component_points(frame, config=config, seed=1)

    pd.testing.assert_frame_equal(first, second)
    assert np.array_equal(first_draws, second_draws)
    assert first.iloc[0]["points_p10"] <= first.iloc[0]["points_p50"] <= first.iloc[0]["points_p90"]
    assert np.isfinite(first.to_numpy()).all()


def test_gameweek_aggregation_sums_draws_not_percentiles():
    predictions = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "gameweek": [30, 30],
            "player_uid": ["p1", "p1"],
            "model_name": ["X", "X"],
            "pre_deadline_population": ["pre_deadline_history_active"] * 2,
        }
    )
    draws = np.array([[0, 10, 10], [10, 0, 10]])

    aggregated = aggregate_gameweek_draws(
        predictions,
        draws,
        key_columns=["season", "gameweek", "player_uid", "model_name", "pre_deadline_population"],
    )

    assert aggregated.iloc[0]["expected_points"] == pytest.approx(40 / 3)
    assert aggregated.iloc[0]["points_p50"] == 10


def test_frozen_forbidden_columns_are_rejected_case_insensitively():
    config = load_xpoints_config()
    frame = pd.DataFrame({"player_uid": ["p1"], "Xp": [1.0]})

    with pytest.raises(ValueError, match="forbidden"):
        assert_frozen_target_free(frame, config.forbidden_frozen_columns)


def test_invalid_position_fails_scoring():
    frame = pd.DataFrame({"fpl_position": ["AM"], "minutes": [90]})

    with pytest.raises(ValueError, match="Invalid"):
        score_frame(frame)
