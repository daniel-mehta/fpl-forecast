from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fpl_forecast.xpoints.components import attacking_shares, award_bonus_from_bps
from fpl_forecast.xpoints.config import load_xpoints_config
from fpl_forecast.xpoints.data import assert_frozen_target_free
from fpl_forecast.xpoints.models import _expected_goal_points
from fpl_forecast.xpoints.scoring import reconstruction_audit, score_frame
from fpl_forecast.xpoints.rules import goal_point_values, goal_points_for_position
from fpl_forecast.xpoints.simulation import (
    aggregate_gameweek_draws,
    coherent_goal_allocation,
    simulate_component_points,
)


def test_scoring_engine_hand_calculates_all_components():
    frame = pd.DataFrame(
        [
            {
                "season": "2024-25",
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
                "season": "2024-25",
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


def test_scoring_engine_applies_defensive_contribution_thresholds():
    frame = pd.DataFrame(
        [
            {"season": "2025-26", "fpl_position": "DEF", "minutes": 90, "defensive_contribution": 10},
            {"season": "2025-26", "fpl_position": "MID", "minutes": 90, "defensive_contribution": 11},
            {"season": "2025-26", "fpl_position": "MID", "minutes": 90, "defensive_contribution": 12},
            {"season": "2025-26", "fpl_position": "FWD", "minutes": 90, "defensive_contribution": 12},
            {"season": "2025-26", "fpl_position": "GKP", "minutes": 90, "defensive_contribution": 20},
        ]
    )

    points = score_frame(frame)

    assert points.tolist() == [4, 2, 4, 4, 2]


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


def test_goalkeeper_goal_points_change_at_2024_25_and_persist_afterward():
    assert goal_points_for_position(season="2023-24", position="GKP") == 6
    assert goal_points_for_position(season="2024-25", position="GKP") == 10
    assert goal_points_for_position(season="2028-29", position="GKP") == 10


def test_goalkeeper_goal_season_boundary_leaves_outfield_values_unchanged():
    positions = pd.Series(["GKP", "DEF", "MID", "FWD"] * 2)
    seasons = pd.Series(["2023-24"] * 4 + ["2024-25"] * 4)

    assert goal_point_values(seasons, positions).tolist() == [6, 6, 5, 4, 10, 6, 5, 4]


def test_analytical_goal_component_uses_season_aware_goal_values():
    frame = pd.DataFrame(
        {
            "season": ["2023-24", "2024-25", "2025-26"],
            "fpl_position": ["GKP", "GKP", "GKP"],
            "expected_goals": [0.25, 0.25, 0.25],
        }
    )

    component = _expected_goal_points(frame)

    assert component.tolist() == [1.5, 2.5, 2.5]


@pytest.mark.parametrize(("season", "expected_goal_points"), [("2023-24", 6), ("2024-25", 10), ("2027-28", 10)])
def test_realized_goalkeeper_goal_scoring_is_season_aware(season, expected_goal_points):
    frame = pd.DataFrame(
        [{"season": season, "fpl_position": "GKP", "minutes": 1, "goals_scored": 1}]
    )

    assert score_frame(frame).iloc[0] == 1 + expected_goal_points


def test_reconstruction_audit_handles_synthetic_goalkeeper_goals_across_boundary():
    frame = pd.DataFrame(
        [
            {
                "entity_type": "player",
                "season": "2023-24",
                "fpl_position": "GKP",
                "minutes": 1,
                "goals_scored": 1,
                "total_points": 7,
            },
            {
                "entity_type": "player",
                "season": "2024-25",
                "fpl_position": "GKP",
                "minutes": 1,
                "goals_scored": 1,
                "total_points": 11,
            },
        ]
    )

    audit = reconstruction_audit(frame)

    assert audit["mismatches"].empty
    assert audit["difference_counts"].to_dict("records") == [{"point_difference": 0, "rows": 2}]


@pytest.mark.parametrize("season", [None, "2024/25", "2024-24", "not-a-season"])
def test_scoring_rejects_missing_or_malformed_season_values(season):
    data = {"fpl_position": ["GKP"], "minutes": [1], "goals_scored": [1]}
    if season is not None:
        data["season"] = [season]

    with pytest.raises(ValueError, match="season|Season"):
        score_frame(pd.DataFrame(data))


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
    config = replace(load_xpoints_config(), draw_count=500)
    frame = pd.DataFrame(
        [
            {
                "season": "2024-25",
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
    numeric = first.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
    assert first.iloc[0]["component_reconciliation_error"] == pytest.approx(0.0)
    assert first.iloc[0]["component_points_sum"] == pytest.approx(first.iloc[0]["expected_points"])
    assert first.iloc[0]["conditional_coherence_error"] == pytest.approx(0.0)
    assert first.iloc[0]["appearance_draw_count"] <= config.draw_count
    assert first.iloc[0]["simulation_draw_count"] == config.draw_count


def test_simulated_non_appearance_produces_exactly_zero_points():
    config = replace(load_xpoints_config(), draw_count=500)
    frame = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "fpl_position": "DEF",
                "p_appearance": 0.0,
                "p_reached_60": 0.0,
                "expected_goals": 2.0,
                "expected_assists": 2.0,
                "clean_sheet_probability": 1.0,
                "expected_saves": 10.0,
                "expected_penalty_saves": 1.0,
                "expected_penalty_misses": 1.0,
                "expected_yellow_cards": 1.0,
                "expected_red_cards": 1.0,
                "expected_own_goals": 1.0,
                "expected_defensive_contribution": 20.0,
                "defensive_contribution_threshold": 10,
                "defensive_contribution_points": 2,
                "expected_bonus": 3.0,
                "expected_goals_conceded_deduction_events": 5.0,
            }
        ]
    )

    summary, draws = simulate_component_points(frame, config=config, seed=9)

    assert summary.iloc[0]["expected_points"] == 0.0
    assert summary.iloc[0]["component_points_sum"] == 0.0
    assert np.array_equal(draws[0], np.zeros(config.draw_count, dtype=np.int16))


def test_attacking_shares_do_not_let_transferred_player_absorb_cold_start_team():
    config = load_xpoints_config()
    rates = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "stable_fixture_uid": "fixture_1",
                "player_uid": "transfer",
                "fpl_position": "MID",
                "player_team_uid": "team_promoted",
                "goals_scored_per90": 0.45,
                "assists_per90": 0.25,
            },
            *[
                {
                    "season": "2026-27",
                    "stable_fixture_uid": "fixture_1",
                    "player_uid": f"cold_{idx}",
                    "fpl_position": "MID" if idx % 2 else "FWD",
                    "player_team_uid": "team_promoted",
                    "goals_scored_per90": 0.0,
                    "assists_per90": 0.0,
                }
                for idx in range(8)
            ],
        ]
    )
    minutes = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "stable_fixture_uid": "fixture_1",
                "player_uid": "transfer",
                "expected_minutes": 65.0,
                "p_appearance": 0.85,
                "p_start": 0.70,
                "cold_start_no_history": False,
                "transferred_player": True,
            },
            *[
                {
                    "season": "2026-27",
                    "stable_fixture_uid": "fixture_1",
                    "player_uid": f"cold_{idx}",
                    "expected_minutes": 22.0,
                    "p_appearance": 0.35,
                    "p_start": 0.18,
                    "cold_start_no_history": True,
                    "transferred_player": False,
                }
                for idx in range(8)
            ],
        ]
    )

    shares = attacking_shares(rates, minutes, config)
    transfer = shares.loc[shares["player_uid"].eq("transfer")].iloc[0]

    assert transfer["goal_share"] < 0.75
    assert transfer["assist_share"] < 0.75
    assert transfer["effective_goal_attackers"] > 2.0
    assert shares["goal_share"].sum() == pytest.approx(1.0)
    assert shares["assist_share"].sum() == pytest.approx(1.0)


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
    frame = pd.DataFrame({"season": ["2025-26"], "fpl_position": ["AM"], "minutes": [90]})

    with pytest.raises(ValueError, match="Invalid"):
        score_frame(frame)
