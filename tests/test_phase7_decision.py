from __future__ import annotations

from itertools import combinations

import pandas as pd
import pytest

from fpl_forecast.decision.inputs import assert_frozen_decisions_target_free
from fpl_forecast.decision.expected_realized import (
    evaluate_expected_realized_points,
    optimize_lineup_expected_realized,
)
from fpl_forecast.decision.lineup import apply_autosubs_and_score, optimize_lineup
from fpl_forecast.decision.milp import optimize_squad_expected_realized, optimize_squad_milp
from fpl_forecast.decision.prices import selling_price_tenths, validate_budget
from fpl_forecast.decision.rules import default_rules, validate_rules
from fpl_forecast.decision.runner import forecast_decisions_guard
from fpl_forecast.decision.squad import optimize_initial_squad
from fpl_forecast.decision.transfers import computed_selling_price, plan_multi_gameweek_transfers


def test_default_rules_match_official_squad_shape() -> None:
    rules = default_rules()

    assert validate_rules(rules) == []
    assert rules.squad_size == 15
    assert rules.position_quotas == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert rules.min_starters == {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
    assert rules.max_players_per_team == 3
    assert rules.budget_tenths == 1000


def test_price_math_uses_integer_tenths_and_sell_on_fee() -> None:
    rules = default_rules()

    assert selling_price_tenths(55, 57, rules) == 56
    assert selling_price_tenths(55, 58, rules) == 56
    assert selling_price_tenths(55, 54, rules) == 54
    assert computed_selling_price(70, 75, rules) == 72
    validate_budget(999, rules.budget_tenths)
    with pytest.raises(ValueError, match="exceeds budget"):
        validate_budget(1001, rules.budget_tenths)


def test_lineup_captain_vice_and_autosubs_are_rule_legal() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad.loc[squad["player_uid"].eq("MID_0"), "expected_points"] = 9.0
    squad.loc[squad["player_uid"].eq("MID_1"), "expected_points"] = 8.0
    squad.loc[squad["player_uid"].eq("MID_0"), "actual_minutes"] = 0
    squad.loc[squad["player_uid"].eq("MID_1"), "actual_minutes"] = 90
    squad.loc[squad["player_uid"].eq("DEF_4"), "actual_minutes"] = 90

    decision = optimize_lineup(squad, rules)
    scored = apply_autosubs_and_score(squad, decision, rules)

    assert len(decision.lineup) == 11
    assert decision.captain == "MID_0"
    assert decision.vice_captain == "MID_1"
    assert scored["captain_multiplier_player"] == "MID_1"
    assert "MID_0" not in scored["active_players"]
    assert len(scored["active_players"]) == 11


def test_initial_squad_optimizer_matches_bruteforce_on_small_universe() -> None:
    rules = default_rules()
    candidates = pd.concat(
        [
            _squad_frame(),
            pd.DataFrame(
                [
                    {
                        "player_uid": "MID_alt",
                        "player_name": "Alternative Mid",
                        "fpl_position": "MID",
                        "player_team_uid": "team_8",
                        "price_tenths": 45,
                        "expected_points": 7.7,
                        "p_appearance": 1.0,
                        "actual_points": 4.0,
                        "actual_minutes": 90,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    solution = optimize_initial_squad(candidates, rules, candidate_limits={"GKP": 2, "DEF": 5, "MID": 6, "FWD": 3})
    expected = _bruteforce_best_objective(candidates, rules)

    assert solution.solver_status == "optimal"
    assert round(solution.objective, 8) == round(expected, 8)
    assert "MID_alt" in solution.squad


def test_milp_optimizer_matches_bruteforce_on_small_universe() -> None:
    rules = default_rules()
    candidates = _small_extra_candidate_frame()

    solution = optimize_squad_milp(candidates, rules)
    expected = _bruteforce_milp_objective(candidates, rules)

    assert solution.solver_name == "scipy_highs_milp"
    assert solution.solver_status == "optimal"
    assert solution.objective_gap == 0
    assert round(solution.objective, 8) == round(expected, 8)
    assert solution.objective_bound == pytest.approx(solution.objective)


def test_milp_selection_stable_under_small_forecast_perturbations() -> None:
    rules = default_rules()
    candidates = _small_extra_candidate_frame()
    solution = optimize_squad_milp(candidates, rules)
    perturbed = candidates.copy()
    perturbed["expected_points"] = perturbed["expected_points"] + 1e-6

    perturbed_solution = optimize_squad_milp(perturbed, rules)

    assert set(perturbed_solution.squad) == set(solution.squad)
    assert set(perturbed_solution.lineup_decision.lineup) == set(solution.lineup_decision.lineup)


def test_expected_realized_evaluator_matches_hand_captain_and_vice_case() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad["p_appearance"] = 1.0
    squad.loc[squad["player_uid"].eq("MID_4"), ["expected_points", "p_appearance"]] = [10.0, 0.5]
    squad.loc[squad["player_uid"].eq("MID_4"), "expected_points_given_appearance"] = 20.0
    squad.loc[squad["player_uid"].eq("MID_3"), ["expected_points", "p_appearance"]] = [7.0, 1.0]
    squad.loc[squad["player_uid"].eq("MID_3"), "expected_points_given_appearance"] = 7.0
    lineup = (
        "GKP_0",
        "DEF_0",
        "DEF_1",
        "DEF_2",
        "DEF_3",
        "MID_0",
        "MID_1",
        "MID_2",
        "MID_3",
        "MID_4",
        "FWD_0",
    )
    bench = ("GKP_1", "DEF_4", "FWD_1", "FWD_2")
    decision = optimize_lineup(squad, rules, appearance_aware=False)
    decision = decision.__class__(
        lineup=lineup,
        captain="MID_4",
        vice_captain="MID_3",
        bench=bench,
        objective=0.0,
        formation="3-5-2",
        method="hand",
    )

    result = evaluate_expected_realized_points(squad, decision, rules, max_scenarios=None)

    assert result.expected_captain_bonus == pytest.approx(10.0)
    assert result.expected_vice_captain_contingency == pytest.approx(3.5)
    assert result.probability_all_starters_appear == pytest.approx(0.5)
    assert result.scenario_count == 32768
    assert result.probability_mass == pytest.approx(1.0, abs=1e-12)
    assert result.analytic_method == "exact_32768_state_independent_appearance_enumeration"


def test_expected_realized_uses_unconditional_points_once() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad["p_appearance"] = 1.0
    squad.loc[squad["player_uid"].eq("MID_4"), ["expected_points", "p_appearance"]] = [5.0, 0.5]
    squad.loc[squad["player_uid"].eq("MID_4"), "expected_points_given_appearance"] = 10.0
    squad.loc[squad["player_uid"].eq("DEF_3"), "expected_points"] = 3.0
    squad.loc[squad["player_uid"].eq("DEF_3"), "expected_points_given_appearance"] = 3.0
    lineup = (
        "GKP_0",
        "DEF_0",
        "DEF_1",
        "DEF_2",
        "MID_0",
        "MID_1",
        "MID_2",
        "MID_3",
        "MID_4",
        "FWD_0",
        "FWD_1",
    )
    decision = optimize_lineup(squad, rules, appearance_aware=False).__class__(
        lineup=lineup,
        captain="MID_0",
        vice_captain="MID_1",
        bench=("GKP_1", "DEF_3", "DEF_4", "FWD_2"),
        objective=0.0,
        formation="3-5-2",
        method="hand",
    )

    result = evaluate_expected_realized_points(squad, decision, rules, max_scenarios=1, seed=1)
    unconditional_sum = float(squad.set_index("player_uid").loc[list(lineup), "expected_points"].sum())

    assert result.nominal_starting_xi_xpoints == pytest.approx(unconditional_sum)
    assert result.expected_active_starter_points == pytest.approx(unconditional_sum)
    assert result.expected_nominal_starting_xi_points == pytest.approx(unconditional_sum)
    assert result.expected_autosub_contribution == pytest.approx(1.5)
    assert result.expected_realized_total == pytest.approx(
        result.expected_active_starter_points
        + result.expected_autosub_contribution
        + result.expected_captain_bonus
        + result.expected_vice_captain_contingency
    )
    assert result.scenario_count == 32768


def test_expected_realized_uses_direct_conditional_not_rounded_public_xpoints() -> None:
    rules = default_rules()
    squad = _squad_frame()
    player = squad["player_uid"].eq("GKP_0")
    squad.loc[player, "p_appearance"] = 0.01
    squad.loc[player, "expected_points"] = 0.05
    squad.loc[player, "expected_points_given_appearance"] = 2.0
    lineup = (
        "GKP_0",
        "DEF_0",
        "DEF_1",
        "DEF_2",
        "MID_0",
        "MID_1",
        "MID_2",
        "MID_3",
        "MID_4",
        "FWD_0",
        "FWD_1",
    )
    decision = optimize_lineup(squad, rules, appearance_aware=False).__class__(
        lineup=lineup,
        captain="MID_0",
        vice_captain="MID_1",
        bench=("GKP_1", "DEF_3", "DEF_4", "FWD_2"),
        objective=0.0,
        formation="3-5-2",
        method="hand",
    )

    first = evaluate_expected_realized_points(squad, decision, rules)
    squad.loc[player, "expected_points"] = 0.01
    second = evaluate_expected_realized_points(squad, decision, rules)

    assert first == second
    expected_nominal = sum(
        float(row.p_appearance * row.expected_points_given_appearance)
        for row in squad.loc[squad["player_uid"].isin(lineup)].itertuples(index=False)
    )
    assert first.nominal_starting_xi_xpoints == pytest.approx(expected_nominal)


def test_goalkeeper_order_uses_stabilized_conditional_points() -> None:
    rules = default_rules()
    squad = _squad_frame()
    rare = squad["player_uid"].eq("GKP_0")
    reliable = squad["player_uid"].eq("GKP_1")
    squad.loc[rare, ["expected_points", "p_appearance", "expected_points_given_appearance"]] = [
        0.05,
        0.01,
        2.0,
    ]
    squad.loc[reliable, ["expected_points", "p_appearance", "expected_points_given_appearance"]] = [
        3.6,
        0.9,
        4.0,
    ]

    base = optimize_lineup(squad, rules, appearance_aware=False)
    rare_starts = base.__class__(
        lineup=tuple("GKP_0" if player == "GKP_1" else player for player in base.lineup),
        captain=base.captain,
        vice_captain=base.vice_captain,
        bench=("GKP_1", *base.bench[1:]),
        objective=0.0,
        formation=base.formation,
        method="hand",
    )
    reliable_starts = rare_starts.__class__(
        lineup=tuple("GKP_1" if player == "GKP_0" else player for player in rare_starts.lineup),
        captain=rare_starts.captain,
        vice_captain=rare_starts.vice_captain,
        bench=("GKP_0", *rare_starts.bench[1:]),
        objective=0.0,
        formation=rare_starts.formation,
        method="hand",
    )

    rare_result = evaluate_expected_realized_points(squad, rare_starts, rules)
    reliable_result = evaluate_expected_realized_points(squad, reliable_starts, rules)
    repeated = evaluate_expected_realized_points(squad, reliable_starts, rules)

    assert reliable_result.expected_realized_total > rare_result.expected_realized_total
    assert repeated == reliable_result


def test_expected_realized_captain_and_vice_both_may_be_absent() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad["p_appearance"] = 1.0
    squad.loc[squad["player_uid"].eq("MID_4"), ["expected_points", "p_appearance"]] = [10.0, 0.5]
    squad.loc[squad["player_uid"].eq("MID_4"), "expected_points_given_appearance"] = 20.0
    squad.loc[squad["player_uid"].eq("MID_3"), ["expected_points", "p_appearance"]] = [2.0, 0.25]
    squad.loc[squad["player_uid"].eq("MID_3"), "expected_points_given_appearance"] = 8.0
    lineup = (
        "GKP_0",
        "DEF_0",
        "DEF_1",
        "DEF_2",
        "MID_0",
        "MID_1",
        "MID_2",
        "MID_3",
        "MID_4",
        "FWD_0",
        "FWD_1",
    )
    decision = optimize_lineup(squad, rules, appearance_aware=False).__class__(
        lineup=lineup,
        captain="MID_4",
        vice_captain="MID_3",
        bench=("GKP_1", "DEF_3", "DEF_4", "FWD_2"),
        objective=0.0,
        formation="3-5-2",
        method="hand",
    )

    result = evaluate_expected_realized_points(squad, decision, rules)

    assert result.expected_captain_bonus == pytest.approx(10.0)
    assert result.expected_vice_captain_contingency == pytest.approx(1.0)


def test_expected_realized_autosub_rules_goalkeeper_and_formation() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad["p_appearance"] = 1.0
    squad.loc[squad["player_uid"].eq("GKP_0"), "p_appearance"] = 0.0
    squad.loc[squad["player_uid"].eq("GKP_1"), "expected_points"] = 4.0
    squad.loc[squad["player_uid"].eq("GKP_1"), "expected_points_given_appearance"] = 4.0
    squad.loc[squad["player_uid"].eq("DEF_4"), "p_appearance"] = 0.0
    lineup = ("GKP_0", "DEF_0", "DEF_1", "DEF_2", "DEF_3", "DEF_4", "MID_0", "MID_1", "MID_2", "MID_3", "FWD_0")
    bench = ("GKP_1", "MID_4", "FWD_1", "FWD_2")
    decision = optimize_lineup(squad, rules, appearance_aware=False).__class__(
        lineup=lineup,
        captain="MID_0",
        vice_captain="MID_1",
        bench=bench,
        objective=0.0,
        formation="5-4-1",
        method="hand",
    )

    result = evaluate_expected_realized_points(squad, decision, rules, max_scenarios=None)

    assert result.expected_automatic_substitutions == pytest.approx(2.0)
    assert result.probability_unreplaced_starter == pytest.approx(0.0)
    assert result.expected_autosub_contribution >= 4.0


def test_expected_realized_leaves_starter_unreplaced_when_no_legal_sub_appears() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad["p_appearance"] = 1.0
    squad.loc[squad["player_uid"].isin(["DEF_2", "DEF_3", "DEF_4"]), "p_appearance"] = 0.0
    lineup = (
        "GKP_0",
        "DEF_0",
        "DEF_1",
        "DEF_2",
        "MID_0",
        "MID_1",
        "MID_2",
        "MID_3",
        "MID_4",
        "FWD_0",
        "FWD_1",
    )
    decision = optimize_lineup(squad, rules, appearance_aware=False).__class__(
        lineup=lineup,
        captain="MID_0",
        vice_captain="MID_1",
        bench=("GKP_1", "DEF_3", "DEF_4", "FWD_2"),
        objective=0.0,
        formation="3-5-2",
        method="hand",
    )

    result = evaluate_expected_realized_points(squad, decision, rules)

    assert result.probability_unreplaced_starter == pytest.approx(1.0)
    assert result.expected_automatic_substitutions == pytest.approx(0.0)


def test_bench_order_materially_changes_expected_realized_points() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad["p_appearance"] = 1.0
    squad.loc[squad["player_uid"].eq("MID_0"), "p_appearance"] = 0.0
    squad.loc[squad["player_uid"].eq("FWD_2"), ["expected_points", "p_appearance"]] = [8.0, 1.0]
    squad.loc[squad["player_uid"].eq("FWD_2"), "expected_points_given_appearance"] = 8.0
    squad.loc[squad["player_uid"].eq("DEF_3"), ["expected_points", "p_appearance"]] = [1.0, 1.0]
    squad.loc[squad["player_uid"].eq("DEF_3"), "expected_points_given_appearance"] = 1.0
    lineup = ("GKP_0", "DEF_0", "DEF_1", "DEF_2", "MID_0", "MID_1", "MID_2", "MID_3", "MID_4", "FWD_0", "FWD_1")
    good = optimize_lineup(squad, rules, appearance_aware=False).__class__(
        lineup=lineup,
        captain="MID_1",
        vice_captain="MID_1",
        bench=("GKP_1", "FWD_2", "DEF_3", "DEF_4"),
        objective=0.0,
        formation="3-5-2",
        method="hand",
    )
    good = good.__class__(
        lineup=good.lineup,
        captain="MID_1",
        vice_captain="MID_2",
        bench=good.bench,
        objective=good.objective,
        formation=good.formation,
        method=good.method,
    )
    bad = good.__class__(
        lineup=good.lineup,
        captain=good.captain,
        vice_captain=good.vice_captain,
        bench=("GKP_1", "DEF_3", "FWD_2", "DEF_4"),
        objective=0.0,
        formation=good.formation,
        method=good.method,
    )

    assert evaluate_expected_realized_points(squad, good, rules).expected_realized_total > evaluate_expected_realized_points(
        squad,
        bad,
        rules,
    ).expected_realized_total


def test_expected_realized_lineup_selects_better_vice_and_order() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad.loc[squad["player_uid"].eq("MID_4"), ["expected_points", "p_appearance"]] = [10.0, 0.2]
    squad.loc[squad["player_uid"].eq("MID_4"), "expected_points_given_appearance"] = 50.0
    squad.loc[squad["player_uid"].eq("MID_3"), ["expected_points", "p_appearance"]] = [7.0, 1.0]
    squad.loc[squad["player_uid"].eq("MID_3"), "expected_points_given_appearance"] = 7.0

    decision, breakdown = optimize_lineup_expected_realized(squad, rules, max_scenarios=512, shortlist=200)

    assert decision.captain != decision.vice_captain
    assert breakdown.expected_realized_total == pytest.approx(decision.objective)
    assert breakdown.expected_vice_captain_contingency > 0


def test_expected_realized_optimizer_prefers_stronger_bench_when_it_improves_realized_score() -> None:
    rules = default_rules()
    candidates = _small_extra_candidate_frame()
    candidates["p_appearance"] = 1.0
    candidates.loc[candidates["player_uid"].eq("MID_4"), "p_appearance"] = 0.05
    candidates.loc[candidates["player_uid"].eq("MID_4"), "expected_points_given_appearance"] = (
        candidates.loc[candidates["player_uid"].eq("MID_4"), "expected_points"] / 0.05
    )
    bench_mid = pd.DataFrame(
        [
            {
                "player_uid": "MID_reliable_bench",
                "player_name": "Reliable Bench Mid",
                "fpl_position": "MID",
                "player_team_uid": "team_reliable",
                "price_tenths": 45,
                "expected_points": 4.0,
                "expected_points_given_appearance": 4.0,
                "p_appearance": 1.0,
                "actual_points": 4.0,
                "actual_minutes": 90,
            }
        ]
    )
    candidates = pd.concat([candidates, bench_mid], ignore_index=True)

    d1 = optimize_squad_milp(candidates, rules)
    d2 = optimize_squad_expected_realized(candidates, rules, search_limit=30, max_scenarios=512)
    indexed = candidates.set_index("player_uid", drop=False)
    d1_eval = evaluate_expected_realized_points(indexed.loc[list(d1.squad)].reset_index(drop=True), d1.lineup_decision, rules)

    assert d2.objective >= d1_eval.expected_realized_total
    assert (d2.diagnostics or {})["expected_bench_points_used"] >= d1_eval.expected_bench_points_used
    assert (d2.diagnostics or {})["d1_expected_realized_total"] == pytest.approx(d1_eval.expected_realized_total)


def test_expected_realized_optimizer_allows_unused_bank_and_does_not_reward_spend() -> None:
    rules = default_rules()
    candidates = _small_extra_candidate_frame()
    expensive = candidates.copy()
    expensive.loc[expensive["player_uid"].eq("MID_upgrade_0"), "price_tenths"] += 50

    cheap = optimize_squad_expected_realized(candidates, rules, search_limit=5, max_scenarios=256)
    costly = optimize_squad_expected_realized(expensive, rules, search_limit=5, max_scenarios=256)

    assert cheap.objective == pytest.approx(costly.objective)
    assert cheap.bank_tenths >= costly.bank_tenths


def test_expected_realized_optimizer_is_deterministic_and_does_not_blacklist_cheap_players() -> None:
    rules = default_rules()
    cheap = pd.DataFrame(
        [
            {
                "player_uid": "cheap_low_projection_def",
                "player_name": "Cheap Low Projection Def",
                "fpl_position": "DEF",
                "player_team_uid": "team_cheap",
                "price_tenths": 40,
                "expected_points": 0.1,
                "expected_points_given_appearance": 0.5,
                "p_appearance": 0.2,
                "actual_points": 0.0,
                "actual_minutes": 0,
            }
        ]
    )
    candidates = pd.concat([_small_extra_candidate_frame(), cheap], ignore_index=True)

    first = optimize_squad_expected_realized(candidates, rules, search_limit=20, max_scenarios=256)
    second = optimize_squad_expected_realized(candidates, rules, search_limit=20, max_scenarios=256)

    assert first.squad == second.squad
    assert first.lineup_decision == second.lineup_decision
    assert "cheap_low_projection_def" in set(candidates["player_uid"])


def test_expected_realized_search_covers_full_legal_one_swap_pool() -> None:
    rules = default_rules()
    candidates = _small_extra_candidate_frame()
    solution = optimize_squad_expected_realized(candidates, rules, search_iterations=1)
    diagnostics = solution.diagnostics or {}
    unselected = len(candidates) - rules.squad_size
    raw = rules.squad_size * unselected

    assert diagnostics["eligible_players_full_pool"] == len(candidates)
    assert diagnostics["selected_players_in_seed"] == rules.squad_size
    assert diagnostics["unselected_replacements_considered"] == unselected
    assert diagnostics["raw_swap_proposals_generated"] == raw
    assert (
        diagnostics["proposals_rejected_position_mismatch"]
        + diagnostics["proposals_rejected_budget"]
        + diagnostics["proposals_rejected_club_limit"]
        + diagnostics["other_illegal_proposals_rejected"]
        + diagnostics["feasible_unique_squad_proposals"]
        == raw
    )
    assert diagnostics["final_unique_squads_evaluated"] == 1 + diagnostics["feasible_unique_squad_proposals"]
    assert diagnostics["squad_evaluation_calls"] == 1 + diagnostics["feasible_unique_squad_proposals"]
    assert solution.evaluated_squads == diagnostics["final_unique_squads_evaluated"]
    assert solution.objective + 1e-9 >= diagnostics["d1_expected_realized_total"]
    assert diagnostics["termination_reason"] in {"one_swap_local_optimum", "configured_iteration_bound_reached"}


def test_multi_gameweek_transfer_planner_tracks_bank_rollover_hits_and_lineups() -> None:
    rules = default_rules()
    current = _squad_frame()
    all_candidates = pd.concat(
        [
            current,
            _upgrade_rows("DEF", 1, 48, 8.0),
            _upgrade_rows("FWD", 1, 52, 10.0),
        ],
        ignore_index=True,
    )
    weekly = {
        1: all_candidates.copy(),
        2: all_candidates.copy(),
    }

    plan = plan_multi_gameweek_transfers(
        current,
        weekly,
        rules,
        bank_tenths=10,
        free_transfers=1,
        max_transfers_per_gameweek=1,
    )

    assert plan.solver_status == "exhaustive_optimal"
    assert plan.objective_gap == 0
    assert len(plan.actions) == 2
    assert all(len(action.lineup) == 11 for action in plan.actions)
    assert all(action.captain != action.vice_captain for action in plan.actions)
    assert plan.actions[0].free_transfers_after >= 1
    assert all(action.bank_after >= 0 for action in plan.actions)


def test_frozen_decision_artifact_rejects_future_or_target_columns() -> None:
    allowed = pd.DataFrame({"player_uid": ["p1"], "expected_points": [3.2]})
    assert_frozen_decisions_target_free(allowed, ("total_points", "actual_minutes"))

    forbidden = pd.DataFrame({"player_uid": ["p1"], "actual_minutes": [90]})
    with pytest.raises(ValueError, match="target/future"):
        assert_frozen_decisions_target_free(forbidden, ("total_points", "actual_minutes"))


def test_forecast_decisions_guard_fails_before_writing_when_current_data_invalid(monkeypatch) -> None:
    def fail_current_fixture_load(*, season: str, as_of: str):
        raise ValueError(f"Current fixture season mismatch: requested {season}, inferred 2025-26.")

    monkeypatch.setattr("fpl_forecast.decision.runner.load_current_fixture_frame", fail_current_fixture_load)

    with pytest.raises(ValueError, match="season mismatch"):
        forecast_decisions_guard(season="2026-27", gameweek=1, as_of="2026-07-22T00:00:00Z")


def _squad_frame() -> pd.DataFrame:
    rows = []
    specs = {"GKP": (2, 40, 3.0), "DEF": (5, 45, 4.0), "MID": (5, 55, 5.0), "FWD": (3, 50, 4.5)}
    team_index = 0
    for position, (count, price, base_points) in specs.items():
        for index in range(count):
            rows.append(
                {
                    "player_uid": f"{position}_{index}",
                    "player_name": f"{position} {index}",
                    "fpl_position": position,
                    "player_team_uid": f"team_{team_index % 8}",
                    "price_tenths": price,
                    "expected_points": base_points + index / 10,
                    "expected_points_given_appearance": base_points + index / 10,
                    "p_appearance": 1.0,
                    "actual_points": float(index + 1),
                    "actual_minutes": 90,
                }
            )
            team_index += 1
    return pd.DataFrame(rows)


def _small_extra_candidate_frame() -> pd.DataFrame:
    return pd.concat(
        [
            _squad_frame(),
            _upgrade_rows("MID", 1, 45, 7.7),
            _upgrade_rows("DEF", 1, 44, 6.2),
        ],
        ignore_index=True,
    )


def _upgrade_rows(position: str, count: int, price: int, expected_points: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_uid": f"{position}_upgrade_{index}",
                "player_name": f"{position} Upgrade {index}",
                "fpl_position": position,
                "player_team_uid": f"team_upgrade_{position}_{index}",
                "price_tenths": price,
                "expected_points": expected_points + index,
                "expected_points_given_appearance": expected_points + index,
                "p_appearance": 1.0,
                "actual_points": expected_points,
                "actual_minutes": 90,
            }
            for index in range(count)
        ]
    )


def _bruteforce_best_objective(candidates: pd.DataFrame, rules) -> float:
    indexed = candidates.set_index("player_uid")
    by_position = {
        position: list(candidates.loc[candidates["fpl_position"].eq(position), "player_uid"])
        for position in rules.position_quotas
    }
    best = float("-inf")
    for gkp in combinations(by_position["GKP"], 2):
        for defenders in combinations(by_position["DEF"], 5):
            for mids in combinations(by_position["MID"], 5):
                for forwards in combinations(by_position["FWD"], 3):
                    selected = (*gkp, *defenders, *mids, *forwards)
                    squad = indexed.loc[list(selected)].reset_index()
                    if int(squad["price_tenths"].sum()) > rules.budget_tenths:
                        continue
                    if squad["player_team_uid"].value_counts().max() > rules.max_players_per_team:
                        continue
                    best = max(best, optimize_lineup(squad, rules).objective)
    return best


def _bruteforce_milp_objective(candidates: pd.DataFrame, rules) -> float:
    indexed = candidates.set_index("player_uid")
    by_position = {
        position: list(candidates.loc[candidates["fpl_position"].eq(position), "player_uid"])
        for position in rules.position_quotas
    }
    best = float("-inf")
    for gkp in combinations(by_position["GKP"], 2):
        for defenders in combinations(by_position["DEF"], 5):
            for mids in combinations(by_position["MID"], 5):
                for forwards in combinations(by_position["FWD"], 3):
                    selected = (*gkp, *defenders, *mids, *forwards)
                    squad = indexed.loc[list(selected)].reset_index()
                    if int(squad["price_tenths"].sum()) > rules.budget_tenths:
                        continue
                    if squad["player_team_uid"].value_counts().max() > rules.max_players_per_team:
                        continue
                    for lineup_ids in combinations(selected, 11):
                        lineup = indexed.loc[list(lineup_ids)].reset_index()
                        counts = lineup["fpl_position"].value_counts().to_dict()
                        if any(counts.get(pos, 0) < minimum for pos, minimum in rules.min_starters.items()):
                            continue
                        if any(counts.get(pos, 0) > maximum for pos, maximum in rules.max_starters.items()):
                            continue
                        starter_points = float(lineup["expected_points"].sum())
                        for captain in lineup_ids:
                            row = indexed.loc[captain]
                            best = max(best, starter_points + float(row["expected_points"] * row["p_appearance"]))
    return best
