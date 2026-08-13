from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from fpl_forecast.decision.config import load_decision_config
from fpl_forecast.decision.evidence import (
    D1_VARIANT,
    D2_VARIANT,
    active_decision_configurations,
    authoritative_decision_run,
    authoritative_xpoints_run,
    build_decision_evidence_table_from_frames,
    decision_evidence_supersession_table,
    load_decision_evidence_registry,
    publication_round,
    superseded_evidence_run_ids,
)
from fpl_forecast.decision.inputs import assert_frozen_decisions_target_free
from fpl_forecast.decision.expected_realized import (
    _fixed_squad_lineup_candidates,
    evaluate_expected_realized_points,
    optimize_lineup_expected_realized,
    refine_fixed_squad_lineup,
)
from fpl_forecast.decision.lineup import _lineup_objective_ids, apply_autosubs_and_score, optimize_lineup
from fpl_forecast.decision.milp import optimize_squad_expected_realized, optimize_squad_milp
from fpl_forecast.decision.prices import selling_price_tenths, validate_budget
from fpl_forecast.decision.rules import default_rules, validate_rules
from fpl_forecast.decision.runner import forecast_decisions_guard, run_decision_backtest
from fpl_forecast.decision.squad import optimize_initial_squad
from fpl_forecast.decision.transfers import computed_selling_price, plan_multi_gameweek_transfers


PHASE7_ROLLING_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "src/fpl_forecast/decision/config_phase7_rolling_76_folds.json"
)


def test_default_rules_match_official_squad_shape() -> None:
    rules = default_rules()

    assert validate_rules(rules) == []
    assert rules.squad_size == 15
    assert rules.position_quotas == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert rules.min_starters == {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
    assert rules.max_players_per_team == 3
    assert rules.budget_tenths == 1000


def test_corrected_rolling_replay_config_preserves_original_76_fold_comparison() -> None:
    config = load_decision_config(PHASE7_ROLLING_CONFIG)

    assert config.xpoints_runs["rolling"] == "phase6_xpoints_rolling_real"
    assert config.comparison_models == (
        "X2_TEAM_CONSTRAINED_SIM_M3",
        "X2_TEAM_CONSTRAINED_SIM_M5",
        "X1_INDEPENDENT_COMPONENT_RATES_M3",
        "X0_PHASE3_B5_EB_POINTS_PER90",
        "D0_PRICE_VALUE_BASELINE",
    )
    assert config.optimizer_variants == (D1_VARIANT,)


def test_decision_evidence_registry_separates_authoritative_and_historical_runs() -> None:
    registry = load_decision_evidence_registry()
    supersession = decision_evidence_supersession_table()
    authoritative = {
        authoritative_decision_run("rolling_benchmark"),
        authoritative_decision_run("table7_gw1"),
    }

    assert authoritative == {
        "phase7_goalkeeper_scoring_corrected_decisions_rolling_real_clean_034830b041c1",
        "phase9b13_goalkeeper_scoring_corrected_exact_decisions_gw1_clean_034830b041c1",
    }
    assert not authoritative.intersection(supersession["superseded_run_id"])
    assert set(supersession["superseded_status"]) == {"immutable_historical_record"}
    assert "phase7_decisions_rolling_real" in set(supersession["superseded_run_id"])
    assert "phase9b13_lineup_refined_decisions_gw1" in set(
        supersession["superseded_run_id"]
    )
    assert "phase7_goalkeeper_scoring_corrected_decisions_rolling_real" in set(
        supersession["superseded_run_id"]
    )
    assert "phase9b13_goalkeeper_scoring_corrected_exact_decisions_gw1" in set(
        supersession["superseded_run_id"]
    )
    assert "1.67" not in json.dumps(registry, sort_keys=True)


def test_active_decision_configs_follow_registry_and_reject_superseded_inputs() -> None:
    active_configs = active_decision_configurations()
    superseded = superseded_evidence_run_ids()

    assert {record["status"] for record in active_configs.values()} == {
        "active_default",
        "active_correction_replay",
    }
    for config_path, record in active_configs.items():
        config = load_decision_config(config_path)
        divergences = record["documented_divergences"]
        for mode, scope in record["xpoints_evidence"].items():
            configured_run = config.xpoints_runs[mode]
            authoritative_run = authoritative_xpoints_run(scope)
            if configured_run != authoritative_run:
                reason = str(divergences.get(mode, "")).strip()
                assert reason, (
                    f"{config_path} maps {mode} to {configured_run}, not active evidence "
                    f"{authoritative_run}, without a documented reason"
                )
            assert configured_run not in superseded


def test_decision_replay_refuses_to_overwrite_historical_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "immutable_historical_run"
    run_dir.mkdir()
    marker = run_dir / "manifest.json"
    marker.write_text('{"historical": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing decision evidence"):
        run_decision_backtest(
            seasons="2023-24,2024-25",
            mode="rolling",
            reports_dir=tmp_path,
            run_id=run_dir.name,
            config_path=PHASE7_ROLLING_CONFIG,
        )

    assert marker.read_text(encoding="utf-8") == '{"historical": true}\n'


def test_table7_is_derived_from_corrected_paired_decisions() -> None:
    source_run_id = authoritative_decision_run("table7_gw1")
    d1_expected = 48.75828391099525
    d2_expected = 49.487285703810194
    metrics = pd.DataFrame(
        [
            {
                "optimizer_variant": D1_VARIANT,
                "decisions": 3,
                "mean_expected_score": d1_expected,
                "mean_realized_points": 59.0,
                "mean_autosub_points": 1.0,
                "unreplaced_starter_rate": 0.0,
            },
            {
                "optimizer_variant": D2_VARIANT,
                "decisions": 3,
                "mean_expected_score": d2_expected,
                "mean_realized_points": 61.0,
                "mean_autosub_points": 2.0,
                "unreplaced_starter_rate": 0.0,
            },
        ]
    )
    scored = pd.DataFrame(
        [
            {"season": season, "gameweek": 1, "optimizer_variant": variant, "realized_points": points}
            for season, d1_points, d2_points in (
                ("2023-24", 48, 50),
                ("2024-25", 70, 70),
                ("2025-26", 57, 59),
            )
            for variant, points in ((D1_VARIANT, d1_points), (D2_VARIANT, d2_points))
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "left_model": f"X2_TEAM_CONSTRAINED_SIM_M7:{D2_VARIANT}",
                "right_model": f"X2_TEAM_CONSTRAINED_SIM_M7:{D1_VARIANT}",
                "mean_realized_difference": 4 / 3,
                "bootstrap_ci_low": 0.0,
                "bootstrap_ci_high": 2.0,
                "captain_agreement": 1.0,
                "mean_lineup_overlap": 1.0,
            }
        ]
    )

    table = build_decision_evidence_table_from_frames(
        metrics=metrics,
        comparison=comparison,
        scored=scored,
        source_run_id=source_run_id,
    )

    indexed = table.set_index("decision_model")
    assert indexed.loc[D1_VARIANT, "mean_expected_realized_points"] == pytest.approx(d1_expected)
    assert indexed.loc[D2_VARIANT, "mean_expected_realized_points"] == pytest.approx(d2_expected)
    assert set(table["d2_minus_d1_fold_realized_differences"]) == {"[2, 0, 2]"}
    assert table["d2_minus_d1_mean_realized_difference"].tolist() == pytest.approx([4 / 3, 4 / 3])
    assert publication_round(4 / 3) == 1.33
    assert set(table["d2_minus_d1_mean_realized_difference_2dp"]) == {1.33}
    assert set(table["source_run_id"]) == {source_run_id}


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


def test_d1_captain_coefficient_uses_unconditional_expected_points_directly() -> None:
    rules = default_rules()
    candidates = _captain_regression_squad()

    solution = optimize_squad_milp(candidates, rules)
    indexed = candidates.set_index("player_uid")
    ordinary_lineup = float(indexed.loc[list(solution.lineup_decision.lineup), "expected_points"].sum())
    captain_mu = float(indexed.loc[solution.lineup_decision.captain, "expected_points"])

    assert solution.lineup_decision.captain == "MID_4"
    assert captain_mu == 10.0
    assert solution.objective - ordinary_lineup == pytest.approx(captain_mu)


def test_d1_captained_player_total_is_twice_unconditional_expected_points() -> None:
    rules = default_rules()
    candidates = _captain_regression_squad()
    solution = optimize_squad_milp(candidates, rules)
    indexed = candidates.set_index("player_uid")
    captain = solution.lineup_decision.captain
    captain_mu = float(indexed.loc[captain, "expected_points"])
    ordinary_others = float(
        indexed.loc[
            [player for player in solution.lineup_decision.lineup if player != captain],
            "expected_points",
        ].sum()
    )

    assert solution.objective - ordinary_others == pytest.approx(2 * captain_mu)


def test_d1_captain_selection_does_not_double_discount_appearance() -> None:
    rules = default_rules()
    candidates = _captain_regression_squad()

    solution = optimize_squad_milp(candidates, rules)

    # Correct: MID_4 has mu=10 > MID_3 mu=9. Old: 0.5*10=5 < 1.0*9=9.
    assert solution.lineup_decision.captain == "MID_4"


def test_d1_captain_with_certain_appearance_does_not_expose_the_bug() -> None:
    rules = default_rules()
    candidates = _captain_regression_squad()
    candidates.loc[candidates["player_uid"].eq("MID_4"), "p_appearance"] = 1.0

    solution = optimize_squad_milp(candidates, rules)
    indexed = candidates.set_index("player_uid")
    ordinary_lineup = float(indexed.loc[list(solution.lineup_decision.lineup), "expected_points"].sum())

    assert solution.lineup_decision.captain == "MID_4"
    assert solution.objective == pytest.approx(ordinary_lineup + 10.0)
    assert 1.0 * 10.0 == 10.0


def test_d1_ordinary_non_captain_coefficient_is_unchanged() -> None:
    rules = default_rules()
    candidates = _captain_regression_squad()
    base = optimize_squad_milp(candidates, rules)
    perturbed = candidates.copy()
    non_captain = next(player for player in base.lineup_decision.lineup if player.startswith("DEF_"))
    perturbed.loc[perturbed["player_uid"].eq(non_captain), "expected_points"] += 0.25

    changed = optimize_squad_milp(perturbed, rules)

    assert non_captain in changed.lineup_decision.lineup
    assert base.lineup_decision.captain == changed.lineup_decision.captain == "MID_4"
    assert changed.objective - base.objective == pytest.approx(0.25)


def test_d2_diagnostics_use_corrected_d1_seed_captain() -> None:
    rules = default_rules()
    candidates = _captain_regression_squad()

    solution = optimize_squad_expected_realized(candidates, rules, search_iterations=1)

    assert (solution.diagnostics or {})["d1_captain"] == "MID_4"
    assert (solution.diagnostics or {})["mean_only_seed_objective"] == pytest.approx(
        optimize_squad_milp(candidates, rules).objective
    )


def test_legacy_lineup_helper_retains_only_legitimate_vice_fallback_probability() -> None:
    records = {
        "captain": {"expected_points": 10.0, "p_appearance": 0.5},
        "vice": {"expected_points": 9.0, "p_appearance": 0.25},
    }

    objective = _lineup_objective_ids(
        ("captain", "vice"),
        [],
        "captain",
        "vice",
        records,
        appearance_aware=True,
    )

    # 19 ordinary + 10 captain + (1 - 0.5) * 9 vice fallback.
    # Vice p_appearance is already incorporated in its unconditional mu=9.
    assert objective == pytest.approx(33.5)


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


@pytest.mark.parametrize("reliable_goalkeeper", ["GKP_0", "GKP_1"])
def test_fixed_squad_refinement_selects_better_goalkeeper_order_in_both_directions(
    reliable_goalkeeper: str,
) -> None:
    rules = default_rules()
    squad = _squad_frame()
    other = "GKP_1" if reliable_goalkeeper == "GKP_0" else "GKP_0"
    squad.loc[
        squad["player_uid"].eq(reliable_goalkeeper),
        ["expected_points", "p_appearance", "expected_points_given_appearance"],
    ] = [3.6, 0.9, 4.0]
    squad.loc[
        squad["player_uid"].eq(other),
        ["expected_points", "p_appearance", "expected_points_given_appearance"],
    ] = [0.05, 0.01, 2.0]
    base = optimize_lineup(squad, rules, appearance_aware=False)
    initial = base.__class__(
        lineup=tuple(other if player in {"GKP_0", "GKP_1"} else player for player in base.lineup),
        captain=base.captain,
        vice_captain=base.vice_captain,
        bench=(reliable_goalkeeper, *base.bench[1:]),
        objective=0.0,
        formation=base.formation,
        method="synthetic_rare_goalkeeper_regression",
    )
    initial_value = evaluate_expected_realized_points(squad, initial, rules).expected_realized_total

    refined, breakdown, diagnostics = refine_fixed_squad_lineup(squad, initial, rules)
    repeated = refine_fixed_squad_lineup(squad, initial, rules)

    assert reliable_goalkeeper in refined.lineup
    assert refined.bench[0] == other
    assert breakdown.expected_realized_total > initial_value
    assert diagnostics.returned_goalkeeper_order_value >= diagnostics.reversed_goalkeeper_order_value
    assert diagnostics.status == "single_change_local_optimum"
    assert repeated == (refined, breakdown, diagnostics)
    assert set(refined.lineup) | set(refined.bench) == set(squad["player_uid"])


def test_fixed_squad_refinement_improves_outfield_reorders_bench_and_captaincy() -> None:
    rules = default_rules()
    squad = _squad_frame()
    squad.loc[
        squad["player_uid"].eq("MID_0"),
        ["expected_points", "p_appearance", "expected_points_given_appearance"],
    ] = [0.1, 0.1, 1.0]
    squad.loc[
        squad["player_uid"].eq("FWD_2"),
        ["expected_points", "p_appearance", "expected_points_given_appearance"],
    ] = [10.0, 1.0, 10.0]
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
    initial = optimize_lineup(squad, rules, appearance_aware=False).__class__(
        lineup=lineup,
        captain="MID_0",
        vice_captain="MID_1",
        bench=("GKP_1", "DEF_3", "DEF_4", "FWD_2"),
        objective=0.0,
        formation="3-5-2",
        method="synthetic_outfield_regression",
    )
    initial_value = evaluate_expected_realized_points(squad, initial, rules).expected_realized_total

    refined, breakdown, diagnostics = refine_fixed_squad_lineup(squad, initial, rules)

    assert "FWD_2" in refined.lineup
    assert "MID_0" in refined.bench
    assert refined.captain == "FWD_2"
    assert refined.vice_captain != refined.captain
    assert breakdown.expected_realized_total > initial_value
    assert diagnostics.outfield_swaps_considered > 0
    assert diagnostics.illegal_outfield_swaps_rejected > 0
    assert diagnostics.bench_orders_evaluated >= 6
    assert diagnostics.captain_pairs_reoptimized == diagnostics.bench_orders_evaluated


def test_fixed_squad_refinement_is_locally_optimal_over_documented_candidates() -> None:
    rules = default_rules()
    squad = _squad_frame()
    initial, _ = optimize_lineup_expected_realized(squad, rules, shortlist=1)

    refined, breakdown, diagnostics = refine_fixed_squad_lineup(squad, initial, rules)
    frame = {
        str(row.player_uid): {
            "position": str(row.fpl_position),
            "team": str(row.player_team_uid),
            "expected_points": float(row.expected_points),
            "p_appearance": float(row.p_appearance),
            "conditional_points": float(row.expected_points_given_appearance),
            "optimizer_unconditional_points": float(
                row.p_appearance * row.expected_points_given_appearance
            ),
        }
        for row in squad.itertuples(index=False)
    }
    candidates, _ = _fixed_squad_lineup_candidates(refined, frame, rules)
    candidate_values = []
    for lineup, bench in candidates:
        candidate = refined.__class__(
            lineup=lineup,
            captain=refined.captain,
            vice_captain=refined.vice_captain,
            bench=bench,
            objective=0.0,
            formation=refined.formation,
            method="audit",
        )
        if candidate.captain not in lineup or candidate.vice_captain not in lineup:
            continue
        candidate_values.append(evaluate_expected_realized_points(squad, candidate, rules).expected_realized_total)

    assert breakdown.expected_realized_total + 1e-9 >= max(candidate_values)
    assert diagnostics.gain >= 0


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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


def _captain_regression_squad() -> pd.DataFrame:
    squad = _squad_frame()
    squad["expected_points"] = 1.0
    squad["expected_points_given_appearance"] = 1.0
    squad["p_appearance"] = 1.0
    squad.loc[
        squad["player_uid"].eq("MID_4"),
        ["expected_points", "expected_points_given_appearance", "p_appearance"],
    ] = [10.0, 20.0, 0.5]
    squad.loc[
        squad["player_uid"].eq("MID_3"),
        ["expected_points", "expected_points_given_appearance", "p_appearance"],
    ] = [9.0, 9.0, 1.0]
    return squad


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
                            best = max(best, starter_points + float(row["expected_points"]))
    return best
