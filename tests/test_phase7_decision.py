from __future__ import annotations

from itertools import combinations

import pandas as pd
import pytest

from fpl_forecast.decision.inputs import assert_frozen_decisions_target_free
from fpl_forecast.decision.lineup import apply_autosubs_and_score, optimize_lineup
from fpl_forecast.decision.milp import optimize_squad_milp
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
