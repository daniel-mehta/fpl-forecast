from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_array

from fpl_forecast.decision.expected_realized import (
    evaluate_expected_realized_points,
    optimize_lineup_expected_realized,
    refine_fixed_squad_lineup,
)
from fpl_forecast.decision.lineup import LineupDecision
from fpl_forecast.decision.rules import DecisionRules, POSITIONS
from fpl_forecast.decision.squad import SquadSolution, validate_candidate_universe


@dataclass(frozen=True)
class MilpDiagnostics:
    solver_status: str
    solver_message: str
    objective_bound: float
    objective_gap: float
    runtime_seconds: float
    node_count: int


def optimize_squad_milp(
    candidates: pd.DataFrame,
    rules: DecisionRules,
    *,
    appearance_aware: bool = True,
) -> SquadSolution:
    validate_candidate_universe(candidates)
    start = perf_counter()
    frame = candidates.sort_values("player_uid").reset_index(drop=True).copy()
    n_players = len(frame)
    offsets = {"squad": 0, "starter": n_players, "captain": 2 * n_players}
    n_vars = 3 * n_players

    objective = np.zeros(n_vars)
    expected = pd.to_numeric(frame["expected_points"], errors="raise").to_numpy(dtype=float)
    appearances = pd.to_numeric(frame.get("p_appearance", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    objective[offsets["starter"] : offsets["starter"] + n_players] = -expected
    captain_bonus = expected * appearances if appearance_aware else expected
    objective[offsets["captain"] : offsets["captain"] + n_players] = -captain_bonus

    rows: list[tuple[dict[int, float], float, float]] = []
    rows.append(({offsets["squad"] + i: 1.0 for i in range(n_players)}, rules.squad_size, rules.squad_size))
    rows.append(({offsets["starter"] + i: 1.0 for i in range(n_players)}, rules.lineup_size, rules.lineup_size))
    rows.append(({offsets["captain"] + i: 1.0 for i in range(n_players)}, 1.0, 1.0))

    for position in POSITIONS:
        idx = [i for i, value in enumerate(frame["fpl_position"]) if value == position]
        rows.append(({offsets["squad"] + i: 1.0 for i in idx}, rules.position_quotas[position], rules.position_quotas[position]))
        rows.append(({offsets["starter"] + i: 1.0 for i in idx}, rules.min_starters[position], rules.max_starters[position]))

    for _, idx in frame.groupby("player_team_uid").groups.items():
        rows.append(({offsets["squad"] + int(i): 1.0 for i in idx}, 0.0, rules.max_players_per_team))

    prices = pd.to_numeric(frame["price_tenths"], errors="raise").to_numpy(dtype=float)
    rows.append(({offsets["squad"] + i: float(prices[i]) for i in range(n_players)}, 0.0, rules.budget_tenths))

    for i in range(n_players):
        rows.append(({offsets["starter"] + i: 1.0, offsets["squad"] + i: -1.0}, -np.inf, 0.0))
        rows.append(({offsets["captain"] + i: 1.0, offsets["starter"] + i: -1.0}, -np.inf, 0.0))

    matrix = lil_array((len(rows), n_vars), dtype=float)
    lower = np.empty(len(rows), dtype=float)
    upper = np.empty(len(rows), dtype=float)
    for row_index, (coefficients, row_lower, row_upper) in enumerate(rows):
        lower[row_index] = row_lower
        upper[row_index] = row_upper
        for column, value in coefficients.items():
            matrix[row_index, column] = value

    result = milp(
        objective,
        integrality=np.ones(n_vars),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    runtime = perf_counter() - start
    if not result.success:
        raise ValueError(f"MILP squad optimization failed: {result.message}")

    values = np.rint(result.x).astype(int)
    squad_ids = tuple(frame.loc[values[offsets["squad"] : offsets["squad"] + n_players].astype(bool), "player_uid"].astype(str))
    starter_ids = tuple(
        frame.loc[values[offsets["starter"] : offsets["starter"] + n_players].astype(bool), "player_uid"].astype(str)
    )
    captain = str(frame.loc[values[offsets["captain"] : offsets["captain"] + n_players].astype(bool), "player_uid"].iloc[0])
    selected = frame.loc[frame["player_uid"].isin(squad_ids)].copy()
    starters = frame.loc[frame["player_uid"].isin(starter_ids)].copy()
    vice = _vice_captain(starters, captain)
    bench = _bench_order(selected.loc[~selected["player_uid"].isin(starter_ids)].copy())
    cost = int(selected["price_tenths"].sum())
    objective_value = -float(result.fun)
    diagnostics = MilpDiagnostics(
        solver_status="optimal",
        solver_message=str(result.message),
        objective_bound=-float(result.mip_dual_bound),
        objective_gap=float(result.mip_gap),
        runtime_seconds=runtime,
        node_count=int(result.mip_node_count),
    )
    lineup = LineupDecision(
        lineup=starter_ids,
        captain=captain,
        vice_captain=vice,
        bench=tuple(str(value) for value in bench["player_uid"]),
        objective=objective_value,
        formation=_formation(starters),
        method="milp_appearance_aware" if appearance_aware else "milp_mean_only",
    )
    return SquadSolution(
        squad=squad_ids,
        lineup_decision=lineup,
        objective=objective_value,
        cost_tenths=cost,
        bank_tenths=rules.budget_tenths - cost,
        solver_status=diagnostics.solver_status,
        solver_name="scipy_highs_milp",
        candidate_count=int(n_players),
        evaluated_squads=0,
        runtime_seconds=runtime,
        optimality_scope="globally optimal for the full candidate MILP objective",
        objective_bound=diagnostics.objective_bound,
        objective_gap=diagnostics.objective_gap,
        solver_message=diagnostics.solver_message,
        solver_nodes=diagnostics.node_count,
    )


def optimize_squad_expected_realized(
    candidates: pd.DataFrame,
    rules: DecisionRules,
    *,
    search_limit: int = 24,
    search_iterations: int = 3,
    max_scenarios: int | None = None,
    seed: int = 9113,
) -> SquadSolution:
    # Retained for compatibility with earlier Phase 9B1.3 configuration. Legal
    # one-swap proposals are no longer truncated to this shortlist.
    _ = search_limit
    validate_candidate_universe(candidates)
    start = perf_counter()
    frame = candidates.sort_values("player_uid").reset_index(drop=True).copy()
    seed_solution = optimize_squad_milp(frame, rules, appearance_aware=True)
    indexed = frame.set_index("player_uid", drop=False)
    seed_squad = indexed.loc[list(seed_solution.squad)].reset_index(drop=True)
    d1_breakdown = evaluate_expected_realized_points(
        seed_squad,
        seed_solution.lineup_decision,
        rules,
    )
    best = _expected_realized_solution(
        seed_squad,
        rules,
        candidate_count=len(frame),
        evaluated_squads=1,
        runtime_seconds=perf_counter() - start,
        solver_message="seeded from D1 mean-only full-pool MILP",
        max_scenarios=max_scenarios,
        seed=seed,
    )
    seed_expected_realized_total = best.objective
    evaluated_keys = {tuple(sorted(best.squad))}
    evaluation_calls = 1
    accepted_moves = 0
    completed_iterations = 0
    termination_reason = "configured_iteration_bound_reached"
    coverage = {
        "raw_swap_proposals_generated": 0,
        "proposals_rejected_position_mismatch": 0,
        "proposals_rejected_budget": 0,
        "proposals_rejected_club_limit": 0,
        "other_illegal_proposals_rejected": 0,
        "feasible_unique_squad_proposals": 0,
        "duplicate_feasible_proposals_reevaluated": 0,
    }
    for iteration in range(max(1, search_iterations)):
        completed_iterations += 1
        neighbours, iteration_coverage = _swap_neighbourhood(frame, best.squad, rules)
        for key, value in iteration_coverage.items():
            coverage[key] += value
        iteration_best = best
        for squad_ids in neighbours:
            key = tuple(sorted(squad_ids))
            if key in evaluated_keys:
                coverage["duplicate_feasible_proposals_reevaluated"] += 1
            evaluated_keys.add(key)
            squad = indexed.loc[list(squad_ids)].reset_index(drop=True)
            evaluation_calls += 1
            candidate_solution = _inherited_expected_realized_solution(
                squad,
                rules,
                reference=best,
                candidate_count=len(frame),
                evaluated_squads=evaluation_calls,
                runtime_seconds=perf_counter() - start,
                solver_message=(
                    "D2 expected-realized optimizer: D1 full-pool MILP seed plus deterministic "
                    f"full-pool one-swap local search iteration {iteration + 1}"
                ),
                max_scenarios=max_scenarios,
                seed=seed,
            )
            if _solution_key(candidate_solution) > _solution_key(iteration_best):
                iteration_best = candidate_solution
        if _solution_key(iteration_best) <= _solution_key(best):
            termination_reason = "one_swap_local_optimum"
            break
        refined = _expected_realized_solution(
            indexed.loc[list(iteration_best.squad)].reset_index(drop=True),
            rules,
            candidate_count=len(frame),
            evaluated_squads=evaluation_calls,
            runtime_seconds=perf_counter() - start,
            solver_message=(
                "D2 expected-realized optimizer: refined lineup after accepting "
                f"full-pool one-swap improvement in iteration {iteration + 1}"
            ),
            max_scenarios=max_scenarios,
            seed=seed,
        )
        best = refined if _solution_key(refined) > _solution_key(iteration_best) else iteration_best
        accepted_moves += 1
    final_squad = indexed.loc[list(best.squad)].reset_index(drop=True)
    refined_lineup, refined_breakdown, refinement = refine_fixed_squad_lineup(
        final_squad,
        best.lineup_decision,
        rules,
    )
    diagnostics = refined_breakdown.to_dict()
    diagnostics.update(
        {
            **coverage,
            "eligible_players_full_pool": int(len(frame)),
            "selected_players_in_seed": int(len(seed_solution.squad)),
            "unselected_replacements_considered": int(len(frame) - len(seed_solution.squad)),
            "seed_expected_realized_total": float(seed_expected_realized_total),
            "mean_only_seed_objective": float(seed_solution.objective),
            "d1_nominal_starting_xi_xpoints": d1_breakdown.nominal_starting_xi_xpoints,
            "d1_expected_active_starter_points": d1_breakdown.expected_active_starter_points,
            "d1_expected_autosub_contribution": d1_breakdown.expected_autosub_contribution,
            "d1_expected_captain_bonus": d1_breakdown.expected_captain_bonus,
            "d1_expected_vice_captain_contingency": d1_breakdown.expected_vice_captain_contingency,
            "d1_expected_realized_total": d1_breakdown.expected_realized_total,
            "d1_expected_automatic_substitutions": d1_breakdown.expected_automatic_substitutions,
            "d1_probability_all_starters_appear": d1_breakdown.probability_all_starters_appear,
            "d1_probability_unreplaced_starter": d1_breakdown.probability_unreplaced_starter,
            "d1_cost_tenths": int(seed_solution.cost_tenths),
            "d1_bank_tenths": int(seed_solution.bank_tenths),
            "d1_formation": seed_solution.lineup_decision.formation,
            "d1_squad": ",".join(seed_solution.squad),
            "d1_lineup": ",".join(seed_solution.lineup_decision.lineup),
            "d1_captain": seed_solution.lineup_decision.captain,
            "d1_vice_captain": seed_solution.lineup_decision.vice_captain,
            "d1_bench_order": ",".join(seed_solution.lineup_decision.bench),
            "configured_iteration_bound": int(max(1, search_iterations)),
            "local_search_iterations": int(completed_iterations),
            "accepted_improving_moves": int(accepted_moves),
            "squad_evaluation_calls": int(evaluation_calls),
            "final_unique_squads_evaluated": int(len(evaluated_keys)),
            "termination_reason": termination_reason,
            "scenario_count": int(diagnostics.get("scenario_count", 2**rules.squad_size)),
            **{f"lineup_refinement_{key}": value for key, value in refinement.to_dict().items()},
        }
    )
    if refined_breakdown.expected_realized_total + 1e-9 < d1_breakdown.expected_realized_total:
        raise ValueError(
            "D2 expected-realized search returned a solution worse than its D1 seed "
            "under the common exact evaluator."
        )
    return SquadSolution(
        squad=best.squad,
        lineup_decision=refined_lineup,
        objective=refined_breakdown.expected_realized_total,
        cost_tenths=best.cost_tenths,
        bank_tenths=best.bank_tenths,
        solver_status="heuristic_feasible",
        solver_name="d2_expected_realized_milp_seed_local_search",
        candidate_count=int(len(frame)),
        evaluated_squads=int(len(evaluated_keys)),
        runtime_seconds=perf_counter() - start,
        optimality_scope=(
            "D1 globally optimal full-candidate MILP seed, then deterministic full-pool one-swap "
            f"local search scored by exact expected realized points; termination={termination_reason}; "
            "final fixed-squad lineup refined to a single-change local optimum; not a global proof for D2."
        ),
        objective_bound=None,
        objective_gap=None,
        solver_message=f"{best.solver_message}; fixed-squad lineup refinement complete",
        solver_nodes=seed_solution.solver_nodes,
        diagnostics=diagnostics,
    )


def _expected_realized_solution(
    squad: pd.DataFrame,
    rules: DecisionRules,
    *,
    candidate_count: int,
    evaluated_squads: int,
    runtime_seconds: float,
    solver_message: str,
    max_scenarios: int | None,
    seed: int,
) -> SquadSolution:
    lineup, breakdown = optimize_lineup_expected_realized(
        squad,
        rules,
        max_scenarios=max_scenarios,
        seed=seed,
        shortlist=1,
    )
    cost = int(squad["price_tenths"].sum())
    return SquadSolution(
        squad=tuple(str(value) for value in squad["player_uid"]),
        lineup_decision=lineup,
        objective=breakdown.expected_realized_total,
        cost_tenths=cost,
        bank_tenths=rules.budget_tenths - cost,
        solver_status="heuristic_feasible",
        solver_name="d2_expected_realized_milp_seed_local_search",
        candidate_count=int(candidate_count),
        evaluated_squads=int(evaluated_squads),
        runtime_seconds=runtime_seconds,
        optimality_scope="expected-realized scored candidate squad",
        solver_message=solver_message,
        diagnostics=breakdown.to_dict(),
    )


def _inherited_expected_realized_solution(
    squad: pd.DataFrame,
    rules: DecisionRules,
    *,
    reference: SquadSolution,
    candidate_count: int,
    evaluated_squads: int,
    runtime_seconds: float,
    solver_message: str,
    max_scenarios: int | None,
    seed: int,
) -> SquadSolution:
    squad_ids = set(squad["player_uid"].astype(str))
    old_ids = set(reference.squad)
    outgoing = tuple(sorted(old_ids - squad_ids))
    incoming = tuple(sorted(squad_ids - old_ids))
    if len(outgoing) != 1 or len(incoming) != 1:
        raise ValueError("Expected-realized inherited lineup evaluation requires exactly one squad swap.")
    outgoing_id = outgoing[0]
    incoming_id = incoming[0]
    lineup = tuple(incoming_id if player == outgoing_id else player for player in reference.lineup_decision.lineup)
    bench = tuple(incoming_id if player == outgoing_id else player for player in reference.lineup_decision.bench)
    indexed = squad.set_index("player_uid", drop=False)
    captain, vice = _expected_captain_pair(lineup, indexed)
    decision = LineupDecision(
        lineup=lineup,
        captain=captain,
        vice_captain=vice,
        bench=bench,
        objective=0.0,
        formation=reference.lineup_decision.formation,
        method="expected_realized_inherited_one_swap_lineup",
    )
    breakdown = evaluate_expected_realized_points(
        squad,
        decision,
        rules,
        max_scenarios=max_scenarios,
        seed=seed,
    )
    decision = LineupDecision(
        lineup=decision.lineup,
        captain=decision.captain,
        vice_captain=decision.vice_captain,
        bench=decision.bench,
        objective=breakdown.expected_realized_total,
        formation=decision.formation,
        method=decision.method,
    )
    cost = int(squad["price_tenths"].sum())
    return SquadSolution(
        squad=tuple(str(value) for value in squad["player_uid"]),
        lineup_decision=decision,
        objective=breakdown.expected_realized_total,
        cost_tenths=cost,
        bank_tenths=rules.budget_tenths - cost,
        solver_status="heuristic_feasible",
        solver_name="d2_expected_realized_milp_seed_local_search",
        candidate_count=int(candidate_count),
        evaluated_squads=int(evaluated_squads),
        runtime_seconds=runtime_seconds,
        optimality_scope="exact evaluation of an inherited legal one-swap lineup",
        solver_message=solver_message,
        diagnostics=breakdown.to_dict(),
    )


def _expected_captain_pair(lineup: tuple[str, ...], indexed: pd.DataFrame) -> tuple[str, str]:
    best: tuple[float, str, str] | None = None
    for captain in lineup:
        captain_row = indexed.loc[captain]
        captain_appearance = float(np.clip(float(captain_row.get("p_appearance", 1.0)), 0.0, 1.0))
        captain_expected = captain_appearance * float(captain_row["expected_points_given_appearance"])
        for vice in lineup:
            if vice == captain:
                continue
            vice_row = indexed.loc[vice]
            vice_expected = float(np.clip(float(vice_row.get("p_appearance", 1.0)), 0.0, 1.0)) * float(
                vice_row["expected_points_given_appearance"]
            )
            key = (captain_expected + (1.0 - captain_appearance) * vice_expected, captain, vice)
            if best is None or key > best:
                best = key
    if best is None:
        raise ValueError("Could not select captain and vice-captain for inherited lineup.")
    return best[1], best[2]


def _swap_neighbourhood(
    frame: pd.DataFrame,
    squad_ids: tuple[str, ...],
    rules: DecisionRules,
) -> tuple[list[tuple[str, ...]], dict[str, int]]:
    selected = set(squad_ids)
    indexed = frame.set_index("player_uid", drop=False)
    neighbours: set[tuple[str, ...]] = set()
    coverage = {
        "raw_swap_proposals_generated": 0,
        "proposals_rejected_position_mismatch": 0,
        "proposals_rejected_budget": 0,
        "proposals_rejected_club_limit": 0,
        "other_illegal_proposals_rejected": 0,
        "feasible_unique_squad_proposals": 0,
    }
    replacements = frame.loc[~frame["player_uid"].isin(selected)].sort_values("player_uid")
    for outgoing in sorted(squad_ids):
        outgoing_position = str(indexed.loc[outgoing, "fpl_position"])
        for incoming in replacements.itertuples(index=False):
            coverage["raw_swap_proposals_generated"] += 1
            if str(incoming.fpl_position) != outgoing_position:
                coverage["proposals_rejected_position_mismatch"] += 1
                continue
            trial_ids = tuple(str(incoming.player_uid) if player == outgoing else player for player in squad_ids)
            trial = indexed.loc[list(trial_ids)].copy()
            if int(trial["price_tenths"].sum()) > rules.budget_tenths:
                coverage["proposals_rejected_budget"] += 1
                continue
            if trial["player_team_uid"].value_counts().max() > rules.max_players_per_team:
                coverage["proposals_rejected_club_limit"] += 1
                continue
            if len(set(trial_ids)) != rules.squad_size:
                coverage["other_illegal_proposals_rejected"] += 1
                continue
            neighbours.add(tuple(str(value) for value in trial_ids))
    coverage["feasible_unique_squad_proposals"] = len(neighbours)
    return sorted(neighbours, key=lambda values: tuple(sorted(values))), coverage


def _solution_key(solution: SquadSolution) -> tuple[float, float, float, float, float, int, str]:
    diagnostics = solution.diagnostics or {}
    return (
        round(float(solution.objective), 8),
        round(float(diagnostics.get("expected_nominal_starting_xi_points", 0.0)), 8),
        round(float(diagnostics.get("expected_autosub_contribution", 0.0)), 8),
        round(float(diagnostics.get("expected_vice_captain_contingency", 0.0)), 8),
        -round(float(diagnostics.get("probability_unreplaced_starter", 0.0)), 8),
        int(solution.bank_tenths),
        ",".join(sorted(solution.squad)) + "|" + ",".join(solution.lineup_decision.bench),
    )


def _vice_captain(starters: pd.DataFrame, captain: str) -> str:
    candidates = starters.loc[starters["player_uid"].ne(captain)].copy()
    candidates = candidates.sort_values(["expected_points", "player_uid"], ascending=[False, True])
    return str(candidates.iloc[0]["player_uid"])


def _bench_order(bench: pd.DataFrame) -> pd.DataFrame:
    goalkeeper = bench.loc[bench["fpl_position"].eq("GKP")].sort_values(["expected_points", "player_uid"])
    outfield = bench.loc[bench["fpl_position"].ne("GKP")].sort_values(
        ["expected_points", "player_uid"],
        ascending=[False, True],
    )
    return pd.concat([goalkeeper, outfield], ignore_index=True)


def _formation(lineup: pd.DataFrame) -> str:
    counts = lineup["fpl_position"].value_counts()
    return f"{int(counts.get('DEF', 0))}-{int(counts.get('MID', 0))}-{int(counts.get('FWD', 0))}"
