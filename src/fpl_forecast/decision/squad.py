from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest
from itertools import combinations
from time import perf_counter

import pandas as pd

from fpl_forecast.decision.lineup import LineupDecision, optimize_lineup
from fpl_forecast.decision.rules import DecisionRules, POSITIONS


@dataclass(frozen=True)
class SquadSolution:
    squad: tuple[str, ...]
    lineup_decision: LineupDecision
    objective: float
    cost_tenths: int
    bank_tenths: int
    solver_status: str
    solver_name: str
    candidate_count: int
    evaluated_squads: int
    runtime_seconds: float
    optimality_scope: str
    objective_bound: float | None = None
    objective_gap: float | None = None
    solver_message: str | None = None
    solver_nodes: int | None = None


def optimize_initial_squad(
    candidates: pd.DataFrame,
    rules: DecisionRules,
    *,
    candidate_limits: dict[str, int] | None = None,
    squad_evaluation_limit: int | None = None,
    appearance_aware: bool = True,
) -> SquadSolution:
    validate_candidate_universe(candidates)
    start = perf_counter()
    pruned = prune_candidates(candidates, rules, candidate_limits or {})
    best: SquadSolution | None = None
    legal = 0
    grouped = {position: pruned.loc[pruned["fpl_position"].eq(position)].copy() for position in POSITIONS}
    for position, quota in rules.position_quotas.items():
        if len(grouped[position]) < quota:
            raise ValueError(f"Infeasible candidate universe: not enough {position}.")
    combo_iters = [
        list(combinations(list(grouped[position]["player_uid"]), rules.position_quotas[position]))
        for position in POSITIONS
    ]
    indexed = pruned.set_index("player_uid")
    shortlist: list[tuple[float, int, tuple[str, ...]]] = []
    for gkp in combo_iters[0]:
        for defenders in combo_iters[1]:
            for mids in combo_iters[2]:
                for fwds in combo_iters[3]:
                    selected = (*gkp, *defenders, *mids, *fwds)
                    squad = indexed.loc[list(selected)].reset_index()
                    cost = int(squad["price_tenths"].sum())
                    if cost > rules.budget_tenths:
                        continue
                    if squad["player_team_uid"].value_counts().max() > rules.max_players_per_team:
                        continue
                    legal += 1
                    shortlist.append((_rough_squad_score(squad), rules.budget_tenths - cost, selected))
    if not shortlist:
        raise ValueError("No legal squad found within budget and club constraints.")
    if squad_evaluation_limit is not None and len(shortlist) > squad_evaluation_limit:
        shortlist = nlargest(squad_evaluation_limit, shortlist, key=lambda item: (item[0], item[1], sorted(item[2])))
    evaluated = 0
    for _, _, selected in shortlist:
        squad = indexed.loc[list(selected)].reset_index()
        cost = int(squad["price_tenths"].sum())
        try:
            lineup = optimize_lineup(squad, rules, appearance_aware=appearance_aware)
        except ValueError:
            continue
        evaluated += 1
        objective = lineup.objective
        solution = SquadSolution(
            squad=tuple(str(value) for value in selected),
            lineup_decision=lineup,
            objective=objective,
            cost_tenths=cost,
            bank_tenths=rules.budget_tenths - cost,
            solver_status="optimal",
            solver_name="deterministic_pruned_enumeration",
            candidate_count=int(len(pruned)),
            evaluated_squads=evaluated,
            runtime_seconds=perf_counter() - start,
            optimality_scope=_optimality_scope(len(shortlist), legal, squad_evaluation_limit),
        )
        if best is None or _solution_key(solution) > _solution_key(best):
            best = solution
    if best is None:
        raise ValueError("No legal lineup found for any legal squad.")
    return best


def prune_candidates(
    candidates: pd.DataFrame,
    rules: DecisionRules,
    candidate_limits: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for position in POSITIONS:
        quota = rules.position_quotas[position]
        limit = max(int(candidate_limits.get(position, quota)), quota)
        group = candidates.loc[candidates["fpl_position"].eq(position)].copy()
        group["_value_score"] = group["expected_points"] / group["price_tenths"].clip(lower=1)
        expected = group.sort_values(
            ["expected_points", "price_tenths", "player_uid"],
            ascending=[False, True, True],
        )
        value = group.sort_values(
            ["_value_score", "expected_points", "player_uid"],
            ascending=[False, False, True],
        )
        cheap = group.sort_values(
            ["price_tenths", "expected_points", "player_uid"],
            ascending=[True, False, True],
        )
        team_best = expected.groupby("player_team_uid", as_index=False, group_keys=False).head(1)
        selected: list[pd.DataFrame] = [
            expected.head(max(1, quota // 2)),
            team_best.head(limit),
            value.head(max(1, limit // 3)),
            cheap.head(limit),
        ]
        combined = pd.concat(selected, ignore_index=True).drop_duplicates("player_uid")
        if len(combined) < quota:
            filler = group.sort_values(
                ["expected_points", "price_tenths", "player_uid"],
                ascending=[False, True, True],
            ).head(quota)
            combined = pd.concat([combined, filler], ignore_index=True).drop_duplicates("player_uid")
        rows.append(combined.head(limit).drop(columns=["_value_score"], errors="ignore"))
    return pd.concat(rows, ignore_index=True)


def validate_candidate_universe(frame: pd.DataFrame) -> None:
    required = {"player_uid", "fpl_position", "player_team_uid", "price_tenths", "expected_points"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Candidate universe missing columns: {', '.join(sorted(missing))}")
    if frame["player_uid"].duplicated().any():
        raise ValueError("Duplicate player IDs in candidate universe.")
    if frame["fpl_position"].isna().any() or set(frame["fpl_position"]).difference(POSITIONS):
        raise ValueError("Invalid FPL positions in candidate universe.")
    prices = pd.to_numeric(frame["price_tenths"], errors="coerce")
    if prices.isna().any() or prices.le(0).any() or (prices % 1).ne(0).any():
        raise ValueError("Candidate prices must be positive integer tenths.")
    points = pd.to_numeric(frame["expected_points"], errors="coerce")
    if points.isna().any():
        raise ValueError("Candidate expected points must be numeric.")


def squad_table(solution: SquadSolution, candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.set_index("player_uid").loc[list(solution.squad)].reset_index()
    frame["selected_role"] = "squad"
    frame.loc[frame["player_uid"].isin(solution.lineup_decision.lineup), "selected_role"] = "starter"
    frame.loc[frame["player_uid"].eq(solution.lineup_decision.captain), "selected_role"] = "captain"
    frame.loc[frame["player_uid"].eq(solution.lineup_decision.vice_captain), "selected_role"] = "vice_captain"
    bench_order = {player: index for index, player in enumerate(solution.lineup_decision.bench, start=1)}
    frame["bench_order"] = frame["player_uid"].map(bench_order)
    return frame


def _solution_key(solution: SquadSolution) -> tuple[float, int, str]:
    return (round(solution.objective, 10), solution.bank_tenths, ",".join(sorted(solution.squad)))


def _rough_squad_score(squad: pd.DataFrame) -> float:
    sorted_points = squad["expected_points"].sort_values(ascending=False)
    return float(sorted_points.head(11).sum() + sorted_points.iloc[0])


def _optimality_scope(shortlisted: int, legal: int, limit: int | None) -> str:
    if limit is None or shortlisted == legal:
        return "globally optimal within deterministic pruned candidate universe"
    return (
        "lineup-optimal within deterministic top-"
        f"{shortlisted} shortlist from {legal} legal pruned candidate squads"
    )
