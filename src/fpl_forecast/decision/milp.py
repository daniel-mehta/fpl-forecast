from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_array

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
