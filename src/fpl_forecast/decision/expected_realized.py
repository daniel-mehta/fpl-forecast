from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from fpl_forecast.decision.lineup import LineupDecision
from fpl_forecast.decision.rules import DecisionRules, POSITIONS


@dataclass(frozen=True)
class ExpectedRealizedBreakdown:
    nominal_starting_xi_xpoints: float
    expected_nominal_starting_xi_points: float
    expected_active_starter_points: float
    expected_autosub_contribution: float
    expected_captain_bonus: float
    expected_vice_captain_contingency: float
    expected_realized_total: float
    probability_all_starters_appear: float
    expected_automatic_substitutions: float
    probability_unreplaced_starter: float
    expected_bench_points_used: float
    scenario_count: int
    probability_mass: float
    analytic_method: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def evaluate_expected_realized_points(
    squad: pd.DataFrame,
    decision: LineupDecision,
    rules: DecisionRules,
    *,
    max_scenarios: int | None = None,
    seed: int = 9113,
) -> ExpectedRealizedBreakdown:
    # Retained for API compatibility. D2 always evaluates the complete state space.
    _ = (max_scenarios, seed)
    frame = _records(squad)
    starters = tuple(decision.lineup)
    bench = tuple(decision.bench)
    _validate_decision(starters, bench, frame, rules)
    ordered_ids = [*starters, *bench]
    probabilities = np.array([frame[player]["p_appearance"] for player in ordered_ids], dtype=float)
    scenarios = _exact_appearance_states(len(ordered_ids))
    weights = np.where(scenarios, probabilities, 1.0 - probabilities).prod(axis=1)
    probability_mass = float(weights.sum())
    if not np.isclose(probability_mass, 1.0, atol=1e-12):
        raise ValueError(f"Independent appearance state probability mass is {probability_mass:.16f}, expected 1.")

    starter_conditional = np.array([_conditional_points(frame[player]) for player in starters], dtype=float)
    bench_conditional = np.array([_conditional_points(frame[player]) for player in bench], dtype=float)
    used_bench, unreplaced = _exact_autosub_outcomes(
        tuple(str(frame[player]["position"]) for player in starters),
        tuple(str(frame[player]["position"]) for player in bench),
        tuple(sorted(rules.min_starters.items())),
        tuple(sorted(rules.max_starters.items())),
    )
    nominal_by_state = scenarios[:, : rules.lineup_size] @ starter_conditional
    autosub_by_state = used_bench @ bench_conditional
    captain_index = ordered_ids.index(decision.captain)
    vice_index = ordered_ids.index(decision.vice_captain)
    captain_by_state = scenarios[:, captain_index] * _conditional_points(frame[decision.captain])
    vice_by_state = (
        ~scenarios[:, captain_index]
        & scenarios[:, vice_index]
    ) * _conditional_points(frame[decision.vice_captain])

    expected_active_starters = float(weights @ nominal_by_state)
    expected_autosubs = float(weights @ autosub_by_state)
    expected_captain = float(weights @ captain_by_state)
    expected_vice = float(weights @ vice_by_state)
    realized = expected_active_starters + expected_autosubs + expected_captain + expected_vice
    nominal_xpoints = float(sum(float(frame[player]["expected_points"]) for player in starters))
    return ExpectedRealizedBreakdown(
        nominal_starting_xi_xpoints=nominal_xpoints,
        expected_nominal_starting_xi_points=expected_active_starters,
        expected_active_starter_points=expected_active_starters,
        expected_autosub_contribution=expected_autosubs,
        expected_captain_bonus=expected_captain,
        expected_vice_captain_contingency=expected_vice,
        expected_realized_total=float(realized),
        probability_all_starters_appear=float(weights @ scenarios[:, : rules.lineup_size].all(axis=1)),
        expected_automatic_substitutions=float(weights @ used_bench.sum(axis=1)),
        probability_unreplaced_starter=float(weights @ (unreplaced > 0)),
        expected_bench_points_used=expected_autosubs,
        scenario_count=int(len(weights)),
        probability_mass=probability_mass,
        analytic_method="exact_32768_state_independent_appearance_enumeration",
    )


def optimize_lineup_expected_realized(
    squad: pd.DataFrame,
    rules: DecisionRules,
    *,
    max_scenarios: int | None = None,
    seed: int = 9113,
    shortlist: int = 8,
) -> tuple[LineupDecision, ExpectedRealizedBreakdown]:
    if len(squad) != rules.squad_size:
        raise ValueError("Expected-realized lineup optimization requires a full 15-player squad.")
    frame = _records(squad)
    candidates: list[tuple[float, tuple[str, ...], tuple[str, ...], str, str]] = []
    for lineup_ids in combinations(tuple(frame), rules.lineup_size):
        if not _is_legal_lineup(lineup_ids, frame, rules):
            continue
        bench_ids = tuple(player for player in frame if player not in lineup_ids)
        goalkeeper = tuple(player for player in bench_ids if frame[player]["position"] == "GKP")
        outfield = tuple(player for player in bench_ids if frame[player]["position"] != "GKP")
        if len(goalkeeper) != 1 or len(outfield) != 3:
            continue
        for outfield_order in permutations(outfield):
            bench_order = (*goalkeeper, *outfield_order)
            captain, vice = _best_captain_pair(lineup_ids, frame)
            score = _approximate_realized_score(lineup_ids, bench_order, captain, vice, frame, rules)
            candidates.append((score, tuple(lineup_ids), tuple(bench_order), captain, vice))
    if not candidates:
        raise ValueError("No legal expected-realized lineup can be formed from squad.")
    candidates.sort(key=lambda item: (round(item[0], 10), ",".join(sorted(item[1])), ",".join(item[2])), reverse=True)
    best_decision: LineupDecision | None = None
    best_breakdown: ExpectedRealizedBreakdown | None = None
    for _, lineup_ids, bench_order, captain, vice in candidates[: max(1, shortlist)]:
        decision = LineupDecision(
            lineup=tuple(str(value) for value in lineup_ids),
            captain=str(captain),
            vice_captain=str(vice),
            bench=tuple(str(value) for value in bench_order),
            objective=0.0,
            formation=_formation(lineup_ids, frame),
            method="expected_realized_independent_scenarios",
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
        if best_decision is None or _decision_key(decision, breakdown, squad, rules) > _decision_key(
            best_decision,
            best_breakdown,  # type: ignore[arg-type]
            squad,
            rules,
        ):
            best_decision = decision
            best_breakdown = breakdown
    if best_decision is None or best_breakdown is None:
        raise ValueError("No expected-realized lineup survived evaluation.")
    return best_decision, best_breakdown


def _records(squad: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    rows = {}
    for row in squad.sort_values("player_uid").itertuples(index=False):
        player = str(row.player_uid)
        p_appearance = float(np.clip(float(getattr(row, "p_appearance", 1.0)), 0.0, 1.0))
        expected_points = float(getattr(row, "expected_points", 0.0))
        rows[player] = {
            "position": str(row.fpl_position),
            "team": str(row.player_team_uid),
            "expected_points": expected_points,
            "p_appearance": p_appearance,
            "conditional_points": 0.0 if p_appearance <= 1e-9 else expected_points / p_appearance,
        }
    return rows


@lru_cache(maxsize=4)
def _exact_appearance_states(n_players: int) -> np.ndarray:
    masks = np.arange(2**n_players, dtype=np.uint32)
    return ((masks[:, None] >> np.arange(n_players, dtype=np.uint32)) & 1).astype(bool)


@lru_cache(maxsize=128)
def _exact_autosub_outcomes(
    starter_positions: tuple[str, ...],
    bench_positions: tuple[str, ...],
    min_starters: tuple[tuple[str, int], ...],
    max_starters: tuple[tuple[str, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    scenarios = _exact_appearance_states(len(starter_positions) + len(bench_positions))
    used_bench = np.zeros((len(scenarios), len(bench_positions)), dtype=bool)
    unreplaced = np.zeros(len(scenarios), dtype=np.uint8)
    minimums = dict(min_starters)
    maximums = dict(max_starters)
    frame = {
        **{f"s{index}": {"position": position} for index, position in enumerate(starter_positions)},
        **{f"b{index}": {"position": position} for index, position in enumerate(bench_positions)},
    }
    starters = tuple(f"s{index}" for index in range(len(starter_positions)))
    bench = tuple(f"b{index}" for index in range(len(bench_positions)))
    rules = _AutosubRules(
        lineup_size=len(starter_positions),
        min_starters=minimums,
        max_starters=maximums,
    )
    for scenario_index, appearance in enumerate(scenarios):
        appeared = {
            player
            for index, player in enumerate((*starters, *bench))
            if bool(appearance[index])
        }
        _, used, missing = _active_after_autosubs(starters, bench, appeared, frame, rules)
        for player in used:
            used_bench[scenario_index, int(player[1:])] = True
        unreplaced[scenario_index] = missing
    return used_bench, unreplaced


@dataclass(frozen=True)
class _AutosubRules:
    lineup_size: int
    min_starters: dict[str, int]
    max_starters: dict[str, int]


def _active_after_autosubs(
    starters: tuple[str, ...],
    bench: tuple[str, ...],
    appeared: set[str],
    frame: dict[str, dict[str, float | str]],
    rules: DecisionRules,
) -> tuple[list[str], list[str], int]:
    active = list(starters)
    available_bench = list(bench)
    used_bench: list[str] = []
    unreplaced = 0
    for starter in starters:
        if starter in appeared:
            continue
        position = str(frame[starter]["position"])
        replacement = _find_replacement(starter, position, active, available_bench, appeared, frame, rules)
        if replacement is None:
            unreplaced += 1
            continue
        active.remove(starter)
        active.append(replacement)
        available_bench.remove(replacement)
        used_bench.append(replacement)
    return active, used_bench, unreplaced


def _find_replacement(
    starter: str,
    position: str,
    active: list[str],
    bench: list[str],
    appeared: set[str],
    frame: dict[str, dict[str, float | str]],
    rules: DecisionRules,
) -> str | None:
    if position == "GKP":
        candidates = [player for player in bench if frame[player]["position"] == "GKP"]
    else:
        candidates = [player for player in bench if frame[player]["position"] != "GKP"]
    for candidate in candidates:
        if candidate not in appeared:
            continue
        trial = active.copy()
        trial.remove(starter)
        trial.append(candidate)
        if _is_legal_lineup(tuple(trial), frame, rules):
            return candidate
    return None


def _validate_decision(
    starters: tuple[str, ...],
    bench: tuple[str, ...],
    frame: dict[str, dict[str, float | str]],
    rules: DecisionRules,
) -> None:
    if len(starters) != rules.lineup_size or len(bench) != rules.bench_size:
        raise ValueError("Decision does not contain the required lineup and bench sizes.")
    if set(starters) & set(bench) or set(starters) | set(bench) != set(frame):
        raise ValueError("Decision lineup and bench must partition the 15-player squad.")
    if not _is_legal_lineup(starters, frame, rules):
        raise ValueError("Decision starting lineup is not a legal formation.")
    if sum(1 for player in bench if frame[player]["position"] == "GKP") != 1:
        raise ValueError("Bench order must contain exactly one goalkeeper.")
    if bench[0] not in frame or frame[bench[0]]["position"] != "GKP":
        raise ValueError("Bench goalkeeper must be first in the bench tuple.")
    if decision_players := {player for player in (starters + bench) if player not in frame}:
        raise ValueError(f"Decision references unknown players: {', '.join(sorted(decision_players))}")


def _is_legal_lineup(
    lineup: tuple[str, ...],
    frame: dict[str, dict[str, float | str]],
    rules: DecisionRules,
) -> bool:
    if len(lineup) != rules.lineup_size:
        return False
    counts = {position: 0 for position in POSITIONS}
    for player in lineup:
        position = str(frame[player]["position"])
        counts[position] = counts.get(position, 0) + 1
    return all(counts.get(pos, 0) >= minimum for pos, minimum in rules.min_starters.items()) and all(
        counts.get(pos, 0) <= maximum for pos, maximum in rules.max_starters.items()
    )


def _best_captain_pair(lineup: tuple[str, ...], frame: dict[str, dict[str, float | str]]) -> tuple[str, str]:
    best: tuple[float, str, str] | None = None
    for captain in lineup:
        p_captain = float(frame[captain]["p_appearance"])
        captain_value = float(frame[captain]["conditional_points"]) * p_captain
        for vice in lineup:
            if vice == captain:
                continue
            value = captain_value + (1.0 - p_captain) * float(frame[vice]["p_appearance"]) * float(
                frame[vice]["conditional_points"]
            )
            key = (value, str(captain), str(vice))
            if best is None or key > best:
                best = key
    if best is None:
        raise ValueError("Could not choose captain and vice-captain.")
    return (best[1], best[2])


def _approximate_realized_score(
    lineup: tuple[str, ...],
    bench: tuple[str, ...],
    captain: str,
    vice: str,
    frame: dict[str, dict[str, float | str]],
    rules: DecisionRules,
) -> float:
    starter_points = sum(float(frame[player]["expected_points"]) for player in lineup)
    missing_starter_expectation = sum(1.0 - float(frame[player]["p_appearance"]) for player in lineup)
    bench_value = 0.0
    for index, player in enumerate(bench):
        if frame[player]["position"] == "GKP":
            gkp_missing = sum(
                1.0 - float(frame[starter]["p_appearance"])
                for starter in lineup
                if frame[starter]["position"] == "GKP"
            )
            bench_value += gkp_missing * float(frame[player]["expected_points"])
            continue
        order_discount = max(0.0, 1.0 - 0.25 * index)
        if _can_ever_substitute(player, lineup, frame, rules):
            bench_value += missing_starter_expectation * order_discount * float(frame[player]["expected_points"]) / 3.0
    p_captain = float(frame[captain]["p_appearance"])
    captain_bonus = float(frame[captain]["expected_points"])
    vice_bonus = (1.0 - p_captain) * float(frame[vice]["expected_points"])
    return starter_points + bench_value + captain_bonus + vice_bonus


def _can_ever_substitute(
    bench_player: str,
    lineup: tuple[str, ...],
    frame: dict[str, dict[str, float | str]],
    rules: DecisionRules,
) -> bool:
    for starter in lineup:
        if frame[starter]["position"] == "GKP":
            continue
        trial = list(lineup)
        trial.remove(starter)
        trial.append(bench_player)
        if _is_legal_lineup(tuple(trial), frame, rules):
            return True
    return False


def _conditional_points(record: dict[str, float | str]) -> float:
    return float(record["conditional_points"])


def _formation(lineup: tuple[str, ...], frame: dict[str, dict[str, float | str]]) -> str:
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for player in lineup:
        position = str(frame[player]["position"])
        if position in counts:
            counts[position] += 1
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def _decision_key(
    decision: LineupDecision,
    breakdown: ExpectedRealizedBreakdown,
    squad: pd.DataFrame,
    rules: DecisionRules,
) -> tuple[float, float, float, float, float, int, str]:
    selected = squad.set_index("player_uid")
    starter_sum = float(selected.loc[list(decision.lineup), "expected_points"].sum())
    bench_sum = float(selected.loc[list(decision.bench), "expected_points"].sum())
    bank = rules.budget_tenths - int(squad["price_tenths"].sum())
    return (
        round(breakdown.expected_realized_total, 8),
        round(starter_sum, 8),
        round(bench_sum, 8),
        round(breakdown.expected_vice_captain_contingency, 8),
        -round(breakdown.probability_unreplaced_starter, 8),
        bank,
        ",".join(sorted(decision.lineup)) + "|" + ",".join(decision.bench) + "|" + decision.captain + "|" + decision.vice_captain,
    )
