from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import pandas as pd

from fpl_forecast.decision.lineup import optimize_lineup
from fpl_forecast.decision.prices import selling_price_tenths
from fpl_forecast.decision.rules import DecisionRules


@dataclass(frozen=True)
class TransferAction:
    gameweek: int
    transfers_out: tuple[str, ...]
    transfers_in: tuple[str, ...]
    bank_before: int
    bank_after: int
    free_transfers_before: int
    free_transfers_after: int
    points_hit: int
    lineup: tuple[str, ...]
    captain: str
    vice_captain: str
    bench: tuple[str, ...]
    objective_after_hit: float


@dataclass(frozen=True)
class MultiGameweekTransferPlan:
    actions: tuple[TransferAction, ...]
    total_objective: float
    solver_status: str
    objective_bound: float
    objective_gap: float
    runtime_seconds: float
    states_evaluated: int


def plan_multi_gameweek_transfers(
    initial_squad: pd.DataFrame,
    weekly_candidates: dict[int, pd.DataFrame],
    rules: DecisionRules,
    *,
    bank_tenths: int,
    free_transfers: int,
    max_transfers_per_gameweek: int = 2,
) -> MultiGameweekTransferPlan:
    start = perf_counter()
    purchase_prices = {
        str(row.player_uid): int(row.price_tenths) for row in initial_squad.itertuples(index=False)
    }
    initial_ids = tuple(str(value) for value in initial_squad["player_uid"])
    best: tuple[float, tuple[TransferAction, ...]] | None = None
    states = 0

    def search(
        index: int,
        owned: tuple[str, ...],
        prices_paid: dict[str, int],
        bank: int,
        free: int,
        actions: tuple[TransferAction, ...],
        total: float,
    ) -> None:
        nonlocal best, states
        gameweeks = sorted(weekly_candidates)
        if index == len(gameweeks):
            if best is None or total > best[0]:
                best = (total, actions)
            return
        gameweek = gameweeks[index]
        candidates = weekly_candidates[gameweek].copy()
        for transfers_out, transfers_in, next_owned, next_paid, next_bank in _transfer_options(
            owned,
            prices_paid,
            candidates,
            rules,
            bank,
            max_transfers_per_gameweek,
        ):
            used = len(transfers_in)
            hit = max(0, used - free) * rules.transfer_hit_cost
            next_free = min(rules.free_transfer_cap, max(0, free - used) + 1)
            squad = candidates.set_index("player_uid").loc[list(next_owned)].reset_index()
            if not _legal_squad(squad, rules):
                continue
            lineup = optimize_lineup(squad, rules, appearance_aware=True)
            objective = lineup.objective - hit
            states += 1
            action = TransferAction(
                gameweek=gameweek,
                transfers_out=transfers_out,
                transfers_in=transfers_in,
                bank_before=bank,
                bank_after=next_bank,
                free_transfers_before=free,
                free_transfers_after=next_free,
                points_hit=hit,
                lineup=lineup.lineup,
                captain=lineup.captain,
                vice_captain=lineup.vice_captain,
                bench=lineup.bench,
                objective_after_hit=objective,
            )
            search(
                index + 1,
                tuple(sorted(next_owned)),
                next_paid,
                next_bank,
                next_free,
                (*actions, action),
                total + objective,
            )

    search(0, tuple(sorted(initial_ids)), purchase_prices, bank_tenths, free_transfers, (), 0.0)
    if best is None:
        raise ValueError("No legal multi-gameweek transfer plan found.")
    runtime = perf_counter() - start
    return MultiGameweekTransferPlan(
        actions=best[1],
        total_objective=float(best[0]),
        solver_status="exhaustive_optimal",
        objective_bound=float(best[0]),
        objective_gap=0.0,
        runtime_seconds=runtime,
        states_evaluated=states,
    )


def computed_selling_price(purchase_price: int, current_price: int, rules: DecisionRules) -> int:
    return selling_price_tenths(purchase_price, current_price, rules)


def plan_to_frame(plan: MultiGameweekTransferPlan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameweek": action.gameweek,
                "transfers_out": action.transfers_out,
                "transfers_in": action.transfers_in,
                "bank_before": action.bank_before,
                "bank_after": action.bank_after,
                "free_transfers_before": action.free_transfers_before,
                "free_transfers_after": action.free_transfers_after,
                "points_hit": action.points_hit,
                "captain": action.captain,
                "vice_captain": action.vice_captain,
                "lineup": action.lineup,
                "bench": action.bench,
                "objective_after_hit": action.objective_after_hit,
                "solver_status": plan.solver_status,
                "objective_bound": plan.objective_bound,
                "objective_gap": plan.objective_gap,
                "runtime_seconds": plan.runtime_seconds,
                "states_evaluated": plan.states_evaluated,
            }
            for action in plan.actions
        ]
    )


def _transfer_options(
    owned: tuple[str, ...],
    purchase_prices: dict[str, int],
    candidates: pd.DataFrame,
    rules: DecisionRules,
    bank: int,
    max_transfers: int,
):
    indexed = candidates.set_index("player_uid")
    owned_set = set(owned)
    available = [str(value) for value in candidates.loc[~candidates["player_uid"].isin(owned_set), "player_uid"]]
    yield (), (), tuple(sorted(owned)), dict(purchase_prices), bank
    for count in range(1, max_transfers + 1):
        for outs in combinations(owned, count):
            out_positions = indexed.loc[list(outs), "fpl_position"].tolist()
            for ins in combinations(available, count):
                in_positions = indexed.loc[list(ins), "fpl_position"].tolist()
                if sorted(out_positions) != sorted(in_positions):
                    continue
                next_bank = bank
                next_paid = dict(purchase_prices)
                for player in outs:
                    next_bank += selling_price_tenths(
                        next_paid[player],
                        int(indexed.loc[player, "price_tenths"]),
                        rules,
                    )
                    del next_paid[player]
                for player in ins:
                    price = int(indexed.loc[player, "price_tenths"])
                    next_bank -= price
                    next_paid[player] = price
                if next_bank < 0:
                    continue
                next_owned = tuple(sorted((owned_set.difference(outs)).union(ins)))
                yield tuple(str(value) for value in outs), tuple(str(value) for value in ins), next_owned, next_paid, next_bank


def _legal_squad(squad: pd.DataFrame, rules: DecisionRules) -> bool:
    if len(squad) != rules.squad_size:
        return False
    if int(squad["price_tenths"].sum()) > rules.budget_tenths:
        return False
    if squad["player_team_uid"].value_counts().max() > rules.max_players_per_team:
        return False
    counts = squad["fpl_position"].value_counts().to_dict()
    return all(counts.get(position, 0) == quota for position, quota in rules.position_quotas.items())
