from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl_forecast.decision.prices import selling_price_tenths
from fpl_forecast.decision.rules import DecisionRules
from fpl_forecast.decision.squad import optimize_initial_squad


@dataclass(frozen=True)
class TransferPlan:
    transfers_out: tuple[str, ...]
    transfers_in: tuple[str, ...]
    bank_before: int
    bank_after: int
    points_hit: int
    expected_gain: float
    accepted: bool
    solver_status: str


def plan_one_week_transfer(
    current_squad: pd.DataFrame,
    candidates: pd.DataFrame,
    rules: DecisionRules,
    *,
    bank_tenths: int,
    free_transfers: int,
) -> TransferPlan:
    current = current_squad.copy()
    best_current = optimize_initial_squad(current, rules, candidate_limits={p: int((current["fpl_position"] == p).sum()) for p in rules.position_quotas})
    best_plan = TransferPlan((), (), bank_tenths, bank_tenths, 0, 0.0, False, "hold")
    owned = set(current["player_uid"])
    pool = candidates.loc[~candidates["player_uid"].isin(owned)].sort_values("expected_points", ascending=False).head(8)
    for out_row in current.itertuples(index=False):
        sell_price = int(getattr(out_row, "selling_price_tenths", getattr(out_row, "price_tenths")))
        for in_row in pool.itertuples(index=False):
            if in_row.fpl_position != out_row.fpl_position:
                continue
            new_bank = bank_tenths + sell_price - int(in_row.price_tenths)
            if new_bank < 0:
                continue
            trial = current.loc[current["player_uid"].ne(out_row.player_uid)].copy()
            trial = pd.concat([trial, pd.DataFrame([in_row._asdict()])], ignore_index=True)
            try:
                solution = optimize_initial_squad(
                    trial,
                    rules,
                    candidate_limits={p: int((trial["fpl_position"] == p).sum()) for p in rules.position_quotas},
                )
            except ValueError:
                continue
            hit = max(0, 1 - free_transfers) * rules.transfer_hit_cost
            gain = solution.objective - best_current.objective - hit
            if gain > best_plan.expected_gain:
                best_plan = TransferPlan(
                    (str(out_row.player_uid),),
                    (str(in_row.player_uid),),
                    bank_tenths,
                    new_bank,
                    hit,
                    float(gain),
                    gain > 0,
                    "exact_single_transfer_scan",
                )
    return best_plan


def computed_selling_price(purchase_price: int, current_price: int, rules: DecisionRules) -> int:
    return selling_price_tenths(purchase_price, current_price, rules)
