from __future__ import annotations

from fpl_forecast.decision.rules import DecisionRules


def selling_price_tenths(purchase_price: int, current_price: int, rules: DecisionRules) -> int:
    if purchase_price <= 0 or current_price <= 0:
        raise ValueError("Prices must be positive tenths.")
    if current_price <= purchase_price:
        return current_price
    gain = current_price - purchase_price
    return purchase_price + int(gain * (1 - rules.sell_on_fee))


def validate_budget(cost_tenths: int, budget_tenths: int) -> None:
    if not isinstance(cost_tenths, int) or not isinstance(budget_tenths, int):
        raise TypeError("Budget arithmetic must use integer tenths.")
    if cost_tenths > budget_tenths:
        raise ValueError("Squad exceeds budget.")
