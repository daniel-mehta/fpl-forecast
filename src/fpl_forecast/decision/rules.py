from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


POSITIONS = ("GKP", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class DecisionRules:
    version: str
    squad_size: int
    position_quotas: dict[str, int]
    lineup_size: int
    min_starters: dict[str, int]
    max_starters: dict[str, int]
    max_players_per_team: int
    budget_tenths: int
    bench_size: int
    transfer_hit_cost: int
    free_transfer_cap: int
    sell_on_fee: float
    vice_captain_enabled: bool


def default_rules() -> DecisionRules:
    return DecisionRules(
        version="fpl_2025_26_api_verified",
        squad_size=15,
        position_quotas={"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
        lineup_size=11,
        min_starters={"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
        max_starters={"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
        max_players_per_team=3,
        budget_tenths=1000,
        bench_size=4,
        transfer_hit_cost=4,
        free_transfer_cap=5,
        sell_on_fee=0.5,
        vice_captain_enabled=True,
    )


def validate_rules(rules: DecisionRules) -> list[str]:
    warnings: list[str] = []
    if sum(rules.position_quotas.values()) != rules.squad_size:
        raise ValueError("Position quotas do not sum to squad size.")
    if rules.lineup_size + rules.bench_size != rules.squad_size:
        raise ValueError("Lineup and bench sizes do not sum to squad size.")
    if rules.position_quotas["GKP"] < 2 or rules.min_starters["GKP"] != 1 or rules.max_starters["GKP"] != 1:
        raise ValueError("Goalkeeper squad or lineup rules are inconsistent.")
    if sum(rules.min_starters.values()) > rules.lineup_size:
        raise ValueError("Minimum starters exceed lineup size.")
    if rules.budget_tenths <= 0 or rules.budget_tenths % 1:
        raise ValueError("Budget must be a positive integer in tenths.")
    if rules.max_players_per_team <= 0:
        raise ValueError("Maximum players per team must be positive.")
    if not rules.vice_captain_enabled:
        warnings.append("Vice-captain disabled by rules.")
    return warnings


def rules_from_bootstrap(path: Path) -> DecisionRules:
    data = json.loads(path.read_text(encoding="utf-8"))
    settings = data["game_settings"]
    element_types = data["element_types"]
    quotas = {item["singular_name_short"]: int(item["squad_select"]) for item in element_types}
    min_play = {item["singular_name_short"]: int(item["squad_min_play"]) for item in element_types}
    max_play = {item["singular_name_short"]: int(item["squad_max_play"]) for item in element_types}
    return DecisionRules(
        version="fpl_2025_26_api_verified",
        squad_size=int(settings["squad_squadsize"]),
        position_quotas=quotas,
        lineup_size=int(settings["squad_squadplay"]),
        min_starters=min_play,
        max_starters=max_play,
        max_players_per_team=int(settings["squad_team_limit"]),
        budget_tenths=int(settings["squad_total_spend"]),
        bench_size=int(settings["squad_squadsize"]) - int(settings["squad_squadplay"]),
        transfer_hit_cost=4,
        free_transfer_cap=int(settings.get("max_extra_free_transfers", 4)) + 1,
        sell_on_fee=float(settings.get("transfers_sell_on_fee", 0.5)),
        vice_captain_enabled=bool(settings.get("sys_vice_captain_enabled", True)),
    )


def assert_rules_match_config(api_rules: DecisionRules, configured: DecisionRules) -> None:
    fields = [
        "squad_size",
        "position_quotas",
        "lineup_size",
        "min_starters",
        "max_starters",
        "max_players_per_team",
        "budget_tenths",
    ]
    mismatches = [field for field in fields if getattr(api_rules, field) != getattr(configured, field)]
    if mismatches:
        raise ValueError(f"API rules conflict with configured rules: {', '.join(mismatches)}")
