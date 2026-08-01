from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from fpl_forecast.decision.rules import DecisionRules


@dataclass(frozen=True)
class LineupDecision:
    lineup: tuple[str, ...]
    captain: str
    vice_captain: str
    bench: tuple[str, ...]
    objective: float
    formation: str
    method: str


def optimize_lineup(
    squad: pd.DataFrame,
    rules: DecisionRules,
    *,
    appearance_aware: bool = True,
) -> LineupDecision:
    if len(squad) != rules.squad_size:
        raise ValueError("Lineup optimization requires a full 15-player squad.")
    best: LineupDecision | None = None
    records = {
        str(row.player_uid): {
            "position": str(row.fpl_position),
            "expected_points": float(row.expected_points),
            "p_appearance": float(getattr(row, "p_appearance", 1.0)),
        }
        for row in squad.itertuples(index=False)
    }
    ids = list(records)
    for lineup_ids in combinations(ids, rules.lineup_size):
        if not _is_legal_lineup_ids(lineup_ids, records, rules):
            continue
        bench_ids = _bench_order_ids([player for player in ids if player not in lineup_ids], records)
        starters = sorted(lineup_ids, key=lambda player: (-float(records[player]["expected_points"]), player))
        captain = starters[0]
        vice = starters[1]
        objective = _lineup_objective_ids(
            lineup_ids,
            bench_ids,
            captain,
            vice,
            records,
            appearance_aware=appearance_aware,
        )
        decision = LineupDecision(
            lineup=tuple(str(value) for value in lineup_ids),
            captain=captain,
            vice_captain=vice,
            bench=tuple(bench_ids),
            objective=objective,
            formation=_formation_ids(lineup_ids, records),
            method="appearance_aware" if appearance_aware else "mean_only",
        )
        if best is None or _lineup_key(decision) > _lineup_key(best):
            best = decision
    if best is None:
        raise ValueError("No legal lineup can be formed from squad.")
    return best


def is_legal_lineup(lineup: pd.DataFrame, rules: DecisionRules) -> bool:
    counts = lineup["fpl_position"].value_counts().to_dict()
    if len(lineup) != rules.lineup_size:
        return False
    for position, minimum in rules.min_starters.items():
        if counts.get(position, 0) < minimum:
            return False
    for position, maximum in rules.max_starters.items():
        if counts.get(position, 0) > maximum:
            return False
    return True


def _is_legal_lineup_ids(
    lineup_ids: tuple[str, ...],
    records: dict[str, dict[str, float | str]],
    rules: DecisionRules,
) -> bool:
    counts: dict[str, int] = {}
    for player in lineup_ids:
        position = str(records[player]["position"])
        counts[position] = counts.get(position, 0) + 1
    for position, minimum in rules.min_starters.items():
        if counts.get(position, 0) < minimum:
            return False
    for position, maximum in rules.max_starters.items():
        if counts.get(position, 0) > maximum:
            return False
    return True


def apply_autosubs_and_score(
    squad: pd.DataFrame,
    decision: LineupDecision,
    rules: DecisionRules,
) -> dict[str, object]:
    frame = squad.set_index("player_uid")
    starters = list(decision.lineup)
    bench = list(decision.bench)
    active = starters.copy()
    used_bench: list[str] = []
    unreplaced_starters: list[str] = []
    for starter in starters:
        if _appeared(frame.loc[starter]):
            continue
        position = frame.loc[starter, "fpl_position"]
        replacement = _find_replacement(starter, position, active, bench, frame, rules)
        if replacement is not None:
            active.remove(starter)
            active.append(replacement)
            bench.remove(replacement)
            used_bench.append(replacement)
        else:
            unreplaced_starters.append(starter)
    captain = decision.captain
    vice = decision.vice_captain
    captain_multiplier_player = None
    if captain in active and _appeared(frame.loc[captain]):
        captain_multiplier_player = captain
    elif vice in active and _appeared(frame.loc[vice]):
        captain_multiplier_player = vice
    points = sum(float(frame.loc[player, "actual_points"]) for player in active)
    autosub_points = sum(float(frame.loc[player, "actual_points"]) for player in used_bench)
    captain_points = 0.0
    if captain_multiplier_player is not None:
        captain_points = float(frame.loc[captain_multiplier_player, "actual_points"])
        points += captain_points
    return {
        "realized_points": points,
        "captain_bonus_points": captain_points,
        "active_players": tuple(active),
        "autosub_players": tuple(used_bench),
        "autosub_points": autosub_points,
        "unreplaced_starters": tuple(unreplaced_starters),
        "captain_multiplier_player": captain_multiplier_player,
    }


def _lineup_objective(
    lineup: pd.DataFrame,
    bench: pd.DataFrame,
    captain: str,
    vice: str,
    *,
    appearance_aware: bool,
) -> float:
    starter_points = float(lineup["expected_points"].sum())
    captain_row = lineup.loc[lineup["player_uid"].eq(captain)].iloc[0]
    vice_row = lineup.loc[lineup["player_uid"].eq(vice)].iloc[0]
    if appearance_aware:
        captain_bonus = float(captain_row["expected_points"])
        captain_bonus += float(
            vice_row["expected_points"]
            * (1 - captain_row.get("p_appearance", 1.0))
        )
        bench_value = float(
            (bench["expected_points"] * (1 - bench.index.to_series().rank(pct=True) * 0.05)).sum() * 0.05
        )
    else:
        captain_bonus = float(captain_row["expected_points"])
        bench_value = 0.0
    return starter_points + captain_bonus + bench_value


def _lineup_objective_ids(
    lineup: tuple[str, ...],
    bench: list[str],
    captain: str,
    vice: str,
    records: dict[str, dict[str, float | str]],
    *,
    appearance_aware: bool,
) -> float:
    starter_points = sum(float(records[player]["expected_points"]) for player in lineup)
    if not appearance_aware:
        return starter_points + float(records[captain]["expected_points"])
    captain_bonus = float(records[captain]["expected_points"])
    captain_bonus += (
        float(records[vice]["expected_points"])
        * (1 - float(records[captain]["p_appearance"]))
    )
    bench_value = sum(
        float(records[player]["expected_points"]) * (1 - (index + 1) * 0.05)
        for index, player in enumerate(bench)
    )
    return starter_points + captain_bonus + bench_value * 0.05


def _bench_order(bench: pd.DataFrame) -> pd.DataFrame:
    goalkeeper = bench.loc[bench["fpl_position"].eq("GKP")].sort_values(["expected_points", "player_uid"])
    outfield = bench.loc[bench["fpl_position"].ne("GKP")].sort_values(
        ["expected_points", "player_uid"],
        ascending=[False, True],
    )
    return pd.concat([goalkeeper, outfield], ignore_index=True)


def _bench_order_ids(bench: list[str], records: dict[str, dict[str, float | str]]) -> list[str]:
    goalkeepers = [player for player in bench if records[player]["position"] == "GKP"]
    outfield = [player for player in bench if records[player]["position"] != "GKP"]
    goalkeepers.sort(key=lambda player: (float(records[player]["expected_points"]), player))
    outfield.sort(key=lambda player: (-float(records[player]["expected_points"]), player))
    return [*goalkeepers, *outfield]


def _find_replacement(
    starter: str,
    position: str,
    active: list[str],
    bench: list[str],
    frame: pd.DataFrame,
    rules: DecisionRules,
) -> str | None:
    if position == "GKP":
        candidates = [player for player in bench if frame.loc[player, "fpl_position"] == "GKP"]
    else:
        candidates = [player for player in bench if frame.loc[player, "fpl_position"] != "GKP"]
    for candidate in candidates:
        if not _appeared(frame.loc[candidate]):
            continue
        trial = active.copy()
        trial.remove(starter)
        trial.append(candidate)
        if is_legal_lineup(frame.loc[trial].reset_index(), rules):
            return candidate
    return None


def _appeared(row: pd.Series) -> bool:
    return float(row.get("actual_minutes", 0)) > 0


def _formation(lineup: pd.DataFrame) -> str:
    counts = lineup["fpl_position"].value_counts()
    return f"{int(counts.get('DEF', 0))}-{int(counts.get('MID', 0))}-{int(counts.get('FWD', 0))}"


def _formation_ids(lineup: tuple[str, ...], records: dict[str, dict[str, float | str]]) -> str:
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for player in lineup:
        position = str(records[player]["position"])
        if position in counts:
            counts[position] += 1
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def _lineup_key(decision: LineupDecision) -> tuple[float, str]:
    return (round(decision.objective, 10), ",".join(sorted(decision.lineup)))
