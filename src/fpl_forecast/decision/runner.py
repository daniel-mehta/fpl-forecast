from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.decision.config import DECISION_REPORTS_DIR, load_decision_config
from fpl_forecast.decision.inputs import (
    assert_frozen_decisions_target_free,
    candidate_slice,
    load_decision_candidates,
)
from fpl_forecast.decision.lineup import apply_autosubs_and_score, optimize_lineup
from fpl_forecast.decision.metrics import compare_models, decision_metrics
from fpl_forecast.decision.rules import (
    assert_rules_match_config,
    default_rules,
    rules_from_bootstrap,
    validate_rules,
)
from fpl_forecast.decision.squad import SquadSolution, squad_table
from fpl_forecast.decision.transfers import plan_one_week_transfer
from fpl_forecast.panel.common import parse_seasons
from fpl_forecast.team_model.data import load_current_fixture_frame


@dataclass(frozen=True)
class DecisionRunResult:
    run_id: str
    run_dir: Path
    frozen_squads_path: Path
    scored_decisions_path: Path
    metrics_paths: dict[str, Path]
    decisions: int


def validate_decision_rules(*, bootstrap_path: Path | None = None) -> list[str]:
    rules = default_rules()
    warnings = validate_rules(rules)
    if bootstrap_path is not None:
        api_rules = rules_from_bootstrap(bootstrap_path)
        assert_rules_match_config(api_rules, rules)
    return warnings


def run_decision_backtest(
    *,
    seasons: str | list[str],
    mode: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
    reports_dir: Path | str = DECISION_REPORTS_DIR,
    run_id: str | None = None,
) -> DecisionRunResult:
    if mode not in {"gw1", "rolling"}:
        raise ValueError("mode must be 'gw1' or 'rolling'.")
    config = load_decision_config()
    rules = default_rules()
    season_list = parse_seasons(seasons)
    candidates = load_decision_candidates(mode=mode, config=config, normalized_dir=normalized_dir)
    candidates = candidates.loc[candidates["season"].isin(season_list)].copy()
    groups = candidates[["season", "gameweek"]].drop_duplicates().sort_values(["season", "gameweek"])
    if mode == "gw1":
        groups = groups.loc[groups["gameweek"].eq(1)]
    frozen_rows = []
    scored_rows = []
    squad_rows = []
    for row in groups.itertuples(index=False):
        for model_name in config.comparison_models:
            universe = candidate_slice(candidates, season=row.season, gameweek=int(row.gameweek), model_name=model_name)
            solution = _optimize_squad_with_fallback(universe, rules, config)
            mean_only = optimize_lineup(
                universe.set_index("player_uid").loc[list(solution.squad)].reset_index(),
                rules,
                appearance_aware=False,
            )
            selected = squad_table(solution, universe)
            scored_payload = apply_autosubs_and_score(
                selected.rename(columns={"actual_points": "actual_points"}),
                solution.lineup_decision,
                rules,
            )
            frozen_rows.append(
                {
                    "season": row.season,
                    "gameweek": int(row.gameweek),
                    "model_name": model_name,
                    "objective": solution.objective,
                    "mean_only_objective": mean_only.objective,
                    "cost_tenths": solution.cost_tenths,
                    "bank_tenths": solution.bank_tenths,
                    "lineup": ",".join(solution.lineup_decision.lineup),
                    "captain": solution.lineup_decision.captain,
                    "vice_captain": solution.lineup_decision.vice_captain,
                    "bench": ",".join(solution.lineup_decision.bench),
                    "formation": solution.lineup_decision.formation,
                    "solver_status": solution.solver_status,
                    "solver_name": solution.solver_name,
                    "candidate_count": solution.candidate_count,
                    "evaluated_squads": solution.evaluated_squads,
                    "runtime_seconds": solution.runtime_seconds,
                    "optimality_scope": solution.optimality_scope,
                }
            )
            scored_rows.append(
                {
                    **frozen_rows[-1],
                    "realized_points": scored_payload["realized_points"],
                    "captain_bonus_points": scored_payload["captain_bonus_points"],
                    "autosub_count": len(scored_payload["autosub_players"]),
                    "captain_multiplier_player": scored_payload["captain_multiplier_player"],
                }
            )
            selected = selected.assign(season=row.season, gameweek=int(row.gameweek), model_name=model_name)
            squad_rows.append(selected)
    frozen = pd.DataFrame(frozen_rows)
    assert_frozen_decisions_target_free(frozen, config.forbidden_frozen_columns)
    scored = pd.DataFrame(scored_rows)
    squads = pd.concat(squad_rows, ignore_index=True) if squad_rows else pd.DataFrame()
    run_id = run_id or f"phase7_decisions_{mode}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = _write_frame(frozen, run_dir / "frozen_decisions.csv")
    scored_path = _write_frame(scored, run_dir / "scored_decisions.csv")
    metrics = decision_metrics(scored)
    comparison = compare_models(
        scored,
        left="X2_TEAM_CONSTRAINED_SIM_M3",
        right="X2_TEAM_CONSTRAINED_SIM_M5",
    )
    metrics_paths = {
        "optimized_squads": _write_frame(squads, run_dir / "optimized_squads.csv"),
        "metrics": _write_frame(metrics, run_dir / "decision_metrics.csv"),
        "model_comparison": _write_frame(comparison, run_dir / "model_comparison.csv"),
        "constraint_audit": _write_frame(_constraint_audit(scored, rules), run_dir / "constraint_audit.csv"),
    }
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "seasons": season_list,
        "xpoints_run": config.xpoints_runs[mode],
        "rules": asdict(rules),
        "config": asdict(config),
        "decisions": int(len(frozen)),
        "git": _git_metadata(),
        "solver": "deterministic_greedy_repair_fallback",
        "optimality_scope": "legal deterministic greedy-repair squad with exact lineup search",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return DecisionRunResult(run_id, run_dir, frozen_path, scored_path, metrics_paths, int(len(frozen)))


def compare_decisions(*, run_id: str, reports_dir: Path | str = DECISION_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    metrics = pd.read_csv(run_dir / "decision_metrics.csv")
    comparison = pd.read_csv(run_dir / "model_comparison.csv")
    lines = [f"run_id={run_id}", "decision_metrics:"]
    lines.extend(
        f"{row.model_name}: decisions={int(row.decisions)} expected={row.mean_expected_score:.2f} "
        f"realized={row.mean_realized_points:.2f} captain_bonus={row.mean_captain_bonus:.2f} "
        f"bank={row.mean_bank_tenths:.1f}"
        for row in metrics.itertuples(index=False)
    )
    if not comparison.empty:
        row = comparison.iloc[0]
        lines.append(
            f"X2_M3_vs_X2_M5: mean_realized_difference={row['mean_realized_difference']:.2f} "
            f"captain_agreement={row['captain_agreement']:.3f}"
        )
    return lines


def inspect_decision_run(*, run_id: str, reports_dir: Path | str = DECISION_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen = pd.read_csv(run_dir / "frozen_decisions.csv")
    audit = pd.read_csv(run_dir / "constraint_audit.csv")
    return [
        f"run_id={run_id}",
        f"mode={manifest['mode']}",
        f"decisions={len(frozen)}",
        f"models={','.join(manifest['config']['comparison_models'])}",
        f"all_legal={bool(audit['legal'].all())}",
        f"max_cost_tenths={int(frozen['cost_tenths'].max())}",
        f"min_bank_tenths={int(frozen['bank_tenths'].min())}",
        f"dirty_worktree={manifest['git']['dirty']}",
    ]


def run_transfer_demo(*, reports_dir: Path | str = DECISION_REPORTS_DIR, run_id: str = "phase7_transfer_demo") -> Path:
    rules = default_rules()
    candidates = _toy_transfer_frame()
    current = pd.concat(
        [
            candidates.loc[candidates["fpl_position"].eq(position)].head(quota)
            for position, quota in rules.position_quotas.items()
        ],
        ignore_index=True,
    )
    plan = plan_one_week_transfer(current, candidates, rules, bank_tenths=10, free_transfers=1)
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return _write_frame(pd.DataFrame([asdict(plan)]), run_dir / "transfer_plan.csv")


def forecast_decisions_guard(*, season: str, gameweek: int | None, as_of: str) -> None:
    current = load_current_fixture_frame(season=season, as_of=as_of)
    if gameweek is not None:
        current = current.loc[pd.to_numeric(current["gameweek"], errors="coerce").eq(gameweek)]
    if current.empty:
        raise ValueError("No forecastable current fixtures match the requested filters.")
    raise ValueError("Current decision optimization requires genuine current xPoints artifacts; none were written.")


def _optimize_squad_with_fallback(candidates: pd.DataFrame, rules, config):
    del config
    return _greedy_repair_squad(candidates, rules)


def _greedy_repair_squad(candidates: pd.DataFrame, rules) -> SquadSolution:
    start = perf_counter()
    selected: list[str] = []
    team_counts: dict[str, int] = {}
    for position, quota in rules.position_quotas.items():
        group = candidates.loc[candidates["fpl_position"].eq(position)].sort_values(
            ["price_tenths", "expected_points", "player_uid"],
            ascending=[True, False, True],
        )
        for row in group.itertuples(index=False):
            team = str(row.player_team_uid)
            if team_counts.get(team, 0) >= rules.max_players_per_team:
                continue
            selected.append(str(row.player_uid))
            team_counts[team] = team_counts.get(team, 0) + 1
            if sum(1 for player in selected if _position(candidates, player) == position) == quota:
                break
    if len(selected) != rules.squad_size:
        raise ValueError("No legal cheap seed squad found for decision fallback.")
    frame = candidates.set_index("player_uid")
    upgrade_pool = pd.concat(
        [
            candidates.loc[candidates["player_uid"].isin(selected)],
            _decision_upgrade_pool(candidates),
        ],
        ignore_index=True,
    ).drop_duplicates("player_uid")
    selected = _improve_greedy_squad(selected, upgrade_pool, rules)
    squad = frame.loc[selected].reset_index()
    cost = int(squad["price_tenths"].sum())
    if cost > rules.budget_tenths or squad["player_team_uid"].value_counts().max() > rules.max_players_per_team:
        raise ValueError("Greedy fallback failed to produce a legal squad.")
    lineup = optimize_lineup(squad, rules, appearance_aware=True)
    return SquadSolution(
        squad=tuple(selected),
        lineup_decision=lineup,
        objective=lineup.objective,
        cost_tenths=cost,
        bank_tenths=rules.budget_tenths - cost,
        solver_status="heuristic_feasible",
        solver_name="deterministic_greedy_repair_fallback",
        candidate_count=int(len(candidates)),
        evaluated_squads=1,
        runtime_seconds=perf_counter() - start,
        optimality_scope="legal greedy-repair fallback after infeasible deterministic shortlist",
    )


def _improve_greedy_squad(selected: list[str], candidates: pd.DataFrame, rules) -> list[str]:
    frame = candidates.set_index("player_uid")
    selected_set = set(selected)
    improved = True
    while improved:
        improved = False
        current = frame.loc[selected].reset_index()
        cost = int(current["price_tenths"].sum())
        for incoming in candidates.sort_values(
            ["expected_points", "price_tenths", "player_uid"],
            ascending=[False, True, True],
        ).itertuples(index=False):
            incoming_id = str(incoming.player_uid)
            if incoming_id in selected_set:
                continue
            same_position = current.loc[current["fpl_position"].eq(incoming.fpl_position)].sort_values(
                ["expected_points", "price_tenths", "player_uid"],
                ascending=[True, False, True],
            )
            for outgoing in same_position.itertuples(index=False):
                if float(incoming.expected_points) <= float(outgoing.expected_points):
                    continue
                new_cost = cost - int(outgoing.price_tenths) + int(incoming.price_tenths)
                if new_cost > rules.budget_tenths:
                    continue
                trial_ids = [player for player in selected if player != outgoing.player_uid]
                trial_ids.append(incoming_id)
                trial = frame.loc[trial_ids].reset_index()
                if trial["player_team_uid"].value_counts().max() > rules.max_players_per_team:
                    continue
                selected = trial_ids
                selected_set = set(selected)
                improved = True
                break
            if improved:
                break
    return selected


def _position(candidates: pd.DataFrame, player_uid: str) -> str:
    return str(candidates.loc[candidates["player_uid"].eq(player_uid), "fpl_position"].iloc[0])


def _decision_upgrade_pool(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = candidates.copy()
    frame["_value_score"] = frame["expected_points"] / frame["price_tenths"].clip(lower=1)
    for _, group in frame.groupby("fpl_position", sort=False):
        expected = group.sort_values(
            ["expected_points", "price_tenths", "player_uid"],
            ascending=[False, True, True],
        ).head(12)
        value = group.sort_values(
            ["_value_score", "expected_points", "player_uid"],
            ascending=[False, False, True],
        ).head(12)
        cheap = group.sort_values(
            ["price_tenths", "expected_points", "player_uid"],
            ascending=[True, False, True],
        ).head(8)
        rows.append(pd.concat([expected, value, cheap], ignore_index=True))
    return pd.concat(rows, ignore_index=True).drop_duplicates("player_uid").drop(columns="_value_score")


def _constraint_audit(frozen: pd.DataFrame, rules) -> pd.DataFrame:
    return frozen.assign(
        legal=(
            frozen["cost_tenths"].le(rules.budget_tenths)
            & frozen["bank_tenths"].ge(0)
            & frozen["captain"].ne(frozen["vice_captain"])
        )
    )[["season", "gameweek", "model_name", "cost_tenths", "bank_tenths", "captain", "vice_captain", "legal"]]


def _toy_transfer_frame() -> pd.DataFrame:
    rows = []
    quotas = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    team_index = 0
    for position, count in quotas.items():
        for index in range(count + 2):
            price = {"GKP": 40, "DEF": 45, "MID": 55, "FWD": 60}[position]
            rows.append(
                {
                    "player_uid": f"{position}_{index}",
                    "player_name": f"{position} {index}",
                    "fpl_position": position,
                    "player_team_uid": f"team_{team_index % 8}",
                    "price_tenths": price,
                    "selling_price_tenths": price,
                    "expected_points": float(index + 1),
                    "p_appearance": 0.9,
                    "actual_points": float(index),
                    "actual_minutes": 90,
                }
            )
            team_index += 1
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def _git_metadata() -> dict[str, object]:
    return {"commit": _git(["rev-parse", "--short", "HEAD"]), "dirty": bool(_git(["status", "--short"]))}


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip()
