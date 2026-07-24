from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.decision.config import DECISION_REPORTS_DIR, load_decision_config
from fpl_forecast.decision.inputs import (
    assert_frozen_decisions_target_free,
    candidate_slice,
    load_decision_candidates,
)
from fpl_forecast.decision.expected_realized import evaluate_expected_realized_points
from fpl_forecast.decision.lineup import apply_autosubs_and_score, optimize_lineup
from fpl_forecast.decision.metrics import compare_models, decision_metrics, selected_player_calibration
from fpl_forecast.decision.milp import optimize_squad_expected_realized, optimize_squad_milp
from fpl_forecast.decision.rules import (
    assert_rules_match_config,
    default_rules,
    rules_from_bootstrap,
    validate_rules,
)
from fpl_forecast.decision.squad import squad_table
from fpl_forecast.decision.transfers import plan_multi_gameweek_transfers, plan_to_frame
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
            for optimizer_variant in config.optimizer_variants:
                solution = _optimize_squad_with_fallback(universe, rules, config, optimizer_variant=optimizer_variant)
                mean_only = optimize_lineup(
                    universe.set_index("player_uid").loc[list(solution.squad)].reset_index(),
                    rules,
                    appearance_aware=False,
                )
                selected = squad_table(solution, universe)
                diagnostics = dict(solution.diagnostics or {})
                if "expected_realized_total" not in diagnostics:
                    diagnostics.update(
                        evaluate_expected_realized_points(
                            selected,
                            solution.lineup_decision,
                            rules,
                            max_scenarios=config.expected_realized_scenarios,
                            seed=config.expected_realized_seed,
                        ).to_dict()
                    )
                scored_payload = apply_autosubs_and_score(
                    selected.rename(columns={"actual_points": "actual_points"}),
                    solution.lineup_decision,
                    rules,
                )
                decision_model = f"{model_name}:{optimizer_variant}"
                frozen_rows.append(
                    {
                        "season": row.season,
                        "gameweek": int(row.gameweek),
                        "model_name": model_name,
                        "optimizer_variant": optimizer_variant,
                        "decision_model": decision_model,
                        "objective": solution.objective,
                        "mean_only_objective": mean_only.objective,
                        "nominal_starting_xi_xpoints": diagnostics.get("nominal_starting_xi_xpoints"),
                        "expected_nominal_starting_xi_points": diagnostics.get("expected_nominal_starting_xi_points"),
                        "expected_active_starter_points": diagnostics.get("expected_active_starter_points"),
                        "expected_autosub_contribution": diagnostics.get("expected_autosub_contribution"),
                        "expected_captain_bonus": diagnostics.get("expected_captain_bonus"),
                        "expected_vice_captain_contingency": diagnostics.get("expected_vice_captain_contingency"),
                        "expected_realized_total": diagnostics.get("expected_realized_total", solution.objective),
                        "probability_all_starters_appear": diagnostics.get("probability_all_starters_appear"),
                        "expected_automatic_substitutions": diagnostics.get("expected_automatic_substitutions"),
                        "probability_unreplaced_starter": diagnostics.get("probability_unreplaced_starter"),
                        "expected_bench_points_used": diagnostics.get("expected_bench_points_used"),
                        "scenario_count": diagnostics.get("scenario_count"),
                        "probability_mass": diagnostics.get("probability_mass"),
                        "analytic_method": diagnostics.get("analytic_method"),
                        "cost_tenths": solution.cost_tenths,
                        "bank_tenths": solution.bank_tenths,
                        "squad": ",".join(solution.squad),
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
                        "objective_bound": solution.objective_bound,
                        "objective_gap": solution.objective_gap,
                        "solver_message": solution.solver_message,
                        "solver_nodes": solution.solver_nodes,
                        "raw_swap_proposals_generated": diagnostics.get("raw_swap_proposals_generated"),
                        "feasible_unique_squad_proposals": diagnostics.get("feasible_unique_squad_proposals"),
                        "squad_evaluation_calls": diagnostics.get("squad_evaluation_calls"),
                        "accepted_improving_moves": diagnostics.get("accepted_improving_moves"),
                        "local_search_iterations": diagnostics.get("local_search_iterations"),
                        "termination_reason": diagnostics.get("termination_reason"),
                    }
                )
                scored_rows.append(
                    {
                        **frozen_rows[-1],
                        "realized_points": scored_payload["realized_points"],
                        "captain_bonus_points": scored_payload["captain_bonus_points"],
                        "autosub_count": len(scored_payload["autosub_players"]),
                        "autosub_points": scored_payload["autosub_points"],
                        "unreplaced_starter_count": len(scored_payload["unreplaced_starters"]),
                        "captain_multiplier_player": scored_payload["captain_multiplier_player"],
                        "captain_fallback_used": scored_payload["captain_multiplier_player"]
                        == solution.lineup_decision.vice_captain,
                    }
                )
                selected = selected.assign(
                    season=row.season,
                    gameweek=int(row.gameweek),
                    model_name=model_name,
                    optimizer_variant=optimizer_variant,
                    decision_model=decision_model,
                )
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
        left=f"{config.default_model}:D2_EXPECTED_REALIZED_POINTS",
        right=f"{config.default_model}:D1_MEAN_ONLY_MILP",
    )
    metrics_paths = {
        "optimized_squads": _write_frame(squads, run_dir / "optimized_squads.csv"),
        "metrics": _write_frame(metrics, run_dir / "decision_metrics.csv"),
        "model_comparison": _write_frame(comparison, run_dir / "model_comparison.csv"),
        "selected_player_calibration": _write_frame(
            selected_player_calibration(squads),
            run_dir / "selected_player_calibration.csv",
        ),
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
        "solver": "scipy_highs_milp plus expected-realized local search",
        "optimality_scope": "D1 exact MILP benchmark and D2 deterministic expected-realized challenger",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return DecisionRunResult(run_id, run_dir, frozen_path, scored_path, metrics_paths, int(len(frozen)))


def compare_decisions(*, run_id: str, reports_dir: Path | str = DECISION_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    metrics = pd.read_csv(run_dir / "decision_metrics.csv")
    comparison = pd.read_csv(run_dir / "model_comparison.csv")
    lines = [f"run_id={run_id}", "decision_metrics:"]
    lines.extend(
        f"{row.decision_model}: decisions={int(row.decisions)} expected={row.mean_expected_score:.2f} "
        f"realized={row.mean_realized_points:.2f} captain_bonus={row.mean_captain_bonus:.2f} "
        f"bank={row.mean_bank_tenths:.1f}"
        for row in metrics.itertuples(index=False)
    )
    if not comparison.empty:
        row = comparison.iloc[0]
        lines.append(
            f"{row['left_model']}_vs_{row['right_model']}: mean_realized_difference={row['mean_realized_difference']:.2f} "
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
        f"optimizer_variants={','.join(manifest['config'].get('optimizer_variants', []))}",
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
    weekly_candidates = {
        1: candidates.copy(),
        2: _toy_transfer_frame(upgrade_shift=1.5),
    }
    plan = plan_multi_gameweek_transfers(
        current,
        weekly_candidates,
        rules,
        bank_tenths=10,
        free_transfers=1,
        max_transfers_per_gameweek=1,
    )
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return _write_frame(plan_to_frame(plan), run_dir / "transfer_plan.csv")


def forecast_decisions_guard(*, season: str, gameweek: int | None, as_of: str) -> None:
    current = load_current_fixture_frame(season=season, as_of=as_of)
    if gameweek is not None:
        current = current.loc[pd.to_numeric(current["gameweek"], errors="coerce").eq(gameweek)]
    if current.empty:
        raise ValueError("No forecastable current fixtures match the requested filters.")
    raise ValueError("Current decision optimization requires genuine current xPoints artifacts; none were written.")


def _optimize_squad_with_fallback(candidates: pd.DataFrame, rules, config, *, optimizer_variant: str):
    if optimizer_variant == "D1_MEAN_ONLY_MILP":
        return optimize_squad_milp(candidates, rules, appearance_aware=True)
    if optimizer_variant == "D2_EXPECTED_REALIZED_POINTS":
        return optimize_squad_expected_realized(
            candidates,
            rules,
            search_limit=config.expected_realized_search_limit,
            search_iterations=config.expected_realized_search_iterations,
            max_scenarios=config.expected_realized_scenarios,
            seed=config.expected_realized_seed,
        )
    raise ValueError(f"Unknown optimizer variant: {optimizer_variant}")


def _constraint_audit(frozen: pd.DataFrame, rules) -> pd.DataFrame:
    return frozen.assign(
        legal=(
            frozen["cost_tenths"].le(rules.budget_tenths)
            & frozen["bank_tenths"].ge(0)
            & frozen["captain"].ne(frozen["vice_captain"])
        )
    )[["season", "gameweek", "model_name", "cost_tenths", "bank_tenths", "captain", "vice_captain", "legal"]]


def _toy_transfer_frame(upgrade_shift: float = 0.0) -> pd.DataFrame:
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
                    "expected_points": float(index + 1 + (upgrade_shift if index >= count else 0.0)),
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
