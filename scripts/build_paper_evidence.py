#!/usr/bin/env python3
"""Build the aggregate evidence and figures for the preseason technical paper.

The script reads existing repository artifacts only. It does not run models,
backtests, operational forecasts, or outcome ingestion.

PNG rendering uses the Node ``sharp`` package. Set ``NODE_PATH`` to a Node
module directory containing sharp when it is not installed in the active Node
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fpl_forecast.decision.evidence import (
    authoritative_decision_run,
    build_decision_evidence_table,
    decision_evidence_supersession_table,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
PROMOTED_SHA = "365f000"
GOALKEEPER_FIX_SHA = "668ece1"
LEGACY_DIRTY_SOURCE_IDENTITY = (
    "3ea29d9+tracked-diff-38fbd0457d1257029a8c6236f98ca611a1ba9ee78b7c9ab204f6b9f3ebdd71b4"
)
PROSPECTIVE_RUN = os.environ.get(
    "FPL_PAPER_PROSPECTIVE_RUN",
    "preseason_sim_hybrid_10000_goalkeeper_corrected_validation_clean_034830b041c1",
)
PROSPECTIVE_BASE = ROOT / "outputs/operational/validation_runs" / PROSPECTIVE_RUN
GOALKEEPER_EVIDENCE_INVENTORY = ROOT / os.environ.get(
    "FPL_PAPER_EVIDENCE_INVENTORY",
    "reports/goalkeeper_scoring_fix/clean_replay_inventory_034830b041c1.json",
)
CONVERGENCE_EVIDENCE = PROSPECTIVE_BASE / "preseason_simulation_convergence.json"
CLOSURE_EVIDENCE = PROSPECTIVE_BASE / "preseason_simulation_closure.json"

TEAM_RUN = "phase4_1_dixon_coles_rolling"
MINUTES_ROLLING_RUN = "phase9b12_minutes_rolling"
MINUTES_GW1_RUN = "phase9b12_minutes_gw1"
XPOINTS_ROLLING_RUN = "phase9b12_xpoints_rolling_goalkeeper_corrected_exact"
XPOINTS_GW1_RUN = "phase9b12_xpoints_gw1_goalkeeper_corrected_exact"
HYBRID_GW1_RUN = os.environ.get(
    "FPL_PAPER_HYBRID_GW1_RUN",
    "preseason_sim_hybrid_10000_gw1_three_fold_goalkeeper_corrected_clean_034830b041c1",
)
DECISION_RUN = os.environ.get(
    "FPL_PAPER_DECISION_RUN",
    authoritative_decision_run("table7_gw1"),
)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "magenta": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#202124",
    "gray": "#6B7280",
    "light": "#E5E7EB",
    "panel": "#F8FAFC",
    "white": "#FFFFFF",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a source file or a directory tree with stable repository-relative names."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def source_hashes(source_paths: str) -> str:
    identities = []
    for source in source_paths.split(";"):
        path = ROOT / source
        identities.append(f"{source}={sha256_path(path)}")
    return ";".join(identities)


def manifest_input_hashes(path: Path) -> str:
    """Return the explicit upstream identities recorded by a run manifest."""
    data = read_json(path)
    identities: list[str] = []
    source_data = data.get("source_data") or {}
    if source_data.get("fact_hash"):
        identities.append(f"fact_player_fixture={source_data['fact_hash']}")
    for key, value in sorted(source_data.items()):
        if key == "fact_hash" or not key.endswith("_hash") or not value:
            continue
        identities.append(f"{key.removesuffix('_hash')}={value}")
    if data.get("config_hash"):
        identities.append(f"config={data['config_hash']}")
    config_provenance = data.get("config_provenance") or {}
    if config_provenance.get("sha256"):
        identities.append(f"config={config_provenance['sha256']}")
    for name, artifact in sorted((data.get("input_artifacts") or {}).items()):
        if artifact.get("sha256"):
            identities.append(f"{name}={artifact['sha256']}")
    for name, artifact in sorted((data.get("official_snapshots") or {}).items()):
        if artifact.get("sha256"):
            identities.append(f"official_{name}={artifact['sha256']}")
    if data.get("original_manifest_sha256"):
        identities.append(f"original_manifest={data['original_manifest_sha256']}")
    return ";".join(dict.fromkeys(identities))


def manifest_code_identity(path: Path, *, legacy_dirty_fallback: str = "") -> str:
    """Return the strongest exact source identity recorded by a run manifest."""
    data = read_json(path)
    for state_key in ("source_state", "software_state"):
        state = data.get(state_key) or {}
        if state.get("source_tree_sha256"):
            return str(state["source_tree_sha256"])
    git = data.get("git") or {}
    commit = str(git.get("commit") or data.get("code_revision") or "")
    if git.get("dirty") or data.get("dirty_worktree"):
        if git.get("tracked_diff_sha256"):
            return f"{commit}+tracked-diff-{git['tracked_diff_sha256']}"
        if legacy_dirty_fallback:
            return legacy_dirty_fallback
        raise ValueError(f"Dirty source state lacks an exact digest: {rel(path)}")
    if commit:
        if len(commit) < 40:
            resolved = subprocess.run(
                ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if resolved:
                return resolved
        return commit
    if legacy_dirty_fallback:
        return legacy_dirty_fallback
    raise ValueError(f"Manifest lacks a source identity: {rel(path)}")


def combined_manifest_inputs(*paths: Path) -> str:
    return ";".join(filter(None, (manifest_input_hashes(path) for path in paths)))


def combined_code_identities(*identities: str) -> str:
    return ";".join(dict.fromkeys(identity for identity in identities if identity))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def manifest_meta(path: Path) -> dict[str, Any]:
    data = read_json(path)
    git = data.get("git") or {}
    folds = data.get("folds") or []
    return {
        "run_id": data.get("run_id", path.parent.name),
        "code_sha": git.get("commit") or data.get("code_revision") or "",
        "mode": data.get("mode", ""),
        "folds": len(folds),
        "seasons": ",".join(data.get("test_seasons") or data.get("seasons") or []),
    }


def table1_coverage() -> pd.DataFrame:
    source = ROOT / "data/normalized/phase2/fact_player_fixture.parquet"
    frame = pd.read_parquet(source)
    critical = [
        "player_uid",
        "fixture_key",
        "player_team_uid",
        "home_team_uid",
        "away_team_uid",
        "opponent_team_uid",
    ]
    key = ["season", "player_uid", "fixture_id"]

    population_path = (
        ROOT
        / "reports/xpoints_backtests"
        / HYBRID_GW1_RUN
        / "scored_player_fixture_predictions.parquet"
    )
    populations = pd.read_parquet(population_path)
    populations = populations.loc[populations["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M7")]
    pop_counts = (
        populations.groupby(["season", "pre_deadline_population"]).size().unstack(fill_value=0)
    )

    rows: list[dict[str, Any]] = []
    for season, group in frame.groupby("season", sort=True):
        positions = group["fpl_position"].value_counts()
        duplicates = int(group.duplicated(key, keep=False).sum())
        missing = int(group[critical].isna().any(axis=1).sum())
        pop = pop_counts.loc[season] if season in pop_counts.index else pd.Series(dtype=int)
        rows.append(
            {
                "season": season,
                "player_fixture_rows": len(group),
                "unique_players": group["player_uid"].nunique(),
                "unique_fixtures": group["fixture_key"].nunique(),
                "teams": group["player_team_uid"].nunique(),
                "gkp_rows": int(positions.get("GKP", 0)),
                "def_rows": int(positions.get("DEF", 0)),
                "mid_rows": int(positions.get("MID", 0)),
                "fwd_rows": int(positions.get("FWD", 0)),
                "assistant_manager_rows": int(positions.get("AM", 0)),
                "gw1_history_active_rows_where_available": (
                    int(pop.get("pre_deadline_history_active", 0)) if not pop.empty else ""
                ),
                "gw1_cold_start_rows_where_available": (
                    int(pop.get("cold_start_no_history", 0)) if not pop.empty else ""
                ),
                "duplicate_key_rows": duplicates,
                "missing_critical_id_rows": missing,
                "source": rel(source),
            }
        )
    return pd.DataFrame(rows)


def table2_team_models() -> pd.DataFrame:
    base = ROOT / "reports/team_backtests" / TEAM_RUN
    goals = pd.read_csv(base / "metrics_expected_goals.csv")
    clean = pd.read_csv(base / "metrics_clean_sheet.csv")
    outcomes = pd.read_csv(base / "metrics_match_outcome.csv")
    joint = pd.read_csv(base / "metrics_joint_score.csv")
    manifest = read_json(base / "manifest.json")
    out = goals.merge(
        clean[["model_name", "brier"]].rename(columns={"brier": "clean_sheet_brier"}),
        on="model_name",
    )
    out = out.merge(
        outcomes[["model_name", "multiclass_log_loss"]].rename(
            columns={"multiclass_log_loss": "outcome_log_loss"}
        ),
        on="model_name",
    ).merge(joint[["model_name", "joint_score_nll"]], on="model_name")
    out.insert(0, "comparison_block", "rolling_same_76_folds")
    out.insert(1, "selected", out["model_name"].eq("T2_REGULARIZED_ATTACK_DEFENCE"))
    out.insert(2, "seasons", ",".join(manifest["test_seasons"]))
    out.insert(3, "mode", manifest["mode"])
    out.insert(4, "folds", len(manifest["folds"]))
    out["metric_scope_note"] = np.where(
        out["model_name"].eq("T3_DIXON_COLES"),
        "T3 is a separately labelled low-score-dependence challenger; joint-score NLL is its "
        "distinctive diagnostic, although all rows share identical folds.",
        "Directly comparable T0-T2 rolling evidence.",
    )
    out["source_run_id"] = TEAM_RUN
    return out


def table3_minutes_models() -> pd.DataFrame:
    rows = []
    for block, run in [("rolling", MINUTES_ROLLING_RUN), ("historical_gw1", MINUTES_GW1_RUN)]:
        base = ROOT / "reports/minutes_backtests" / run
        overall = pd.read_csv(base / "metrics_overall.csv")
        binary = pd.read_csv(base / "metrics_binary.csv")
        meta = manifest_meta(base / "manifest.json")
        for model in [
            "M3_EWMA_MINUTES",
            "M5_REGULARIZED_STATE_SOFTMAX",
            "M7_HIERARCHICAL_AVAILABILITY_STATE",
        ]:
            metric = overall.loc[overall["model_name"].eq(model)].iloc[0]
            bins = binary.loc[binary["model_name"].eq(model)].set_index("target")
            rows.append(
                {
                    "comparison_block": block,
                    "source_run_id": run,
                    "model": model,
                    "seasons": meta["seasons"],
                    "evaluation_mode": meta["mode"],
                    "folds": meta["folds"],
                    "population": "all_observed_players",
                    "grain": "player_fixture",
                    "rows": int(metric["rows"]),
                    "mae": metric["mae"],
                    "rmse": metric["rmse"],
                    "bias": metric["bias"],
                    "appearance_brier": bins.loc["appearance", "brier"],
                    "start_brier": bins.loc["start", "brier"],
                    "spearman": metric["spearman"],
                }
            )
    return pd.DataFrame(rows)


def table4_xpoints_models() -> pd.DataFrame:
    rows = []
    specs = [
        ("rolling_legacy_80", XPOINTS_ROLLING_RUN, None),
        ("historical_gw1_legacy_80", XPOINTS_GW1_RUN, None),
        (
            "historical_gw1_promoted_hybrid",
            HYBRID_GW1_RUN,
            ["X2_TEAM_CONSTRAINED_SIM_M7"],
        ),
    ]
    for block, run, selected_models in specs:
        base = ROOT / "reports/xpoints_backtests" / run
        overall = pd.read_csv(base / "metrics_overall.csv")
        distribution = pd.read_csv(base / "metrics_distribution.csv")
        meta = manifest_meta(base / "manifest.json")
        merged = overall.merge(
            distribution[["model_name", "prob_ge_5_brier", "central_80_coverage"]],
            on="model_name",
        )
        if selected_models is not None:
            merged = merged.loc[merged["model_name"].isin(selected_models)]
        for metric in merged.itertuples(index=False):
            rows.append(
                {
                    "comparison_block": block,
                    "source_run_id": run,
                    "model": metric.model_name,
                    "simulator": (
                        "preseason_hybrid_fixture_v1"
                        if block.endswith("promoted_hybrid")
                        else "phase9b12_component_sim_v2"
                    ),
                    "seasons": meta["seasons"],
                    "mode": meta["mode"],
                    "population": "all_observed_players",
                    "grain": "player_gameweek",
                    "rows": int(metric.rows),
                    "folds": meta["folds"],
                    "mae": metric.mae,
                    "rmse": metric.rmse,
                    "bias": metric.bias,
                    "spearman": metric.spearman,
                    "p5_brier": metric.prob_ge_5_brier,
                    "central_80_coverage": metric.central_80_coverage,
                }
            )
    return pd.DataFrame(rows)


def table5_convergence() -> pd.DataFrame:
    convergence = read_json(CONVERGENCE_EVIDENCE)
    closure = read_json(CLOSURE_EVIDENCE)
    rows: list[dict[str, Any]] = []
    for draw_count in (80, 500, 1000, 5000, 10000):
        result = convergence["results"][str(draw_count)]
        rows.append(
            {
                "draw_count": draw_count,
                "distribution_reference_draws": convergence["reference_draw_count"],
                "runtime_seconds": result["runtime_seconds"],
                "draw_matrix_mib": result["draw_matrix_megabytes"],
                "median_mean_abs_difference": result["analytic_vs_simulated_mean_median_abs"],
                "p95_mean_abs_difference": result["analytic_vs_simulated_mean_p95_abs"],
                "p95_p5_abs_difference": result["reference_p5_p95_abs"],
                "p95_zero_probability_abs_difference": result["reference_zero_p95_abs"],
                "p95_interval_endpoint_abs_difference": result[
                    "reference_interval_endpoint_p95_abs"
                ],
                "rank_correlation": result["rank_spearman"],
                "top_k_overlap": (
                    f"{int(result['top_15_overlap'] * 15)}/15;"
                    f"{int(result['top_30_overlap'] * 30)}/30;"
                    f"{int(result['top_50_overlap'] * 50)}/50"
                ),
                "deterministic_reproducibility": (
                    result["reproducible_draws"] and result["reproducible_summary"]
                ),
                "comparison_block": "initial",
                "process_peak_rss_mib": result["process_peak_rss_megabytes"],
                "source": rel(CONVERGENCE_EVIDENCE),
            }
        )
    rows.extend(
        [
            {
                "draw_count": 10000,
                "distribution_reference_draws": 20000,
                "runtime_seconds": np.nan,
                "draw_matrix_mib": closure["draw_matrix_megabytes"]["10000"],
                "median_mean_abs_difference": closure["simulated_mean_median_abs"],
                "p95_mean_abs_difference": closure["simulated_mean_p95_abs"],
                "p95_p5_abs_difference": closure["p5_p95_abs"],
                "p95_zero_probability_abs_difference": closure["zero_p95_abs"],
                "p95_interval_endpoint_abs_difference": closure["interval_endpoint_p95_abs"],
                "rank_correlation": closure["simulated_mean_spearman"],
                "top_k_overlap": ";".join(
                    f"{closure['top_overlap'][str(k)]}/{k}" for k in (15, 30, 50)
                ),
                "deterministic_reproducibility": closure["deterministic"],
                "comparison_block": "closure",
                "process_peak_rss_mib": np.nan,
                "source": rel(CLOSURE_EVIDENCE),
            },
            {
                "draw_count": 20000,
                "distribution_reference_draws": 20000,
                "runtime_seconds": np.nan,
                "draw_matrix_mib": closure["draw_matrix_megabytes"]["20000"],
                "median_mean_abs_difference": 0.0,
                "p95_mean_abs_difference": 0.0,
                "p95_p5_abs_difference": 0.0,
                "p95_zero_probability_abs_difference": 0.0,
                "p95_interval_endpoint_abs_difference": 0.0,
                "rank_correlation": 1.0,
                "top_k_overlap": "reference",
                "deterministic_reproducibility": closure["deterministic"],
                "comparison_block": "closure",
                "process_peak_rss_mib": np.nan,
                "source": rel(CLOSURE_EVIDENCE),
            },
        ]
    )
    out = pd.DataFrame(rows)
    out["tolerance_note"] = "median mean<=0.02; P95 mean<=0.05; P95 P(5+)<=0.01; exact reruns"
    return out


def xpoints_fold_metrics(path: Path, simulator_label: str) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data = data.loc[data["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M7")].copy()
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [
        *[(str(season), group) for season, group in data.groupby("season", sort=True)],
        ("pooled", data),
    ]
    for fold, group in groups:
        actual = group["actual_total_points"].astype(float)
        predicted = group["expected_points"].astype(float)
        actual_p5 = actual.ge(5).astype(float)
        coverage = group["pre_deadline_population"].value_counts()
        rows.append(
            {
                "fold": fold,
                "simulator": simulator_label,
                "rows": len(group),
                "history_active_rows": int(coverage.get("pre_deadline_history_active", 0)),
                "cold_start_rows": int(coverage.get("cold_start_no_history", 0)),
                "mae": (predicted - actual).abs().mean(),
                "rmse": math.sqrt(((predicted - actual) ** 2).mean()),
                "bias": (predicted - actual).mean(),
                "spearman": predicted.corr(actual, method="spearman"),
                "p5_brier": ((group["prob_points_ge_5"] - actual_p5) ** 2).mean(),
                "predicted_zero_rate": group["prob_points_eq_0"].mean(),
                "actual_zero_rate": actual.eq(0).mean(),
                "central_80_coverage": (
                    actual.ge(group["points_p10"]) & actual.le(group["points_p90"])
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def table6_old_hybrid() -> pd.DataFrame:
    base = ROOT / "reports/xpoints_backtests"
    old = xpoints_fold_metrics(
        base / XPOINTS_GW1_RUN / "scored_player_fixture_predictions.parquet",
        "old_80_draw_m7",
    )
    new = xpoints_fold_metrics(
        base / HYBRID_GW1_RUN / "scored_player_fixture_predictions.parquet",
        "hybrid_10000",
    )
    out = pd.concat([old, new], ignore_index=True)
    out["population"] = "all_observed_players"
    out["grain"] = "player_gameweek"
    out["source_run_id"] = np.where(
        out["simulator"].eq("old_80_draw_m7"), XPOINTS_GW1_RUN, HYBRID_GW1_RUN
    )
    return out


def table7_decisions() -> pd.DataFrame:
    base = ROOT / "reports/decision_backtests" / DECISION_RUN
    return build_decision_evidence_table(base, DECISION_RUN)


def table8_snapshot() -> pd.DataFrame:
    base = PROSPECTIVE_BASE
    projections = pd.read_csv(base / "player_gameweek_projections.csv")
    squad = pd.read_csv(base / "optimized_squad.csv")
    lineup = pd.read_csv(base / "optimized_lineup.csv").iloc[0]
    freshness = read_json(base / "data_freshness.json")
    lineage = read_json(base / "model_lineage.json")
    top = projections.sort_values(
        ["expected_points", "stable_player_id"], ascending=[False, True]
    ).head(15)
    selected_names = squad.sort_values(
        ["selected_role", "bench_order", "player_name"], na_position="last"
    )["player_name"].tolist()
    summary = {
        "record_type": "snapshot_summary",
        "rank": "",
        "player": "",
        "team": "",
        "position": "",
        "price_tenths": "",
        "expected_points": "",
        "appearance_probability": "",
        "run_id": PROSPECTIVE_RUN,
        "forecast_timestamp": freshness["generated_at"],
        "official_retrieval_timestamp": max(
            item["retrieved_at"] for item in freshness["official_snapshots"].values()
        ),
        "projection_rows": len(projections),
        "players": projections["stable_player_id"].nunique(),
        "fixtures": freshness["target_fixture_count"],
        "team_model": lineage["team_model"],
        "minutes_model": "M7_HIERARCHICAL_AVAILABILITY_STATE",
        "xpoints_model": "X2_TEAM_CONSTRAINED_SIM_M7",
        "simulator_version": lineage["xpoints_simulator"]["version"],
        "draw_count": lineage["xpoints_simulator"]["production_draw_count"],
        "optimizer": lineage["decision_optimizer"],
        "formation": lineup["formation"],
        "captain": squad.loc[squad["selected_role"].eq("captain"), "player_name"].iloc[0],
        "vice_captain": squad.loc[squad["selected_role"].eq("vice_captain"), "player_name"].iloc[0],
        "expected_realized_value": lineup["expected_realized_total"],
        "solver_status": lineup["solver_status"],
        "total_cost_tenths": lineup["cost_tenths"],
        "bank_tenths": lineup["bank_tenths"],
        "recommended_squad": "; ".join(selected_names),
        "label": "Prospective example, not an accuracy result",
    }
    rows = [summary]
    for rank, player in enumerate(top.itertuples(index=False), start=1):
        row = {key: "" for key in summary}
        row.update(
            {
                "record_type": "top15_expected_points",
                "rank": rank,
                "player": player.player,
                "team": player.team,
                "position": player.position,
                "price_tenths": player.price_tenths,
                "expected_points": player.expected_points,
                "appearance_probability": player.p_appearance,
                "run_id": PROSPECTIVE_RUN,
                "label": "Prospective example, not an accuracy result",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def publication_evidence_supersession_table() -> pd.DataFrame:
    columns = [
        "evidence_scope",
        "authoritative_run_id",
        "superseded_run_id",
        "superseded_status",
        "reason",
    ]
    out = decision_evidence_supersession_table().copy()
    inventory = read_json(GOALKEEPER_EVIDENCE_INVENTORY)
    scope_by_label = {
        "Phase 6 rolling xPoints": "phase6_rolling_xpoints",
        "Phase 6 GW1 xPoints": "phase6_gw1_xpoints",
        "Phase 9B1.2 rolling xPoints": "table4_rolling_xpoints",
        "Phase 9B1.2 GW1 xPoints": "table4_table6_legacy80_gw1",
        "Promoted hybrid GW1 xPoints": "table4_table6_hybrid_gw1",
        "Rolling D1 decisions": "rolling_benchmark",
        "Table 7 GW1 D1 and D2 decisions": "table7_gw1",
    }
    extra_rows: list[dict[str, str]] = []
    existing_pairs = set(zip(out["authoritative_run_id"], out["superseded_run_id"], strict=False))
    for pair in inventory["evidence_pairs"]:
        authoritative = Path(pair["corrected_successor"]["run_dir"]).name
        superseded = Path(pair["original"]["run_dir"]).name
        if (authoritative, superseded) in existing_pairs:
            continue
        extra_rows.append(
            {
                "evidence_scope": scope_by_label[pair["label"]],
                "authoritative_run_id": authoritative,
                "superseded_run_id": superseded,
                "superseded_status": pair["original_preservation"],
                "reason": pair["supersession_reason"],
            }
        )
    validation_supersedes = inventory["public_validation"]["supersedes"]
    extra_rows.extend(
        [
            {
                "evidence_scope": "table5_simulation_convergence",
                "authoritative_run_id": f"{PROSPECTIVE_RUN}_simulation_convergence",
                "superseded_run_id": "preseason_simulation_convergence",
                "superseded_status": "immutable_historical_record",
                "reason": "Generated from prospective projections that used six-point goalkeeper goals.",
            },
            {
                "evidence_scope": "table5_simulation_closure",
                "authoritative_run_id": f"{PROSPECTIVE_RUN}_simulation_closure",
                "superseded_run_id": "preseason_simulation_closure_10000_vs_20000",
                "superseded_status": "immutable_historical_record",
                "reason": "Generated from prospective projections that used six-point goalkeeper goals.",
            },
            *[
                {
                    "evidence_scope": "table8_prospective_validation",
                    "authoritative_run_id": PROSPECTIVE_RUN,
                    "superseded_run_id": item["run_id"],
                    "superseded_status": "immutable_historical_record",
                    "reason": item["reason"],
                }
                for item in validation_supersedes
            ],
        ]
    )
    if extra_rows:
        out = pd.concat([out, pd.DataFrame(extra_rows, columns=columns)], ignore_index=True)
    return out.loc[:, columns].sort_values(
        ["evidence_scope", "superseded_run_id"], ignore_index=True
    )


@dataclass
class Svg:
    width: int = 1200
    height: int = 700

    def __post_init__(self) -> None:
        self.items: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "none",
        sw: float = 1,
        rx: float = 0,
        opacity: float = 1,
    ) -> None:
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'opacity="{opacity}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = COLORS["black"],
        sw: float = 2,
        dash: str = "",
    ) -> None:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{dashed}/>'
        )

    def circle(
        self,
        x: float,
        y: float,
        r: float,
        fill: str,
        stroke: str = COLORS["white"],
        sw: float = 1,
        opacity: float = 1,
    ) -> None:
        self.items.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def polyline(
        self,
        points: Iterable[tuple[float, float]],
        stroke: str,
        sw: float = 3,
        fill: str = "none",
        dash: str = "",
    ) -> None:
        value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<polyline points="{value}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{dashed}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 20,
        anchor: str = "start",
        weight: int = 400,
        fill: str = COLORS["black"],
        rotate: float | None = None,
    ) -> None:
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.items.append(
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{transform}>{html.escape(str(value))}</text>'
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = COLORS["gray"],
    ) -> None:
        self.line(x1, y1, x2, y2, stroke, 3)
        angle = math.atan2(y2 - y1, x2 - x1)
        wing = 10
        for delta in (2.6, -2.6):
            self.line(
                x2,
                y2,
                x2 + wing * math.cos(angle + delta),
                y2 + wing * math.sin(angle + delta),
                stroke,
                3,
            )

    def title(self, value: str, subtitle: str = "") -> None:
        self.text(55, 45, value, 28, weight=700)
        if subtitle:
            self.text(55, 72, subtitle, 16, fill=COLORS["gray"])

    def render(self, description: str) -> str:
        content = "\n".join(self.items)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            f'role="img" aria-label="{html.escape(description)}">\n'
            f'<rect width="100%" height="100%" fill="{COLORS["white"]}"/>\n'
            f"{content}\n</svg>\n"
        )


def axes(svg: Svg, x: float, y: float, w: float, h: float, title: str) -> None:
    svg.text(x, y - 12, title, 17, weight=700)
    svg.line(x, y, x, y + h, COLORS["gray"], 1)
    svg.line(x, y + h, x + w, y + h, COLORS["gray"], 1)


def horizontal_bars(
    svg: Svg,
    x: float,
    y: float,
    w: float,
    h: float,
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    lower_better: bool,
) -> None:
    axes(svg, x, y, w, h, f"{title} ({'lower' if lower_better else 'higher'} is better)")
    finite = [value for value in values if np.isfinite(value)]
    maximum = max(finite) * 1.15 if finite else 1
    row_h = h / max(len(values), 1)
    for i, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        yy = y + i * row_h + row_h * 0.2
        if not np.isfinite(value):
            svg.text(x - 8, yy + row_h * 0.42, label, 13, anchor="end")
            svg.text(x + 6, yy + row_h * 0.42, "NA (constant predictions)", 13)
            continue
        bw = value / maximum * w
        svg.rect(x, yy, bw, row_h * 0.56, color, rx=3)
        svg.text(x - 8, yy + row_h * 0.42, label, 13, anchor="end")
        svg.text(x + bw + 6, yy + row_h * 0.42, f"{value:.3f}", 13)


def figure1_architecture() -> str:
    svg = Svg(1200, 680)
    svg.title(
        "Figure 1. Leakage-safe forecast and decision architecture",
        "Frozen outcomes are evaluation targets only; they are not forecast inputs.",
    )
    nodes = [
        (55, 145, 165, 105, "Official +\nhistorical data", COLORS["sky"]),
        (255, 145, 165, 105, "Leakage-safe\nplayer-fixture panel", COLORS["panel"]),
        (460, 105, 155, 82, "T2\nteam state", COLORS["blue"]),
        (460, 215, 155, 82, "M7\navailability", COLORS["orange"]),
        (660, 145, 175, 105, "X2 hybrid\nplayer outcomes", COLORS["green"]),
        (875, 145, 135, 105, "D2\ndecisions", COLORS["magenta"]),
        (1040, 145, 120, 105, "Validated\nforecast", COLORS["panel"]),
    ]
    for x, y, w, h, label, color in nodes:
        svg.rect(x, y, w, h, color, COLORS["gray"], 1, 8, 0.18 if color != COLORS["panel"] else 1)
        parts = label.split("\n")
        for j, part in enumerate(parts):
            svg.text(x + w / 2, y + h / 2 - 8 + j * 24, part, 17, "middle", 700)
    svg.arrow(220, 198, 255, 198)
    svg.arrow(420, 185, 460, 146)
    svg.arrow(420, 215, 460, 256)
    svg.arrow(615, 146, 660, 180)
    svg.arrow(615, 256, 660, 215)
    svg.arrow(835, 198, 875, 198)
    svg.arrow(1010, 198, 1040, 198)
    lanes = [
        (
            430,
            350,
            COLORS["blue"],
            "Team state",
            "Poisson scoring intensities and shared scoreline",
        ),
        (
            430,
            415,
            COLORS["orange"],
            "Player availability",
            "DNP, substitute and starter minute states",
        ),
        (
            430,
            480,
            COLORS["green"],
            "Player outcomes",
            "Analytic means + 10,000 joint fixture draws",
        ),
        (
            430,
            545,
            COLORS["magenta"],
            "Decision optimization",
            "Exact 32,768 appearance states for each squad",
        ),
    ]
    for x, y, color, label, detail in lanes:
        svg.circle(x, y, 8, color)
        svg.text(x + 20, y + 6, label, 17, weight=700)
        svg.text(x + 205, y + 6, detail, 16, fill=COLORS["gray"])
    svg.rect(55, 565, 295, 62, COLORS["panel"], COLORS["vermillion"], 2, 8)
    svg.text(202, 590, "Official outcomes", 17, "middle", 700)
    svg.text(202, 614, "Scoring only after fixtures complete", 14, "middle", fill=COLORS["gray"])
    svg.arrow(350, 596, 430, 596, COLORS["vermillion"])
    svg.text(445, 602, "Prospective evaluation", 16, weight=700, fill=COLORS["vermillion"])
    return svg.render(
        "Architecture from official and historical data through T2, M7, X2, D2 and publication."
    )


def figure2_timeline() -> str:
    svg = Svg(1200, 680)
    svg.title(
        "Figure 2. Chronological evaluation design",
        "Historical evidence and the prospective 2026-27 forecast have different roles.",
    )
    seasons = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
    xs = [120, 330, 540, 750, 960]
    svg.line(110, 190, 1080, 190, COLORS["gray"], 3)
    for x, season in zip(xs, seasons, strict=True):
        color = COLORS["vermillion"] if season == "2025-26" else COLORS["blue"]
        if season == "2026-27":
            color = COLORS["green"]
        svg.circle(x, 190, 12, color)
        svg.text(x, 155, season, 18, "middle", 700)
    for x, label in zip(
        xs[1:4],
        ["Rolling test folds", "Rolling + GW1 test folds", "Rolling + GW1; hardening evidence"],
        strict=True,
    ):
        svg.rect(x - 90, 240, 180, 72, COLORS["panel"], COLORS["gray"], 1, 7)
        for j, part in enumerate(label.split("; ")):
            svg.text(x, 268 + j * 22, part, 14, "middle", 700 if j == 0 else 400)
    svg.rect(90, 370, 650, 70, COLORS["sky"], COLORS["blue"], 1, 8, 0.18)
    svg.text(415, 398, "Expanding historical training windows", 18, "middle", 700)
    svg.text(415, 424, "Every fold uses only information available before its cutoff", 15, "middle")
    svg.arrow(740, 405, 845, 405, COLORS["blue"])
    svg.rect(845, 355, 260, 100, COLORS["green"], COLORS["green"], 2, 8, 0.14)
    svg.text(975, 386, "2026-27 GW1", 19, "middle", 700)
    svg.text(975, 414, "Frozen prospective forecast", 16, "middle")
    svg.text(975, 438, "No outcomes observed", 15, "middle", fill=COLORS["green"])
    svg.rect(675, 505, 300, 88, COLORS["panel"], COLORS["vermillion"], 2, 8)
    svg.text(825, 535, "2025-26 is not an untouched holdout", 17, "middle", 700)
    svg.text(825, 563, "Its results informed preseason hardening.", 15, "middle")
    svg.text(825, 585, "Report it as historical development evidence.", 15, "middle")
    return svg.render(
        "Timeline of historical training, rolling validation, GW1 folds and prospective 2026-27 forecast."
    )


def figure3_team(table: pd.DataFrame) -> str:
    svg = Svg(1200, 780)
    svg.title(
        "Figure 3. Rolling team-model comparison",
        "Same 76 folds and 760 fixtures; T3 is shown separately as a challenger.",
    )
    main = table.loc[~table["model_name"].eq("T3_DIXON_COLES")]
    labels = ["T0", "T1", "T2"]
    colors = [COLORS["gray"], COLORS["sky"], COLORS["blue"]]
    metrics = [
        ("goal_mae", "Goal MAE"),
        ("poisson_nll", "Poisson NLL"),
        ("clean_sheet_brier", "Clean-sheet Brier"),
        ("outcome_log_loss", "Outcome log loss"),
    ]
    for idx, (column, title) in enumerate(metrics):
        row, col = divmod(idx, 2)
        horizontal_bars(
            svg,
            175 + col * 565,
            125 + row * 275,
            330,
            190,
            labels,
            main[column].tolist(),
            colors,
            title,
            True,
        )
    t2 = table.loc[table["model_name"].eq("T2_REGULARIZED_ATTACK_DEFENCE")].iloc[0]
    t3 = table.loc[table["model_name"].eq("T3_DIXON_COLES")].iloc[0]
    svg.rect(720, 660, 425, 78, COLORS["panel"], COLORS["orange"], 2, 8)
    svg.text(735, 686, "T3 challenger (same folds; not selected)", 16, weight=700)
    svg.text(
        735,
        714,
        f"Joint NLL T2 {t2.joint_score_nll:.4f} vs T3 {t3.joint_score_nll:.4f}",
        15,
    )
    svg.text(
        175,
        705,
        "Selected model: T2 regularized attack-defence",
        17,
        weight=700,
        fill=COLORS["blue"],
    )
    return svg.render(
        "Four panels compare T0, T1 and selected T2 rolling team metrics, with T3 separately noted."
    )


def figure4_xpoints(table: pd.DataFrame) -> str:
    svg = Svg(1200, 800)
    svg.title(
        "Figure 4. Historical xPoints comparisons",
        "Rolling: 114 folds across 2023-24 to 2025-26; GW1: 3 folds. All use observed players.",
    )
    mapping = {
        "X0_PHASE3_B5_EB_POINTS_PER90": "X0",
        "X1_INDEPENDENT_COMPONENT_RATES_M3": "X1",
        "X2_TEAM_CONSTRAINED_SIM_M3": "X2-M3",
        "X2_TEAM_CONSTRAINED_SIM_M5": "X2-M5",
        "X2_TEAM_CONSTRAINED_SIM_M7": "X2-M7",
    }
    colors = [COLORS["gray"], COLORS["sky"], COLORS["blue"], COLORS["orange"], COLORS["green"]]
    panels = [
        ("rolling_legacy_80", "Rolling MAE", "mae", True),
        ("rolling_legacy_80", "Rolling Spearman", "spearman", False),
        ("historical_gw1_legacy_80", "Historical GW1 MAE (80 draws)", "mae", True),
        ("historical_gw1_legacy_80", "Historical GW1 Spearman (80 draws)", "spearman", False),
    ]
    for idx, (block, title, metric, lower) in enumerate(panels):
        subset = table.loc[table["comparison_block"].eq(block)].copy()
        subset["short"] = subset["model"].map(mapping)
        row, col = divmod(idx, 2)
        vals = subset[metric].tolist()
        horizontal_bars(
            svg,
            175 + col * 565,
            120 + row * 300,
            320,
            215,
            subset["short"].tolist(),
            vals,
            colors[: len(vals)],
            title,
            lower,
        )
    hybrid = table.loc[table["comparison_block"].eq("historical_gw1_promoted_hybrid")].iloc[0]
    svg.rect(660, 695, 485, 70, COLORS["green"], COLORS["green"], 2, 8, 0.12)
    svg.text(675, 722, "Promoted hybrid GW1 block (same 3 folds):", 16, weight=700)
    svg.text(
        675,
        748,
        f"X2-M7 MAE {hybrid.mae:.3f}; Spearman {hybrid.spearman:.3f}; not a rolling result",
        15,
    )
    svg.text(
        175,
        742,
        "GW1 X0 Spearman is undefined because every prediction is zero.",
        13,
        fill=COLORS["gray"],
    )
    return svg.render("Separate rolling and GW1 panels compare xPoints models on MAE and Spearman.")


def figure5_convergence(table: pd.DataFrame) -> str:
    svg = Svg(1200, 760)
    svg.title(
        "Figure 5. Hybrid simulation convergence",
        "Initial points compare with 10k; the selected 10k closure point compares with 20k.",
    )
    initial = table.loc[table["comparison_block"].eq("initial")].copy()
    closure = table.loc[
        table["comparison_block"].eq("closure") & table["draw_count"].eq(10000)
    ].iloc[0]
    panels = [
        ("p95_mean_abs_difference", "P95 simulated-mean difference", 0.05),
        ("p95_p5_abs_difference", "P95 P(5+) difference", 0.01),
        ("runtime_seconds", "Runtime (seconds)", None),
    ]
    for idx, (column, title, tolerance) in enumerate(panels):
        x, y, w, h = 105 + idx * 375, 150, 300, 420
        axes(svg, x, y, w, h, title)
        values = initial[column].tolist()
        if column != "runtime_seconds":
            values[-1] = float(closure[column])
        reference_runtime = None
        if column == "runtime_seconds":
            recorded_runtime = table.loc[
                table["comparison_block"].eq("closure") & table["draw_count"].eq(20000),
                column,
            ].iloc[0]
            if pd.notna(recorded_runtime):
                reference_runtime = float(recorded_runtime)
        ymax_values = values + ([tolerance] if tolerance else [])
        if reference_runtime is not None:
            ymax_values.append(reference_runtime)
        ymax = max(ymax_values) * 1.2
        points = []
        for draws, value in zip(initial["draw_count"], values, strict=True):
            xx = x + (math.log10(draws) - math.log10(80)) / (math.log10(20000) - math.log10(80)) * w
            yy = y + h - value / ymax * h
            points.append((xx, yy))
            color = COLORS["green"] if draws == 10000 else COLORS["blue"]
            svg.circle(xx, yy, 6, color)
            svg.text(
                xx,
                y + h + 24,
                f"{int(draws / 1000)}k" if draws >= 1000 else str(draws),
                12,
                "middle",
            )
            value_label_y = yy + 18 if tolerance is not None and value < tolerance else yy - 10
            svg.text(xx, value_label_y, f"{value:.3g}", 11, "middle", fill=COLORS["gray"])
        if reference_runtime is not None:
            xx = x + w
            yy = y + h - reference_runtime / ymax * h
            points.append((xx, yy))
            svg.circle(xx, yy, 6, COLORS["orange"])
            svg.text(xx, y + h + 24, "20k", 12, "middle")
            svg.text(xx, yy - 10, f"{reference_runtime:.3g}", 11, "middle", fill=COLORS["gray"])
        svg.polyline(points, COLORS["blue"], 3)
        if tolerance is not None:
            ty = y + h - tolerance / ymax * h
            svg.line(x, ty, x + w, ty, COLORS["vermillion"], 2, "7 5")
            svg.text(x + 8, ty - 7, f"tolerance {tolerance:.2f}", 12, fill=COLORS["vermillion"])
    svg.rect(340, 625, 520, 70, COLORS["panel"], COLORS["green"], 2, 8)
    svg.text(600, 653, "10,000 selected; 20,000 used only as closure reference", 17, "middle", 700)
    svg.text(600, 680, "10k vs 20k: P95 mean 0.0307, P95 P(5+) 0.00525", 15, "middle")
    return svg.render(
        "Log draw-count plots of mean convergence, tail probability convergence and runtime."
    )


def figure6_tradeoffs(table: pd.DataFrame) -> str:
    svg = Svg(1200, 760)
    svg.title(
        "Figure 6. Pooled historical GW1 simulator trade-offs",
        "Hybrid 10,000 draws relative to old 80 draws; arrows show the desired direction.",
    )
    pooled = table.loc[table["fold"].eq("pooled")].set_index("simulator")
    old = pooled.loc["old_80_draw_m7"]
    new = pooled.loc["hybrid_10000"]
    metrics = [
        ("MAE ↓", old.mae, new.mae, False),
        ("RMSE ↓", old.rmse, new.rmse, False),
        ("|Bias| ↓", abs(old.bias), abs(new.bias), False),
        ("Spearman ↑", old.spearman, new.spearman, True),
        ("P(5+) Brier ↓", old.p5_brier, new.p5_brier, False),
        ("80% coverage ↑", old.central_80_coverage, new.central_80_coverage, True),
    ]
    for i, (label, old_value, new_value, higher) in enumerate(metrics):
        row, col = divmod(i, 3)
        x, y = 70 + col * 380, 130 + row * 275
        svg.rect(x, y, 340, 215, COLORS["panel"], COLORS["light"], 1, 8)
        svg.text(x + 18, y + 30, label, 17, weight=700)
        maximum = max(old_value, new_value) * 1.2 or 1
        for j, (name, value, color) in enumerate(
            [("Old 80", old_value, COLORS["gray"]), ("Hybrid 10k", new_value, COLORS["green"])]
        ):
            yy = y + 65 + j * 65
            svg.text(x + 18, yy + 18, name, 14)
            svg.rect(x + 105, yy, value / maximum * 190, 26, color, rx=3)
            svg.text(x + 305, yy + 19, f"{value:.4f}", 13, "end")
        improved = new_value > old_value if higher else new_value < old_value
        svg.text(
            x + 18,
            y + 195,
            "improved" if improved else "worsened",
            14,
            weight=700,
            fill=COLORS["green"] if improved else COLORS["vermillion"],
        )
    return svg.render(
        "Six small multiples show old and hybrid pooled GW1 metrics, including worsened MAE."
    )


def p5_calibration(path: Path, label: str) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data = data.loc[data["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M7")].copy()
    bins = np.linspace(0, 1, 11)
    data["bin"] = pd.cut(
        data["prob_points_ge_5"],
        bins=bins,
        include_lowest=True,
        right=True,
        labels=False,
    )
    data["actual_p5"] = data["actual_total_points"].ge(5).astype(float)
    out = (
        data.groupby("bin", observed=False)
        .agg(
            rows=("actual_p5", "size"),
            mean_prediction=("prob_points_ge_5", "mean"),
            observed_rate=("actual_p5", "mean"),
        )
        .reset_index()
    )
    out["model"] = label
    out["bin_lower"] = out["bin"] / 10
    out["bin_upper"] = (out["bin"] + 1) / 10
    return out


def figure7_calibration(old: pd.DataFrame, new: pd.DataFrame) -> str:
    svg = Svg(1200, 760)
    svg.title(
        "Figure 7. Historical GW1 P(5+) calibration",
        "Fixed probability bins, identical three folds, all observed players.",
    )
    x, y, w, h = 150, 115, 520, 520
    axes(svg, x, y, w, h, "Observed frequency")
    for tick in np.linspace(0, 1, 6):
        xx, yy = x + tick * w, y + h - tick * h
        svg.line(x, yy, x + w, yy, COLORS["light"], 1)
        svg.text(x - 12, yy + 5, f"{tick:.1f}", 13, "end")
        svg.text(xx, y + h + 25, f"{tick:.1f}", 13, "middle")
    svg.line(x, y + h, x + w, y, COLORS["gray"], 2, "7 5")
    for frame, color, label in [
        (old, COLORS["gray"], "Old 80"),
        (new, COLORS["green"], "Hybrid 10k"),
    ]:
        points = []
        for row in frame.itertuples(index=False):
            if row.rows == 0 or pd.isna(row.mean_prediction):
                continue
            xx = x + row.mean_prediction * w
            yy = y + h - row.observed_rate * h
            points.append((xx, yy))
            svg.circle(xx, yy, 4 + min(row.rows, 1000) ** 0.5 / 7, color, opacity=0.75)
        svg.polyline(points, color, 3)
        svg.circle(760, 170 + (0 if label == "Old 80" else 38), 7, color)
        svg.text(780, 176 + (0 if label == "Old 80" else 38), label, 16)
    svg.text(x + w / 2, y + h + 58, "Mean predicted P(5+)", 16, "middle", 700)
    svg.text(95, y + h / 2, "Observed P(5+)", 16, "middle", 700, rotate=-90)
    svg.rect(740, 265, 385, 230, COLORS["panel"], COLORS["light"], 1, 8)
    svg.text(760, 294, "Bin sizes (old / hybrid)", 16, weight=700)
    merged = old[["bin", "rows"]].merge(new[["bin", "rows"]], on="bin", suffixes=("_old", "_new"))
    for i, row in enumerate(merged.itertuples(index=False)):
        svg.text(
            760 + (i // 5) * 180,
            324 + (i % 5) * 31,
            f"{row.bin / 10:.1f}-{(row.bin + 1) / 10:.1f}: {row.rows_old}/{row.rows_new}",
            13,
        )
    svg.text(760, 535, "Point area is proportional to bin sample size.", 14, fill=COLORS["gray"])
    svg.text(760, 562, "Perfect calibration is the dashed diagonal.", 14, fill=COLORS["gray"])
    return svg.render(
        "P5 calibration plot compares old and hybrid simulators in fixed bins with sample sizes."
    )


def figure8_price_xpoints() -> str:
    path = PROSPECTIVE_BASE / "player_gameweek_projections.csv"
    data = pd.read_csv(path)
    svg = Svg(1200, 780)
    svg.title(
        "Figure 8. Prospective 2026-27 GW1 snapshot, not evaluated against outcomes",
        "Official price versus frozen expected points; opacity reflects appearance probability.",
    )
    x, y, w, h = 110, 120, 930, 540
    axes(svg, x, y, w, h, "Expected points")
    xmin, xmax = data["price_tenths"].min() / 10, data["price_tenths"].max() / 10
    ymax = max(7.0, data["expected_points"].max() * 1.08)
    position_colors = {
        "GKP": COLORS["orange"],
        "DEF": COLORS["blue"],
        "MID": COLORS["green"],
        "FWD": COLORS["magenta"],
    }
    for tick in np.arange(math.floor(xmin), math.ceil(xmax) + 1, 2):
        xx = x + (tick - xmin) / (xmax - xmin) * w
        svg.text(xx, y + h + 25, f"£{tick:.0f}m", 13, "middle")
    for tick in range(0, int(math.ceil(ymax)) + 1):
        yy = y + h - tick / ymax * h
        svg.line(x, yy, x + w, yy, COLORS["light"], 1)
        svg.text(x - 10, yy + 5, str(tick), 13, "end")
    for row in data.itertuples(index=False):
        xx = x + (row.price_tenths / 10 - xmin) / (xmax - xmin) * w
        yy = y + h - row.expected_points / ymax * h
        svg.circle(
            xx,
            yy,
            3.5 + 3 * float(row.p_appearance),
            position_colors.get(row.position, COLORS["gray"]),
            opacity=0.18 + 0.72 * float(row.p_appearance),
        )
    labels = data.nlargest(7, "expected_points")
    label_offsets = {
        "B.Fernandes": (8, -13),
        "Watkins": (8, -25),
        "Mbeumo": (8, -25),
        "Saka": (8, -13),
        "Haaland": (8, 24),
        "Gabriel": (8, -13),
        "Eze": (8, 24),
    }
    for row in labels.itertuples(index=False):
        xx = x + (row.price_tenths / 10 - xmin) / (xmax - xmin) * w
        yy = y + h - row.expected_points / ymax * h
        dx, dy = label_offsets.get(row.player, (8, -12))
        svg.text(xx + dx, yy + dy, row.player, 12, weight=700)
    for i, (position, color) in enumerate(position_colors.items()):
        svg.circle(1075, 165 + i * 34, 7, color)
        svg.text(1092, 171 + i * 34, position, 14)
    svg.text(x + w / 2, y + h + 62, "Official price", 16, "middle", 700)
    svg.text(52, y + h / 2, "Expected points", 16, "middle", 700, rotate=-90)
    return svg.render(
        "Prospective price versus expected points scatter by position and appearance probability."
    )


def render_png(svg_path: Path, png_path: Path) -> None:
    script = """
const sharp = require('sharp');
const [src, dst] = process.argv.slice(1);
sharp(src, { density: 144 })
  .resize({ width: 2400 })
  .png({ compressionLevel: 9, adaptiveFiltering: false, palette: false })
  .withMetadata({ density: 300 })
  .toFile(dst)
  .catch(error => { console.error(error); process.exit(1); });
"""
    subprocess.run(
        ["node", "-e", script, str(svg_path), str(png_path)],
        cwd=ROOT,
        check=True,
        env={**os.environ, "TZ": "UTC"},
    )


def write_figure(number: int, slug: str, content: str) -> tuple[Path, Path]:
    stem = f"figure{number}_{slug}"
    svg_path = FIGURES / f"{stem}.svg"
    png_path = FIGURES / f"{stem}.png"
    svg_path.write_text(content, encoding="utf-8", newline="\n")
    render_png(svg_path, png_path)
    return svg_path, png_path


def evidence_rows(
    tables: dict[str, pd.DataFrame],
    figure_paths: dict[str, tuple[Path, Path]],
) -> pd.DataFrame:
    columns = [
        "evidence_id",
        "figure_or_table_id",
        "source_path",
        "source_run_id",
        "code_sha",
        "upstream_input_sha256",
        "source_output_sha256",
        "model",
        "seasons",
        "mode",
        "population",
        "grain",
        "rows",
        "folds",
        "metric",
        "direction",
        "prospective_or_historical",
        "notes_and_limitations",
    ]
    rows: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        row = {column: "" for column in columns}
        row.update(kwargs)
        rows.append(row)

    team_manifest = ROOT / f"reports/team_backtests/{TEAM_RUN}/manifest.json"
    minutes_rolling_manifest = (
        ROOT / f"reports/minutes_backtests/{MINUTES_ROLLING_RUN}/manifest.json"
    )
    minutes_gw1_manifest = ROOT / f"reports/minutes_backtests/{MINUTES_GW1_RUN}/manifest.json"
    xpoints_rolling_manifest = (
        ROOT / f"reports/xpoints_backtests/{XPOINTS_ROLLING_RUN}/manifest.json"
    )
    xpoints_gw1_manifest = ROOT / f"reports/xpoints_backtests/{XPOINTS_GW1_RUN}/manifest.json"
    hybrid_manifest = ROOT / f"reports/xpoints_backtests/{HYBRID_GW1_RUN}/manifest.json"
    decision_manifest = ROOT / f"reports/decision_backtests/{DECISION_RUN}/manifest.json"
    prospective_manifest = PROSPECTIVE_BASE / "validation_manifest.json"
    hybrid_code_identity = manifest_code_identity(
        hybrid_manifest,
        legacy_dirty_fallback=LEGACY_DIRTY_SOURCE_IDENTITY,
    )
    prospective_code_identity = manifest_code_identity(
        prospective_manifest,
        legacy_dirty_fallback=LEGACY_DIRTY_SOURCE_IDENTITY,
    )
    table_sources = {
        "Table 1": (
            "data/normalized/phase2/fact_player_fixture.parquet;"
            f"reports/xpoints_backtests/{HYBRID_GW1_RUN}/scored_player_fixture_predictions.parquet",
            f"phase2_normalized_panel;{HYBRID_GW1_RUN}",
            hybrid_code_identity,
            combined_manifest_inputs(hybrid_manifest),
            "historical",
        ),
        "Table 2": (
            f"reports/team_backtests/{TEAM_RUN}/metrics_expected_goals.csv;"
            f"reports/team_backtests/{TEAM_RUN}/metrics_clean_sheet.csv;"
            f"reports/team_backtests/{TEAM_RUN}/metrics_match_outcome.csv;"
            f"reports/team_backtests/{TEAM_RUN}/metrics_joint_score.csv",
            TEAM_RUN,
            "cfbbead",
            combined_manifest_inputs(team_manifest),
            "historical",
        ),
        "Table 3": (
            f"reports/minutes_backtests/{MINUTES_ROLLING_RUN}/metrics_overall.csv;"
            f"reports/minutes_backtests/{MINUTES_GW1_RUN}/metrics_overall.csv",
            f"{MINUTES_ROLLING_RUN};{MINUTES_GW1_RUN}",
            "ddd77a0",
            combined_manifest_inputs(minutes_rolling_manifest, minutes_gw1_manifest),
            "historical",
        ),
        "Table 4": (
            f"reports/xpoints_backtests/{XPOINTS_ROLLING_RUN}/metrics_overall.csv;"
            f"reports/xpoints_backtests/{XPOINTS_GW1_RUN}/metrics_overall.csv;"
            f"reports/xpoints_backtests/{HYBRID_GW1_RUN}/metrics_overall.csv",
            f"{XPOINTS_ROLLING_RUN};{XPOINTS_GW1_RUN};{HYBRID_GW1_RUN}",
            f"6877097;8dc8991;{hybrid_code_identity}",
            combined_manifest_inputs(
                xpoints_rolling_manifest, xpoints_gw1_manifest, hybrid_manifest
            ),
            "historical",
        ),
        "Table 5": (
            f"{rel(CONVERGENCE_EVIDENCE)};{rel(CLOSURE_EVIDENCE)}",
            "preseason_simulation_convergence_goalkeeper_corrected;"
            "preseason_simulation_closure_goalkeeper_corrected",
            prospective_code_identity,
            combined_manifest_inputs(prospective_manifest),
            "historical",
        ),
        "Table 6": (
            f"reports/xpoints_backtests/{XPOINTS_GW1_RUN}/"
            "scored_player_fixture_predictions.parquet;"
            f"reports/xpoints_backtests/{HYBRID_GW1_RUN}/"
            "scored_player_fixture_predictions.parquet",
            f"{XPOINTS_GW1_RUN};{HYBRID_GW1_RUN}",
            f"8dc8991;{hybrid_code_identity}",
            combined_manifest_inputs(xpoints_gw1_manifest, hybrid_manifest),
            "historical",
        ),
        "Table 7": (
            f"reports/decision_backtests/{DECISION_RUN}/decision_metrics.csv;"
            f"reports/decision_backtests/{DECISION_RUN}/model_comparison.csv;"
            f"reports/decision_backtests/{DECISION_RUN}/scored_decisions.csv",
            DECISION_RUN,
            read_json(decision_manifest)["software_state"]["source_tree_sha256"],
            combined_manifest_inputs(decision_manifest),
            "historical",
        ),
        "Table 8": (
            f"{rel(PROSPECTIVE_BASE / 'player_gameweek_projections.csv')};"
            f"{rel(PROSPECTIVE_BASE / 'optimized_squad.csv')};"
            f"{rel(PROSPECTIVE_BASE / 'optimized_lineup.csv')};"
            f"{rel(PROSPECTIVE_BASE / 'data_freshness.json')};"
            f"{rel(PROSPECTIVE_BASE / 'model_lineage.json')}",
            PROSPECTIVE_RUN,
            prospective_code_identity,
            combined_manifest_inputs(prospective_manifest),
            "prospective",
        ),
    }
    table_metrics = {
        "Table 1": "coverage and integrity counts",
        "Table 2": "goal MAE/RMSE/bias/NLL; clean-sheet Brier; outcome log loss",
        "Table 3": "minutes MAE/RMSE/bias; appearance/start Brier; Spearman",
        "Table 4": "xPoints MAE/RMSE/bias/Spearman; P(5+) Brier; 80% coverage",
        "Table 5": "simulation convergence, runtime, memory and reproducibility",
        "Table 6": "old versus hybrid three-fold GW1 metrics",
        "Table 7": "D1/D2 expected-realized and realized decision evidence",
        "Table 8": "prospective GW1 aggregate and top-ranked snapshot",
    }
    for idx, table_id in enumerate(table_sources, start=1):
        source, run, sha, upstream_inputs, evidence_type = table_sources[table_id]
        frame = tables[f"table{idx}"]
        add(
            evidence_id=f"E-T{idx:02d}",
            figure_or_table_id=table_id,
            source_path=source,
            source_run_id=run,
            code_sha=sha,
            upstream_input_sha256=upstream_inputs,
            source_output_sha256=source_hashes(source),
            model="multiple; see table" if idx != 8 else "T2/M7/X2 hybrid/D2",
            seasons="multiple; see table" if idx != 8 else "2026-27",
            mode="multiple; see comparison_block"
            if idx in {3, 4}
            else ("prospective_gw1" if idx == 8 else "historical"),
            population="explicit in table",
            grain="explicit in table",
            rows=len(frame),
            folds="explicit in table",
            metric=table_metrics[table_id],
            direction="metric-specific; explicit in figures/notes",
            prospective_or_historical=evidence_type,
            notes_and_limitations=(
                "Prospective example, not an accuracy result."
                if idx == 8
                else "No population or evaluation-mode pooling beyond explicitly labelled blocks."
            ),
        )

    figure_series = [
        (
            "Figure 1",
            "architecture",
            rel(PROSPECTIVE_BASE / "model_lineage.json"),
            PROSPECTIVE_RUN,
            prospective_code_identity,
            combined_manifest_inputs(prospective_manifest),
            "conceptual",
            "neither",
            "historical+prospective",
        ),
        (
            "Figure 2",
            "evaluation chronology",
            f"{rel(xpoints_rolling_manifest)};{rel(hybrid_manifest)};{rel(CONVERGENCE_EVIDENCE)};{rel(CLOSURE_EVIDENCE)}",
            f"{XPOINTS_ROLLING_RUN};{HYBRID_GW1_RUN};preseason_simulation_convergence_goalkeeper_corrected;preseason_simulation_closure_goalkeeper_corrected",
            combined_code_identities(
                "6877097", hybrid_code_identity, prospective_code_identity
            ),
            combined_manifest_inputs(
                xpoints_rolling_manifest, hybrid_manifest, prospective_manifest
            ),
            "conceptual",
            "neither",
            "historical+prospective",
        ),
        (
            "Figure 3",
            "T0/T1/T2 comparable rolling metrics",
            f"reports/team_backtests/{TEAM_RUN}/metrics_expected_goals.csv;reports/team_backtests/{TEAM_RUN}/metrics_clean_sheet.csv;reports/team_backtests/{TEAM_RUN}/metrics_match_outcome.csv",
            TEAM_RUN,
            "cfbbead",
            combined_manifest_inputs(team_manifest),
            "goal/NLL/Brier/log loss",
            "lower-is-better",
            "historical",
        ),
        (
            "Figure 3",
            "T3 challenger joint NLL",
            f"reports/team_backtests/{TEAM_RUN}/metrics_joint_score.csv",
            TEAM_RUN,
            "cfbbead",
            combined_manifest_inputs(team_manifest),
            "joint score NLL",
            "lower-is-better",
            "historical",
        ),
        (
            "Figure 4",
            "rolling xPoints MAE and Spearman",
            f"reports/xpoints_backtests/{XPOINTS_ROLLING_RUN}/metrics_overall.csv",
            XPOINTS_ROLLING_RUN,
            "6877097",
            combined_manifest_inputs(xpoints_rolling_manifest),
            "MAE/Spearman",
            "mixed",
            "historical",
        ),
        (
            "Figure 4",
            "GW1 xPoints MAE and Spearman",
            f"reports/xpoints_backtests/{XPOINTS_GW1_RUN}/metrics_overall.csv",
            XPOINTS_GW1_RUN,
            "8dc8991",
            combined_manifest_inputs(xpoints_gw1_manifest),
            "MAE/Spearman",
            "mixed",
            "historical",
        ),
        (
            "Figure 4",
            "hybrid GW1 annotation",
            f"reports/xpoints_backtests/{HYBRID_GW1_RUN}/metrics_overall.csv",
            HYBRID_GW1_RUN,
            hybrid_code_identity,
            combined_manifest_inputs(hybrid_manifest),
            "MAE/Spearman",
            "mixed",
            "historical",
        ),
        (
            "Figure 5",
            "P95 simulated mean difference",
            f"{rel(CONVERGENCE_EVIDENCE)};{rel(CLOSURE_EVIDENCE)}",
            "preseason_simulation_convergence_goalkeeper_corrected;preseason_simulation_closure_goalkeeper_corrected",
            prospective_code_identity,
            combined_manifest_inputs(prospective_manifest),
            "P95 mean difference",
            "lower-is-better",
            "historical",
        ),
        (
            "Figure 5",
            "P95 P(5+) difference",
            f"{rel(CONVERGENCE_EVIDENCE)};{rel(CLOSURE_EVIDENCE)}",
            "preseason_simulation_convergence_goalkeeper_corrected;preseason_simulation_closure_goalkeeper_corrected",
            prospective_code_identity,
            combined_manifest_inputs(prospective_manifest),
            "P95 P(5+) difference",
            "lower-is-better",
            "historical",
        ),
        (
            "Figure 5",
            "runtime",
            rel(CONVERGENCE_EVIDENCE),
            "preseason_simulation_convergence_goalkeeper_corrected",
            prospective_code_identity,
            combined_manifest_inputs(prospective_manifest),
            "seconds",
            "lower-is-better subject to tolerances",
            "historical",
        ),
        (
            "Figure 6",
            "old 80 pooled metrics",
            f"reports/xpoints_backtests/{XPOINTS_GW1_RUN}/scored_player_fixture_predictions.parquet",
            XPOINTS_GW1_RUN,
            "8dc8991",
            combined_manifest_inputs(xpoints_gw1_manifest),
            "six pooled metrics",
            "mixed",
            "historical",
        ),
        (
            "Figure 6",
            "hybrid 10k pooled metrics",
            f"reports/xpoints_backtests/{HYBRID_GW1_RUN}/scored_player_fixture_predictions.parquet",
            HYBRID_GW1_RUN,
            hybrid_code_identity,
            combined_manifest_inputs(hybrid_manifest),
            "six pooled metrics",
            "mixed",
            "historical",
        ),
        (
            "Figure 7",
            "old P(5+) calibration",
            f"reports/xpoints_backtests/{XPOINTS_GW1_RUN}/scored_player_fixture_predictions.parquet",
            XPOINTS_GW1_RUN,
            "8dc8991",
            combined_manifest_inputs(xpoints_gw1_manifest),
            "fixed-bin P(5+) calibration",
            "closer-to-diagonal",
            "historical",
        ),
        (
            "Figure 7",
            "hybrid P(5+) calibration",
            f"reports/xpoints_backtests/{HYBRID_GW1_RUN}/scored_player_fixture_predictions.parquet",
            HYBRID_GW1_RUN,
            hybrid_code_identity,
            combined_manifest_inputs(hybrid_manifest),
            "fixed-bin P(5+) calibration",
            "closer-to-diagonal",
            "historical",
        ),
        (
            "Figure 8",
            "prospective price/xPoints by position",
            rel(PROSPECTIVE_BASE / "player_gameweek_projections.csv"),
            PROSPECTIVE_RUN,
            prospective_code_identity,
            combined_manifest_inputs(prospective_manifest),
            "price and expected points",
            "descriptive",
            "prospective",
        ),
    ]
    for index, (
        fig,
        series,
        source,
        run,
        sha,
        upstream_inputs,
        metric,
        direction,
        evidence_type,
    ) in enumerate(figure_series, start=1):
        add(
            evidence_id=f"E-F{index:02d}",
            figure_or_table_id=fig,
            source_path=source,
            source_run_id=run,
            code_sha=sha,
            upstream_input_sha256=upstream_inputs,
            source_output_sha256=source_hashes(source),
            model=series,
            seasons="2022-23 to 2026-27 as labelled",
            mode="explicit in figure",
            population="explicit in figure and notes",
            grain="series-specific",
            rows="see source/table",
            folds="see source/table",
            metric=metric,
            direction=direction,
            prospective_or_historical=evidence_type,
            notes_and_limitations=(
                "Prospective snapshot; no outcome or accuracy interpretation."
                if evidence_type == "prospective"
                else "Traceable full-precision source; limitations in FIGURE_NOTES.md."
            ),
        )
    return pd.DataFrame(rows, columns=columns)


def figure_notes() -> str:
    return f"""# Figure and Table Notes

This file supplies proposed captions and interpretation boundaries for the preseason technical
paper. All paths are repository-relative. Historical and prospective evidence must remain visually
and narratively separate.

## Table 1 — Historical data coverage

- **Caption:** Leakage-safe historical player-fixture panel coverage and integrity checks by season.
- **Exact source:** `data/normalized/phase2/fact_player_fixture.parquet`; GW1 population counts, where
  available, come from the promoted-hybrid three-fold scored artifact.
- **Population and grain:** All normalized football-player and assistant-manager fixture rows;
  player-fixture grain.
- **Supports:** The scale, identity completeness, positional composition and duplicate-key status of
  the historical evidence base.
- **Does not support:** Forecast accuracy or completeness of players absent from the source.
- **Limitation:** Cold-start/history-active counts are available only for evaluated historical GW1
  seasons and are not counts over the full season panel.
- **Suggested placement:** Data section.

## Table 2 — Team-model comparison

- **Caption:** Rolling team-model metrics on the same 76 deadline blocks and 760 fixtures.
- **Exact source:** `reports/team_backtests/phase4_1_dixon_coles_rolling/`.
- **Population and grain:** Historical Premier League fixtures; fixture and fixture-team-side
  metrics as labelled.
- **Supports:** Selection of T2 over T0/T1 and the negligible, mixed T3 challenger changes.
- **Does not support:** Player-level or prospective accuracy.
- **Limitation:** T3's joint-scoreline NLL is its distinctive dependence diagnostic; do not imply
  that it is a separate population or that tiny differences establish superiority.
- **Suggested placement:** Team-model methods/results section.

## Table 3 — Minutes-model comparison

- **Caption:** M3, M5 and M7 minutes and availability performance, with rolling and historical GW1
  evaluation shown as separate blocks.
- **Exact source:** `reports/minutes_backtests/phase9b12_minutes_rolling/` and
  `reports/minutes_backtests/phase9b12_minutes_gw1/`.
- **Population and grain:** All observed player-fixture rows.
- **Supports:** Different error/calibration trade-offs and the operational role of M7.
- **Does not support:** A claim that M7 wins every historical metric.
- **Limitation:** Aggregate all-observed metrics include many non-participants; GW1 has only three
  folds.
- **Suggested placement:** Availability-model results section.

## Table 4 — xPoints comparison

- **Caption:** Historical xPoints metrics in explicitly separated rolling, legacy GW1 and promoted
  hybrid GW1 blocks.
- **Exact source:** The three season-aware goalkeeper-scoring xPoints run directories listed in the
  table. Their immutable predecessors are classified in `paper/evidence_supersession.csv`.
- **Population and grain:** All observed player-gameweeks.
- **Supports:** Historical model trade-offs and the scope of available hybrid evidence.
- **Does not support:** Direct comparison of rolling legacy metrics with a hybrid simulator that was
  rerun only on GW1 folds.
- **Limitation:** No promoted-hybrid rolling backtest was run for this evidence pack.
- **Suggested placement:** xPoints results section.

## Table 5 — Simulation convergence

- **Caption:** Deterministic hybrid-simulation convergence, runtime and memory evidence.
- **Exact source:** `outputs/operational/validation_runs/
  {PROSPECTIVE_RUN}/
  preseason_simulation_convergence.json` and `preseason_simulation_closure.json`.
- **Population and grain:** Current official challenger input, 554 player-fixture rows.
- **Supports:** The declared-tolerance justification for 10,000 production draws.
- **Does not support:** Predictive superiority from increasing draw count.
- **Limitation:** Early distribution differences use 10,000 as reference; the closure row alone uses
  the non-self-referential 20,000 reference. The corrected closure artifact did not record closure
  wall-clock time or process peak RSS, so those cells are intentionally blank rather than copied
  from superseded evidence.
- **Suggested placement:** Simulation methods/validation section.

## Table 6 — Old versus hybrid GW1 folds

- **Caption:** Regression and stability comparison of the old 80-draw and promoted hybrid simulator
  across three historical GW1 folds.
- **Exact source:** Full-precision scored parquets for
  `phase9b12_xpoints_gw1_goalkeeper_corrected_exact` and
  `{HYBRID_GW1_RUN}`.
- **Population and grain:** All observed player-gameweeks, separately by fold and pooled.
- **Supports:** Honest disclosure of improved distribution stability and worsened MAE.
- **Does not support:** Broad predictive superiority.
- **Limitation:** Three GW1 folds are a small development sample; 2025-26 informed hardening.
- **Suggested placement:** Simulator validation section.

## Table 7 — Decision-system evidence

- **Caption:** D1 and D2 historical weekly-reset GW1 decision evidence.
- **Exact source:**
  `reports/decision_backtests/{DECISION_RUN}/`.
- **Population and grain:** One frozen squad decision per historical GW1; three decisions per method.
- **Supports:** Exact paired arithmetic and limited acceptance evidence for D2.
- **Does not support:** Strong statistical or season-long decision superiority.
- **Limitation:** The three-block bootstrap interval is highly unstable; D2 is heuristic rather than
  globally optimal.
- **Supersession:** Pre-fix and earlier corrected runs remain immutable historical records; the
  authoritative mapping and reasons are in `paper/evidence_supersession.csv`.
- **Suggested placement:** Decision-layer validation section.

## Table 8 — Current official GW1 prospective snapshot

- **Caption:** Prospective example, not an accuracy result: frozen 2026-27 GW1 model and decision
  snapshot.
- **Exact source:** `outputs/operational/validation_runs/
  {PROSPECTIVE_RUN}/`.
- **Population and grain:** Officially selectable current players; player-gameweek projections and
  one squad decision.
- **Supports:** A concrete example of forecast outputs, lineage and the recommended decision.
- **Does not support:** Any accuracy, calibration or realized-points claim.
- **Limitation:** This is a non-published validation successor. The frozen published predecessor
  remains immutable; the corrected run changes two bench squad members and the bench order while
  retaining the starting XI, formation, captain and vice-captain.
- **Suggested placement:** Prospective example box near the conclusion.

## Figure 1 — System architecture

- **Caption:** Leakage-safe data flow from official and historical inputs through T2 team state, M7
  availability, X2 hybrid player outcomes, D2 decisions and validated publication.
- **Exact source:** Model contracts and the season-aware validation successor's `model_lineage.json`.
- **Population and grain:** Conceptual system diagram.
- **Supports:** Separation of team, player-availability, player-outcome and decision layers.
- **Does not support:** Accuracy or causal attribution.
- **Limitation:** It omits lower-level feature engineering and fallback branches for readability.
- **Suggested placement:** Methods overview.

## Figure 2 — Chronological evaluation design

- **Caption:** Historical training/evaluation chronology and the transition to the frozen prospective
  2026-27 GW1 forecast.
- **Exact source:** Corrected xPoints manifests and corrected convergence/closure evidence listed in
  `paper/evidence_manifest.csv`.
- **Population and grain:** Historical season/gameweek folds plus one prospective forecast.
- **Supports:** Leakage-safe cutoff design and the distinction between development and prospective
  evidence.
- **Does not support:** Treating 2025-26 as an untouched final holdout.
- **Limitation:** The timeline is schematic rather than proportional to fixture counts.
- **Suggested placement:** Evaluation-design section.

## Figure 3 — Team-model comparison

- **Caption:** Comparable rolling goal, clean-sheet and outcome metrics for T0, T1 and selected T2;
  T3 is separately labelled as a low-score-dependence challenger.
- **Exact source:** `reports/team_backtests/phase4_1_dixon_coles_rolling/`.
- **Population and grain:** Same 76 folds; fixture sides for goal/clean-sheet metrics and fixtures for
  outcome metrics.
- **Supports:** T2 selection and the absence of material T3 improvement.
- **Does not support:** Cross-league generalization.
- **Limitation:** Axes differ by metric; compare models within a panel only.
- **Suggested placement:** Team-model results.

## Figure 4 — Historical xPoints comparison

- **Caption:** xPoints MAE and rank correlation in separate rolling and historical GW1 panels.
- **Visible scope:** The rolling panels contain 114 folds across `2023-24` to `2025-26`; the GW1
  panels contain three folds over the same seasons.
- **Exact source:** `phase9b12_xpoints_rolling_goalkeeper_corrected_exact`,
  `phase9b12_xpoints_gw1_goalkeeper_corrected_exact`, and
  `{HYBRID_GW1_RUN}`.
- **Population and grain:** All observed player-gameweeks.
- **Supports:** Model trade-offs under each registered evaluation mode.
- **Does not support:** Comparing the hybrid GW1 annotation as if it were a rolling result.
- **Limitation:** No uncertainty interval was available consistently across all plotted series.
- **Suggested placement:** xPoints results.

## Figure 5 — Simulation convergence

- **Caption:** Error stabilization and runtime as joint fixture draws increase; 10,000 is selected and
  20,000 is the closure reference.
- **Exact source:** The corrected convergence and 10,000-versus-20,000 closure JSON artifacts listed
  in `paper/evidence_manifest.csv`.
- **Population and grain:** 554 current player-fixture rows.
- **Supports:** Production draw-count selection against predeclared tolerances.
- **Does not support:** Improved football forecasting skill.
- **Limitation:** Initial and closure points have explicitly different distribution references; the
  corrected closure artifact did not retain closure runtime or process peak RSS.
- **Suggested placement:** Simulation validation.

## Figure 6 — Old versus hybrid trade-offs

- **Caption:** Pooled three-fold GW1 comparison of old 80-draw and promoted 10,000-draw hybrid
  simulation.
- **Exact source:** Full-precision scored xPoints parquets for the two corrected registered runs.
- **Population and grain:** 1,964 all-observed player-gameweeks.
- **Supports:** Distribution gains alongside the visible MAE regression.
- **Does not support:** Statistical superiority.
- **Limitation:** Metrics have different units and are therefore shown as separate small multiples.
- **Suggested placement:** Simulator results.

## Figure 7 — Historical P(5+) calibration

- **Caption:** Fixed-bin P(5+) calibration for old and hybrid M7 simulations on identical historical
  GW1 folds; point area represents bin size.
- **Exact source:** Full-precision `prob_points_ge_5` and official outcomes in the two corrected
  scored parquet artifacts.
- **Population and grain:** All observed player-gameweeks across three GW1 folds.
- **Supports:** Distribution calibration comparison without rounded reconstruction.
- **Does not support:** Prospective 2026-27 calibration.
- **Limitation:** High-probability bins contain few rows and should not be overinterpreted.
- **Suggested placement:** Simulator calibration subsection.

## Figure 8 — Prospective GW1 price versus xPoints

- **Caption:** Prospective 2026-27 GW1 snapshot, not evaluated against outcomes: official price versus
  frozen expected points, coloured by position with appearance probability encoded by opacity.
- **Exact source:** Corrected non-published validation-successor
  `player_gameweek_projections.csv`.
- **Population and grain:** 554 current player-gameweek projections.
- **Supports:** Descriptive structure of the released forecast and price/value landscape.
- **Does not support:** Accuracy, value realization or superiority to external forecasts.
- **Limitation:** Labels identify only a small set of top-ranked players to preserve readability.
- **Suggested placement:** Prospective example or appendix.
"""


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    expected_nonempty = {f"table{i}" for i in range(1, 9)}
    if set(tables) != expected_nonempty:
        raise AssertionError("Unexpected table set.")
    for name, frame in tables.items():
        if frame.empty:
            raise AssertionError(f"{name} is empty.")
        if frame.columns.duplicated().any():
            raise AssertionError(f"{name} has duplicate columns.")
    if tables["table1"]["duplicate_key_rows"].sum() != 0:
        raise AssertionError("Historical coverage table contains duplicate player-fixture keys.")
    if tables["table1"]["missing_critical_id_rows"].sum() != 0:
        raise AssertionError("Historical coverage table contains missing critical IDs.")
    closure = (
        tables["table5"]
        .loc[
            (tables["table5"]["comparison_block"] == "closure")
            & (tables["table5"]["draw_count"] == 10000)
        ]
        .iloc[0]
    )
    if not (
        closure["median_mean_abs_difference"] <= 0.02
        and closure["p95_mean_abs_difference"] <= 0.05
        and closure["p95_p5_abs_difference"] <= 0.01
    ):
        raise AssertionError("10,000-draw closure tolerances do not pass.")
    table6 = tables["table6"].set_index(["fold", "simulator"])
    if not (
        table6.loc[("pooled", "hybrid_10000"), "mae"]
        > table6.loc[("pooled", "old_80_draw_m7"), "mae"]
    ):
        raise AssertionError("Expected disclosed hybrid MAE regression is missing.")
    if set(tables["table7"]["gameweeks"]) != {3}:
        raise AssertionError("Decision evidence must contain exactly three gameweeks.")
    if set(tables["table7"]["d2_minus_d1_fold_realized_differences"]) != {"[2, 0, 2]"}:
        raise AssertionError("Decision evidence does not contain the corrected paired differences.")
    if set(tables["table7"]["d2_minus_d1_mean_realized_difference_2dp"]) != {1.33}:
        raise AssertionError(
            "Decision evidence does not contain the corrected rounded mean difference."
        )
    if set(tables["table8"]["label"]) != {"Prospective example, not an accuracy result"}:
        raise AssertionError("Prospective table label is missing.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goalkeeper-scoring-refresh",
        action="store_true",
        help=(
            "Regenerate only Tables/Figures 4-8 and shared provenance files from the "
            "season-aware successor evidence."
        ),
    )
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    tables = {
        "table1": table1_coverage(),
        "table2": table2_team_models(),
        "table3": table3_minutes_models(),
        "table4": table4_xpoints_models(),
        "table5": table5_convergence(),
        "table6": table6_old_hybrid(),
        "table7": table7_decisions(),
        "table8": table8_snapshot(),
    }
    validate_tables(tables)
    table_slugs = {
        "table1": "historical_data_coverage",
        "table2": "team_model_comparison",
        "table3": "minutes_model_comparison",
        "table4": "xpoints_comparison",
        "table5": "hybrid_simulation_convergence",
        "table6": "old_vs_hybrid_gw1_folds",
        "table7": "decision_system_evidence",
        "table8": "prospective_gw1_snapshot",
    }
    table_keys = (
        {"table4", "table5", "table6", "table7", "table8"}
        if args.goalkeeper_scoring_refresh
        else set(tables)
    )
    for key, frame in tables.items():
        if key not in table_keys:
            continue
        write_csv(TABLES / f"{key}_{table_slugs[key]}.csv", frame)

    old_cal = p5_calibration(
        ROOT
        / "reports/xpoints_backtests"
        / XPOINTS_GW1_RUN
        / "scored_player_fixture_predictions.parquet",
        "old_80_draw_m7",
    )
    new_cal = p5_calibration(
        ROOT
        / "reports/xpoints_backtests"
        / HYBRID_GW1_RUN
        / "scored_player_fixture_predictions.parquet",
        "hybrid_10000",
    )
    calibration_table = pd.concat([old_cal, new_cal], ignore_index=True)
    write_csv(TABLES / "figure7_p5_calibration_bins.csv", calibration_table)

    figures = {
        "Figure 1": (
            FIGURES / "figure1_system_architecture.svg",
            FIGURES / "figure1_system_architecture.png",
        ),
        "Figure 2": (
            FIGURES / "figure2_evaluation_timeline.svg",
            FIGURES / "figure2_evaluation_timeline.png",
        ),
        "Figure 3": (
            FIGURES / "figure3_team_model_comparison.svg",
            FIGURES / "figure3_team_model_comparison.png",
        ),
        "Figure 4": write_figure(4, "xpoints_comparison", figure4_xpoints(tables["table4"])),
        "Figure 5": write_figure(
            5, "simulation_convergence", figure5_convergence(tables["table5"])
        ),
        "Figure 6": write_figure(6, "simulator_tradeoffs", figure6_tradeoffs(tables["table6"])),
        "Figure 7": write_figure(7, "p5_calibration", figure7_calibration(old_cal, new_cal)),
        "Figure 8": write_figure(8, "prospective_price_xpoints", figure8_price_xpoints()),
    }
    if not args.goalkeeper_scoring_refresh:
        figures.update(
            {
                "Figure 1": write_figure(1, "system_architecture", figure1_architecture()),
                "Figure 2": write_figure(2, "evaluation_timeline", figure2_timeline()),
                "Figure 3": write_figure(
                    3, "team_model_comparison", figure3_team(tables["table2"])
                ),
            }
        )

    PAPER.joinpath("FIGURE_NOTES.md").write_text(figure_notes(), encoding="utf-8", newline="\n")
    evidence = evidence_rows(tables, figures)
    write_csv(PAPER / "evidence_manifest.csv", evidence)
    write_csv(PAPER / "evidence_supersession.csv", publication_evidence_supersession_table())

    output_paths = [
        *sorted(TABLES.glob("*.csv")),
        *sorted(FIGURES.glob("*.svg")),
        *sorted(FIGURES.glob("*.png")),
        PAPER / "FIGURE_NOTES.md",
        PAPER / "evidence_manifest.csv",
        PAPER / "evidence_supersession.csv",
    ]
    digest = hashlib.sha256()
    for path in output_paths:
        digest.update(rel(path).encode())
        digest.update(path.read_bytes())
    print(
        json.dumps(
            {
                "tables": len(list(TABLES.glob("table*.csv"))),
                "figures_svg": len(list(FIGURES.glob("*.svg"))),
                "figures_png": len(list(FIGURES.glob("*.png"))),
                "manifest_rows": len(evidence),
                "output_sha256": digest.hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
