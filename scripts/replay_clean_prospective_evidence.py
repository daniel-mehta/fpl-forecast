#!/usr/bin/env python3
"""Replay prospective validation and simulator checks from a clean revision.

This script is intentionally limited to the prospective validation, convergence, and
10,000-versus-20,000-draw closure artifacts that were previously produced by ad hoc temporary
scripts. Historical xPoints and decision replays remain explicit CLI commands because their
existing runners already record their own provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.operations.config import CONFIG_PATH as OPERATIONS_CONFIG_PATH
from fpl_forecast.operations.config import load_operational_config
from fpl_forecast.operations.orchestrator import (
    _build_frontend_artifacts,
    _source_state,
    _validate_frontend_artifacts,
    require_clean_source_state,
)
from fpl_forecast.xpoints.config import CONFIG_PATH as XPOINTS_CONFIG_PATH
from fpl_forecast.xpoints.config import load_xpoints_config
from fpl_forecast.xpoints.simulation import simulate_component_points


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "operational" / "validation_runs"
DEFAULT_ORIGINAL_RUN = "preseason_sim_hybrid_10000_final_v2_abf172c"
SIMULATOR_MODEL = "X2_TEAM_CONSTRAINED_SIM_M7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New immutable prospective run ID.")
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--target-gameweek", type=int, default=1)
    parser.add_argument("--normalized-dir", type=Path, default=NORMALIZED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--original-run-id", default=DEFAULT_ORIGINAL_RUN)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def tree_metadata(path: Path) -> dict[str, Any]:
    root = path.resolve()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for child in files:
        content = child.read_bytes()
        digest.update(child.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        total_bytes += len(content)
    return {
        "path": str(root),
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "definition": "sorted relative file paths and contents",
    }


def build_prospective_validation(
    *,
    run_id: str,
    season: str,
    target_gameweek: int,
    normalized_dir: Path,
    output_root: Path,
    original_run_id: str,
    source_state: dict[str, Any],
) -> tuple[Path, Path]:
    run_dir = output_root.resolve() / run_id
    original_manifest = (
        PROJECT_ROOT
        / "outputs"
        / "operational"
        / "runs"
        / original_run_id
        / "run_manifest.json"
    )
    if not original_manifest.is_file():
        raise FileNotFoundError(original_manifest)
    run_dir.mkdir(parents=True, exist_ok=False)

    config = load_operational_config()
    paths = _build_frontend_artifacts(
        run_dir,
        season=season,
        run_id=run_id,
        config=config,
        target_gameweek=target_gameweek,
        normalized_dir=normalized_dir,
        source_mode="official_current_season",
    )
    _validate_frontend_artifacts(paths)
    lineage_path = run_dir / "model_lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "purpose": (
            "Non-published clean validation successor for season-aware goalkeeper goal scoring."
        ),
        "run_class": "research_evidence_replay",
        "supersedes_for_validation": original_run_id,
        "original_manifest_sha256": sha256_file(original_manifest),
        "target_season": season,
        "target_gameweek": target_gameweek,
        "source_mode": "official_current_season",
        "source_state": source_state,
        "input_artifacts": {
            "normalized_tree": tree_metadata(normalized_dir),
            "operations_config": artifact_metadata(OPERATIONS_CONFIG_PATH),
            "xpoints_config": artifact_metadata(XPOINTS_CONFIG_PATH),
            "pyproject": artifact_metadata(PROJECT_ROOT / "pyproject.toml"),
            "lockfile": artifact_metadata(PROJECT_ROOT / "uv.lock"),
            "original_manifest": artifact_metadata(original_manifest),
        },
        "official_snapshots": lineage["official_snapshots"],
        "training_seasons": lineage["training_seasons"],
        "xpoints_simulator": lineage["xpoints_simulator"],
        "publication_performed": False,
        "validated_artifacts": {
            name: artifact_metadata(path) for name, path in sorted(paths.items())
        },
        "model_lineage": artifact_metadata(lineage_path),
        "output_artifacts": {
            path.relative_to(run_dir).as_posix(): artifact_metadata(path)
            for path in sorted(item for item in run_dir.rglob("*") if item.is_file())
        },
        "replay": {
            "command": " ".join(sys.argv),
            "normalized_dir": str(normalized_dir),
        },
    }
    manifest_path = run_dir / "validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return run_dir, manifest_path


def write_convergence(run_dir: Path) -> Path:
    artifact = run_dir / "model_chain" / "xpoints_predictions.parquet"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "preseason_simulation_convergence.py"),
            str(artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    payload.update(
        {
            "run_id": f"{run_dir.name}_simulation_convergence",
            "source_artifact": artifact_metadata(artifact)["path"],
            "source_artifact_sha256": sha256_file(artifact),
            "publication_performed": False,
        }
    )
    destination = run_dir / "preseason_simulation_convergence.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def write_closure(run_dir: Path) -> Path:
    artifact = run_dir / "model_chain" / "xpoints_predictions.parquet"
    frame = _prepare_simulation_frame(artifact)
    base = load_xpoints_config()
    summaries: dict[int, pd.DataFrame] = {}
    draws: dict[int, np.ndarray] = {}
    for count in (10_000, 20_000):
        config = replace(base, draw_count=count)
        first, first_draws = simulate_component_points(
            frame,
            config=config,
            seed=base.random_seed + 4,
            seed_namespace=SIMULATOR_MODEL,
        )
        second, second_draws = simulate_component_points(
            frame,
            config=config,
            seed=base.random_seed + 4,
            seed_namespace=SIMULATOR_MODEL,
        )
        if not first.equals(second) or not np.array_equal(first_draws, second_draws):
            raise RuntimeError(f"Simulation closure replay is non-deterministic at {count} draws.")
        summaries[count] = first
        draws[count] = first_draws

    left, right = summaries[10_000], summaries[20_000]
    mean_diff = (
        left["simulated_expected_points"] - right["simulated_expected_points"]
    ).abs()
    p5_diff = (left["prob_points_ge_5"] - right["prob_points_ge_5"]).abs()
    zero_diff = (left["prob_points_eq_0"] - right["prob_points_eq_0"]).abs()
    interval_diff = pd.concat(
        [
            (left["points_p10"] - right["points_p10"]).abs(),
            (left["points_p90"] - right["points_p90"]).abs(),
        ],
        ignore_index=True,
    )

    def overlap(k: int) -> int:
        left_ids = set(left.nlargest(k, "simulated_expected_points").index)
        right_ids = set(right.nlargest(k, "simulated_expected_points").index)
        return len(left_ids & right_ids)

    payload = {
        "run_id": f"{run_dir.name}_simulation_closure",
        "source_artifact": artifact_metadata(artifact)["path"],
        "source_artifact_sha256": sha256_file(artifact),
        "rows": len(frame),
        "p5_p95_abs": float(p5_diff.quantile(0.95)),
        "zero_p95_abs": float(zero_diff.quantile(0.95)),
        "interval_endpoint_p95_abs": float(interval_diff.quantile(0.95)),
        "simulated_mean_median_abs": float(mean_diff.median()),
        "simulated_mean_p95_abs": float(mean_diff.quantile(0.95)),
        "simulated_mean_spearman": float(
            left["simulated_expected_points"].corr(
                right["simulated_expected_points"], method="spearman"
            )
        ),
        "top_overlap": {str(k): overlap(k) for k in (15, 30, 50)},
        "draw_matrix_megabytes": {
            str(k): draws[k].nbytes / (1024**2) for k in (10_000, 20_000)
        },
        "deterministic": True,
        "publication_performed": False,
    }
    destination = run_dir / "preseason_simulation_closure.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def _prepare_simulation_frame(artifact: Path) -> pd.DataFrame:
    frame = pd.read_parquet(artifact)
    frame = frame.loc[frame["model_name"].eq(SIMULATOR_MODEL)].copy()
    frame["defensive_contribution_threshold"] = (
        frame["fpl_position"].map({"DEF": 10, "MID": 12, "FWD": 12}).fillna(10**9)
    )
    frame["defensive_contribution_points"] = np.where(
        frame["defensive_contribution_threshold"].lt(10**9), 2, 0
    )
    team_keys = ["season", "stable_fixture_uid", "player_team_uid"]
    assisted = frame.groupby(team_keys, as_index=False).agg(
        expected_assists=("expected_assists", "sum"),
        team_xg=("team_expected_goals", "first"),
    )
    assisted["assisted_goal_rate"] = np.divide(
        assisted["expected_assists"],
        assisted["team_xg"],
        out=np.zeros(len(assisted)),
        where=assisted["team_xg"].gt(0),
    ).clip(0, 1)
    return frame.merge(
        assisted[[*team_keys, "assisted_goal_rate"]], on=team_keys, how="left"
    )


def main() -> None:
    args = parse_args()
    source_state = _source_state()
    require_clean_source_state(
        source_state,
        operation="Clean prospective evidence replay",
    )
    run_dir, manifest_path = build_prospective_validation(
        run_id=args.run_id,
        season=args.season,
        target_gameweek=args.target_gameweek,
        normalized_dir=args.normalized_dir,
        output_root=args.output_root,
        original_run_id=args.original_run_id,
        source_state=source_state,
    )
    convergence_path = write_convergence(run_dir)
    closure_path = write_closure(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["derived_artifacts"] = {
        "simulation_convergence": artifact_metadata(convergence_path),
        "simulation_closure": artifact_metadata(closure_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "manifest": str(manifest_path),
                "source_tree_sha256": source_state["source_tree_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
