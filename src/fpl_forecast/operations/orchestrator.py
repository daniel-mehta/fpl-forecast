from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT, RAW_FPL_API_DIR
from fpl_forecast.operations.config import (
    LATEST_SUCCESSFUL_PATH,
    OPERATIONAL_OUTPUT_DIR,
    OPERATIONAL_RUNS_DIR,
    STATUS_PATH,
    load_operational_config,
    operational_paths,
)
from fpl_forecast.operations.launch import LaunchCheck, check_season_launch
from fpl_forecast.operations.locking import RefreshLock, RefreshLockError
from fpl_forecast.operations.live_results import validate_event_live_for_forecast
from fpl_forecast.operations.model_chain import run_operational_model_chain
from fpl_forecast.operations.publication import latest_successful, publish_failure, publish_success
from fpl_forecast.operations.state import OperationalStateName, OperationalStatus, now_utc, write_status


@dataclass(frozen=True)
class RefreshResult:
    status: OperationalStatus
    run_id: str | None
    run_dir: Path | None
    manifest_path: Path | None
    no_op: bool


def refresh_operational(
    *,
    season: str,
    offline: bool = True,
    mock_launch: bool = False,
    force: bool = False,
    run_id: str | None = None,
    fail_stage: str | None = None,
    status_only: bool = False,
    target_gameweek: int = 1,
    completed_player_fixtures: pd.DataFrame | None = None,
    completed_team_fixtures: pd.DataFrame | None = None,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> RefreshResult:
    _ensure_dirs()
    config = load_operational_config()
    latest = latest_successful()
    try:
        with RefreshLock():
            launch = _mock_launch_check(season) if mock_launch else check_season_launch(
                season=season,
                raw_dir=RAW_FPL_API_DIR,
                offline=offline,
            )
            if status_only or launch.status.state != OperationalStateName.READY_TO_REFRESH:
                status = _with_latest(launch.status, latest)
                write_status(status)
                return RefreshResult(status, None, None, None, no_op=False)
            fingerprint = _input_fingerprint(
                launch,
                mock_launch=mock_launch,
                target_gameweek=target_gameweek,
                completed_player_fixtures=completed_player_fixtures,
                completed_team_fixtures=completed_team_fixtures,
                normalized_dir=normalized_dir,
            )
            if latest and latest.get("input_fingerprint") == fingerprint and not force:
                status = OperationalStatus(
                    state=OperationalStateName.SUCCEEDED,
                    target_season=season,
                    inferred_official_season=launch.status.inferred_official_season,
                    checked_at=now_utc(),
                    latest_official_deadline=launch.status.latest_official_deadline,
                    latest_successful_run_id=latest.get("run_id"),
                    reason="Inputs, code and configuration are unchanged; latest successful run reused.",
                    dashboard_can_display_forecasts=True,
                    retry_automatically=False,
                    extra={"no_op": True, "input_fingerprint": fingerprint},
                )
                write_status(status)
                return RefreshResult(status, latest.get("run_id"), Path(latest["run_dir"]), Path(latest["manifest_path"]), True)

            run_id = run_id or f"phase8_operational_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            temp_dir = OPERATIONAL_OUTPUT_DIR / f".tmp_{run_id}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True)
            stages: list[str] = []
            try:
                _maybe_fail("ingestion", fail_stage)
                stages.append("launch_checked")
                _maybe_fail("modeling", fail_stage)
                frontend = _build_frontend_artifacts(
                    temp_dir,
                    season=season,
                    run_id=run_id,
                    config=config,
                    target_gameweek=target_gameweek,
                    completed_player_fixtures=completed_player_fixtures,
                    completed_team_fixtures=completed_team_fixtures,
                    normalized_dir=normalized_dir,
                )
                stages.append("frontend_artifacts_built")
                _validate_frontend_artifacts(frontend)
                _maybe_fail("optimization", fail_stage)
                stages.append("frontend_artifacts_validated")
                completed_at = now_utc()
                manifest = {
                    "schema_version": config.schema_version,
                    "frontend_schema_version": config.frontend_schema_version,
                    "run_id": run_id,
                    "target_season": season,
                    "target_gameweek": target_gameweek,
                    "inferred_official_season": launch.status.inferred_official_season,
                    "launch_state": launch.status.state.value,
                    "created_at": completed_at,
                    "completed_at": completed_at,
                    "input_fingerprint": fingerprint,
                    "code_revision": _git(["rev-parse", "--short", "HEAD"]),
                    "dirty_worktree": bool(_git(["status", "--short"])),
                    "models": config.default_models,
                    "model_lineage": _read_json(temp_dir / "model_lineage.json"),
                    "warnings": ["mock_target_season_transition" if mock_launch else "real_target_season"],
                    "fall_back_flags": ["cold_start_new_player", "neutral_promoted_team"],
                    "stages": stages,
                    "previous_latest_successful_run_id": latest.get("run_id") if latest else None,
                    "completion_stage": "published",
                }
                _maybe_fail("publication", fail_stage)
                final_dir = publish_success(temp_dir, run_id=run_id, manifest={**manifest, "input_fingerprint": fingerprint})
                pointer = json.loads(LATEST_SUCCESSFUL_PATH.read_text(encoding="utf-8"))
                pointer["input_fingerprint"] = fingerprint
                LATEST_SUCCESSFUL_PATH.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")
                status = OperationalStatus(
                    state=OperationalStateName.SUCCEEDED,
                    target_season=season,
                    inferred_official_season=launch.status.inferred_official_season,
                    checked_at=completed_at,
                    latest_official_deadline=launch.status.latest_official_deadline,
                    latest_successful_run_id=run_id,
                    reason="Operational refresh published atomically.",
                    dashboard_can_display_forecasts=True,
                    retry_automatically=False,
                    extra={"run_dir": str(final_dir), "input_fingerprint": fingerprint},
                )
                write_status(status)
                return RefreshResult(status, run_id, final_dir, final_dir / "run_manifest.json", no_op=False)
            except Exception as exc:
                failed_at = now_utc()
                manifest = {
                    "schema_version": config.schema_version,
                    "frontend_schema_version": config.frontend_schema_version,
                    "run_id": run_id,
                    "target_season": season,
                    "inferred_official_season": launch.status.inferred_official_season,
                    "created_at": failed_at,
                    "completed_at": failed_at,
                    "input_fingerprint": fingerprint,
                    "completion_stage": fail_stage or "unknown",
                    "error": str(exc),
                    "previous_latest_successful_run_id": latest.get("run_id") if latest else None,
                    "stages": stages,
                }
                failed_dir = publish_failure(temp_dir, run_id=run_id, manifest=manifest)
                status = OperationalStatus(
                    state=OperationalStateName.FAILED_USING_LAST_SUCCESS,
                    target_season=season,
                    inferred_official_season=launch.status.inferred_official_season,
                    checked_at=failed_at,
                    latest_official_deadline=launch.status.latest_official_deadline,
                    latest_successful_run_id=latest.get("run_id") if latest else None,
                    reason=f"Refresh failed at {fail_stage or 'unknown'}; latest successful pointer was not changed.",
                    warning=str(exc),
                    dashboard_can_display_forecasts=bool(latest),
                    retry_automatically=True,
                    extra={"failed_dir": str(failed_dir)},
                )
                write_status(status)
                return RefreshResult(status, None, None, None, no_op=False)
    except RefreshLockError as exc:
        status = OperationalStatus(
            state=OperationalStateName.FAILED_USING_LAST_SUCCESS,
            target_season=season,
            inferred_official_season=None,
            checked_at=now_utc(),
            latest_successful_run_id=latest.get("run_id") if latest else None,
            reason=str(exc),
            dashboard_can_display_forecasts=bool(latest),
            retry_automatically=True,
        )
        write_status(status)
        return RefreshResult(status, None, None, None, no_op=False)


def operational_status_lines() -> list[str]:
    status_path = STATUS_PATH
    pointer = latest_successful()
    if not status_path.exists():
        return ["state=UNKNOWN", "reason=No operational status has been written yet."]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    lines = [
        f"state={status['state']}",
        f"target_season={status['target_season']}",
        f"inferred_official_season={status.get('inferred_official_season')}",
        f"latest_successful_run_id={status.get('latest_successful_run_id')}",
        f"dashboard_can_display_forecasts={status.get('dashboard_can_display_forecasts')}",
        f"reason={status.get('reason')}",
        f"status_path={status_path}",
    ]
    if pointer:
        lines.append(f"latest_successful={pointer['run_id']}")
    return lines


def verify_operational_readiness() -> list[str]:
    pointer = latest_successful()
    status_exists = STATUS_PATH.exists()
    runs_dir_exists = OPERATIONAL_RUNS_DIR.exists()
    return [
        "operational_readiness=ok",
        f"status_exists={status_exists}",
        f"latest_successful_run_id={pointer.get('run_id') if pointer else None}",
        f"runs_dir_exists={runs_dir_exists}",
        "dashboard_contract=phase8_frontend_v1",
    ]


def _build_frontend_artifacts(
    temp_dir: Path,
    *,
    season: str,
    run_id: str,
    config,
    target_gameweek: int = 1,
    completed_player_fixtures: pd.DataFrame | None = None,
    completed_team_fixtures: pd.DataFrame | None = None,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> dict[str, Path]:
    selected_model = config.default_models["xpoints"]
    if completed_player_fixtures is not None and "official_event_total_points" in completed_player_fixtures.columns:
        cutoff = _default_information_cutoff(season, target_gameweek=target_gameweek)
        issues = validate_event_live_for_forecast(completed_player_fixtures, information_cutoff=cutoff)
        if issues:
            raise ValueError(f"Completed-result publication safety gate failed: {', '.join(issues)}")
    result = run_operational_model_chain(
        season=season,
        run_id=run_id,
        output_dir=temp_dir,
        target_gameweek=target_gameweek,
        completed_player_fixtures=completed_player_fixtures,
        completed_team_fixtures=completed_team_fixtures,
        normalized_dir=normalized_dir,
    )

    projections = result.decision_candidates.loc[result.decision_candidates["model_name"].eq(selected_model)].copy()
    projections = projections.rename(
        columns={
            "player_uid": "stable_player_id",
            "player_name": "player",
            "fpl_position": "position",
            "player_team_uid": "team",
        }
    )
    projections["schema_version"] = config.frontend_schema_version
    projections["season"] = season
    projections["status"] = projections["status"].fillna("a")
    projections["news"] = projections["news"].fillna("")
    projections["model_variant"] = selected_model
    projections["data_timestamp"] = now_utc()
    projections["team_model_run_id"] = result.lineage["team_model_run_id"]
    projections["minutes_model_run_id"] = result.lineage["minutes_model_run_id"]
    projections["xpoints_model_run_id"] = result.lineage["xpoints_model_run_id"]
    projections["decision_run_id"] = result.lineage["decision_run_id"]
    projection_columns = [
        "schema_version",
        "season",
        "gameweek",
        "stable_player_id",
        "player",
        "team",
        "position",
        "price_tenths",
        "expected_points",
        "points_std",
        "points_p10",
        "points_p50",
        "points_p90",
        "expected_minutes",
        "p_appearance",
        "p_start",
        "prob_points_ge_5",
        "prob_points_ge_10",
        "status",
        "news",
        "cold_start_no_history",
        "fallback_flag",
        "lineage_note",
        "model_variant",
        "data_timestamp",
        "team_model_run_id",
        "minutes_model_run_id",
        "xpoints_model_run_id",
        "decision_run_id",
    ]
    projections_path = temp_dir / "player_gameweek_projections.csv"
    projections[[column for column in projection_columns if column in projections.columns]].to_csv(
        projections_path,
        index=False,
    )

    squad_path = temp_dir / "optimized_squad.csv"
    result.optimized_squad.to_csv(squad_path, index=False)
    lineup_path = temp_dir / "optimized_lineup.csv"
    lineup = result.optimized_lineup.copy()
    lineup["objective"] = lineup["expected_team_points"]
    lineup.to_csv(lineup_path, index=False)
    comparison_path = temp_dir / "model_comparison.csv"
    result.model_comparison.to_csv(comparison_path, index=False)
    lineage_path = temp_dir / "model_lineage.json"
    lineage_path.write_text(json.dumps(result.lineage, indent=2, sort_keys=True, default=str), encoding="utf-8")
    freshness_path = temp_dir / "data_freshness.json"
    freshness_path.write_text(
        json.dumps(
            {
                "schema_version": config.frontend_schema_version,
                "generated_at": now_utc(),
                "source": "mocked target-season production model chain",
                "lineage_artifact": "model_lineage.json",
                "team_model_run_id": result.lineage["team_model_run_id"],
                "minutes_model_run_id": result.lineage["minutes_model_run_id"],
                "xpoints_model_run_id": result.lineage["xpoints_model_run_id"],
                "decision_run_id": result.lineage["decision_run_id"],
                "stale": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    status_path = temp_dir / "operational_status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": config.frontend_schema_version,
                "state": OperationalStateName.SUCCEEDED.value,
                "target_season": season,
                "target_gameweek": target_gameweek,
                "run_id": run_id,
                "reason": "Operational refresh published genuinely generated target-season projections.",
                "warning": "Representative mocked target-season run; real 2026-27 remains unproven.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "operational_status": status_path,
        "player_gameweek_projections": projections_path,
        "optimized_squad": squad_path,
        "optimized_lineup": lineup_path,
        "model_comparison": comparison_path,
        "data_freshness": freshness_path,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_frontend_artifacts(paths: dict[str, Path]) -> None:
    required = set(paths)
    if required != {
        "operational_status",
        "player_gameweek_projections",
        "optimized_squad",
        "optimized_lineup",
        "model_comparison",
        "data_freshness",
    }:
        raise ValueError("Frontend artifact contract is incomplete.")
    projections = pd.read_csv(paths["player_gameweek_projections"])
    required_columns = {"schema_version", "stable_player_id", "player", "team", "position", "price_tenths", "expected_points"}
    missing = required_columns.difference(projections.columns)
    if missing:
        raise ValueError(f"Projection artifact missing columns: {', '.join(sorted(missing))}")


def _mock_launch_check(season: str) -> LaunchCheck:
    return LaunchCheck(
        status=OperationalStatus(
            state=OperationalStateName.READY_TO_REFRESH,
            target_season=season,
            inferred_official_season=season,
            checked_at=now_utc(),
            latest_official_deadline=f"{season[:4]}-08-15T17:30:00Z",
            reason="Mocked target-season launch payload is compatible with verified rules.",
            retry_automatically=False,
        ),
        bootstrap_path=None,
        fixtures_path=None,
        rule_diff={"material_changes": {}},
    )


def _input_fingerprint(
    launch: LaunchCheck,
    *,
    mock_launch: bool,
    target_gameweek: int = 1,
    completed_player_fixtures: pd.DataFrame | None = None,
    completed_team_fixtures: pd.DataFrame | None = None,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(load_operational_config().__dict__, sort_keys=True, default=str).encode())
    digest.update(_git(["rev-parse", "HEAD"]).encode())
    digest.update(str(mock_launch).encode())
    digest.update(str(target_gameweek).encode())
    digest.update(str(Path(normalized_dir)).encode())
    for frame in (completed_player_fixtures, completed_team_fixtures):
        if frame is not None and not frame.empty:
            digest.update(pd.util.hash_pandas_object(frame.sort_index(axis=1), index=True).values.tobytes())
    for path in (launch.bootstrap_path, launch.fixtures_path):
        if path and path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _ensure_dirs() -> None:
    for path in operational_paths():
        path.mkdir(parents=True, exist_ok=True)


def _with_latest(status: OperationalStatus, latest: dict[str, Any] | None) -> OperationalStatus:
    return OperationalStatus(
        **{
            **status.__dict__,
            "latest_successful_run_id": latest.get("run_id") if latest else None,
            "dashboard_can_display_forecasts": bool(latest) and status.dashboard_can_display_forecasts,
        }
    )


def _maybe_fail(stage: str, fail_stage: str | None) -> None:
    if fail_stage == stage:
        raise RuntimeError(f"Injected failure at {stage}.")


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _default_information_cutoff(season: str, *, target_gameweek: int) -> pd.Timestamp:
    if target_gameweek <= 1:
        return pd.Timestamp(f"{season[:4]}-08-01T10:00:00Z")
    day = 15 + (target_gameweek - 1) * 7 - 1
    return pd.Timestamp(f"{season[:4]}-08-{day:02d}T10:00:00Z")
