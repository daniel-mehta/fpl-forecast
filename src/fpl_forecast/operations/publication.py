from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fpl_forecast.operations.config import LATEST_SUCCESSFUL_PATH, OPERATIONAL_FAILED_DIR, OPERATIONAL_RUNS_DIR


def latest_successful() -> dict[str, Any] | None:
    if not LATEST_SUCCESSFUL_PATH.exists():
        return None
    return json.loads(LATEST_SUCCESSFUL_PATH.read_text(encoding="utf-8"))


def publish_success(temp_dir: Path, *, run_id: str, manifest: dict[str, Any]) -> Path:
    OPERATIONAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    final_dir = OPERATIONAL_RUNS_DIR / run_id
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temp_manifest = temp_dir / "run_manifest.json"
    temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temp_dir.rename(final_dir)
    pointer_tmp = LATEST_SUCCESSFUL_PATH.with_suffix(".json.tmp")
    pointer = {
        "run_id": run_id,
        "run_dir": str(final_dir),
        "manifest_path": str(final_dir / "run_manifest.json"),
        "published_at": manifest["completed_at"],
        "schema_version": manifest["frontend_schema_version"],
    }
    pointer_tmp.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")
    pointer_tmp.replace(LATEST_SUCCESSFUL_PATH)
    return final_dir


def publish_failure(temp_dir: Path, *, run_id: str, manifest: dict[str, Any]) -> Path:
    OPERATIONAL_FAILED_DIR.mkdir(parents=True, exist_ok=True)
    failed_dir = OPERATIONAL_FAILED_DIR / run_id
    if failed_dir.exists():
        shutil.rmtree(failed_dir)
    (temp_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temp_dir.rename(failed_dir)
    return failed_dir
