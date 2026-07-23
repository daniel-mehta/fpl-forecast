from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_forecast.config import PROJECT_ROOT, RAW_FPL_API_DIR
from fpl_forecast.ingest.fpl_api import BOOTSTRAP_STATIC, EVENT_LIVE, FIXTURES
from fpl_forecast.ingest.snapshots import latest_snapshot_path, read_json_snapshot, read_metadata
from fpl_forecast.operations.live_results import audit_event_live_scoring, normalize_event_live


EVENT_LIVE_REPORTS_DIR = PROJECT_ROOT / "reports" / "operational" / "event_live"


def audit_cached_event_live(
    *,
    season: str,
    gameweek: int,
    run_id: str | None = None,
    raw_dir: Path = RAW_FPL_API_DIR,
    reports_dir: Path = EVENT_LIVE_REPORTS_DIR,
) -> dict[str, Path]:
    live_path = latest_snapshot_path(
        raw_dir,
        season=season,
        endpoint_name=f"{EVENT_LIVE}_{gameweek}",
        content_type="json",
    )
    bootstrap_path = latest_snapshot_path(raw_dir, season=season, endpoint_name=BOOTSTRAP_STATIC, content_type="json")
    fixtures_path = latest_snapshot_path(raw_dir, season=season, endpoint_name=FIXTURES, content_type="json")
    if live_path is None:
        raise FileNotFoundError(f"No cached event-live payload found for {season} GW{gameweek}.")
    if bootstrap_path is None or fixtures_path is None:
        raise FileNotFoundError("Cached bootstrap-static and fixtures payloads are required for event-live audit.")

    live_payload = read_json_snapshot(live_path)
    bootstrap_payload = read_json_snapshot(bootstrap_path)
    fixtures_payload = read_json_snapshot(fixtures_path)
    metadata = read_metadata(live_path)
    retrieved_at = metadata["retrieved_at"]
    normalized = normalize_event_live(
        season=season,
        gameweek=gameweek,
        payload=live_payload,
        retrieved_at=retrieved_at,
        raw_snapshot_path=str(live_path),
        bootstrap_payload=bootstrap_payload,
        fixtures_payload=fixtures_payload,
    )
    audit = audit_event_live_scoring(normalized)
    run_id = run_id or f"{season}_gw{gameweek}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = reports_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    before = _legacy_mismatch_distribution(
        season=season,
        gameweek=gameweek,
        payload=live_payload,
        retrieved_at=retrieved_at,
        raw_snapshot_path=str(live_path),
        bootstrap_payload=bootstrap_payload,
        fixtures_payload=fixtures_payload,
    )
    summary = _summary(
        live_payload=live_payload,
        normalized=normalized,
        audit=audit,
        before=before,
        live_path=live_path,
        bootstrap_path=bootstrap_path,
        fixtures_path=fixtures_path,
    )
    outputs = {
        "normalized_player_fixtures": _write_frame(normalized, run_dir / "normalized_player_fixtures.parquet"),
        "reconciliation": _write_frame(audit["player_event_reconciliation"], run_dir / "player_event_reconciliation.csv"),
        "mismatch_diagnostics": _write_frame(audit["mismatch_diagnostics"], run_dir / "mismatch_diagnostics.csv"),
        "difference_counts": _write_frame(audit["difference_counts"], run_dir / "difference_counts.csv"),
        "status_counts": _write_frame(audit["status_counts"], run_dir / "status_counts.csv"),
        "grouped_mismatches": _write_frame(audit["grouped_mismatches"], run_dir / "grouped_mismatches.csv"),
        "legacy_difference_counts": _write_frame(before, run_dir / "legacy_difference_counts.csv"),
        "summary": _write_json(summary, run_dir / "summary.json"),
    }
    return outputs


def _legacy_mismatch_distribution(
    *,
    season: str,
    gameweek: int,
    payload: dict[str, Any],
    retrieved_at: str,
    raw_snapshot_path: str,
    bootstrap_payload: dict[str, Any],
    fixtures_payload: list[dict[str, Any]],
) -> pd.DataFrame:
    normalized = normalize_event_live(
        season=season,
        gameweek=gameweek,
        payload=payload,
        retrieved_at=retrieved_at,
        raw_snapshot_path=raw_snapshot_path,
        bootstrap_payload=bootstrap_payload,
        fixtures_payload=fixtures_payload,
    )
    legacy = normalized.copy()
    for column in [column for column in normalized.columns if column.startswith("points_")]:
        legacy[column] = 0
    legacy["legacy_reconstructed_points"] = _legacy_score_from_values(legacy)
    grouped = (
        legacy.groupby(["season", "gameweek", "player_id"], as_index=False)
        .agg(
            legacy_reconstructed_points=("legacy_reconstructed_points", "sum"),
            official_event_total_points=("official_event_total_points", "first"),
        )
    )
    grouped["difference"] = grouped["legacy_reconstructed_points"] - grouped["official_event_total_points"]
    return grouped["difference"].value_counts().rename_axis("difference").reset_index(name="rows").sort_values("difference")


def _legacy_score_from_values(frame: pd.DataFrame) -> pd.Series:
    pos = frame["fpl_position"].astype(str)
    minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0)
    points = (minutes > 0).astype(int) + (minutes >= 60).astype(int)
    points += pd.to_numeric(frame["goals_scored"], errors="coerce").fillna(0) * pos.map(
        {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
    ).fillna(0).astype(int)
    points += pd.to_numeric(frame["assists"], errors="coerce").fillna(0) * 3
    clean = pd.to_numeric(frame["clean_sheets"], errors="coerce").fillna(0)
    points += pd.Series(
        pd.Series(pos).isin(["GKP", "DEF"]).to_numpy() * (minutes >= 60).to_numpy() * clean.to_numpy() * 4,
        index=frame.index,
    )
    points += pd.Series((pos == "MID").to_numpy() * (minutes >= 60).to_numpy() * clean.to_numpy(), index=frame.index)
    points += pd.to_numeric(frame["saves"], errors="coerce").fillna(0).astype(int) // 3
    points += pd.to_numeric(frame["penalties_saved"], errors="coerce").fillna(0) * 5
    points -= pd.to_numeric(frame["penalties_missed"], errors="coerce").fillna(0) * 2
    points -= (
        pd.to_numeric(frame["goals_conceded"], errors="coerce").fillna(0).astype(int) // 2
    ) * pos.isin(["GKP", "DEF"]).astype(int)
    points -= pd.to_numeric(frame["yellow_cards"], errors="coerce").fillna(0)
    points -= pd.to_numeric(frame["red_cards"], errors="coerce").fillna(0) * 3
    points -= pd.to_numeric(frame["own_goals"], errors="coerce").fillna(0) * 2
    points += pd.to_numeric(frame["bonus"], errors="coerce").fillna(0)
    return points.astype(int)


def _summary(
    *,
    live_payload: dict[str, Any],
    normalized: pd.DataFrame,
    audit: dict[str, pd.DataFrame],
    before: pd.DataFrame,
    live_path: Path,
    bootstrap_path: Path,
    fixtures_path: Path,
) -> dict[str, Any]:
    reconciliation = audit["player_event_reconciliation"]
    exact = reconciliation["audit_status"].eq("exact_match")
    unresolved = reconciliation["audit_status"].eq("unresolved_source_limitation")
    assistant_rows = normalized["entity_type"].eq("assistant_manager")
    duplicate_keys = int(normalized.duplicated(["season", "fixture_id", "player_uid"]).sum())
    return {
        "raw_element_count": len(live_payload.get("elements", [])),
        "normalized_player_fixture_rows": int(len(normalized)),
        "represented_fixtures": int(normalized["fixture_id"].nunique()),
        "duplicate_key_count": duplicate_keys,
        "exact_match_count": int(exact.sum()),
        "exact_match_pct": float(exact.mean()) if len(exact) else 0.0,
        "unresolved_count": int(unresolved.sum()),
        "excluded_assistant_manager_count": int(assistant_rows.sum()),
        "safe_to_use_for_subsequent_forecasting": bool(
            duplicate_keys == 0 and unresolved.sum() == 0 and exact.all()
        ),
        "source_paths": {
            "event_live": str(live_path),
            "bootstrap_static": str(bootstrap_path),
            "fixtures": str(fixtures_path),
        },
        "legacy_difference_counts": before.to_dict(orient="records"),
        "corrected_difference_counts": audit["difference_counts"].to_dict(orient="records"),
        "status_counts": audit["status_counts"].to_dict(orient="records"),
        "diagnosis": (
            "The prior audit treated fixture explain values as a complete raw-stat source and missed "
            "official awarded component points, including defensive_contribution. Correct fixture "
            "reconstruction sums explain points and points_modification, then validates against "
            "top-level event total_points."
        ),
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
