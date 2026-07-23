from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fpl_forecast.ingest.fpl_api import BOOTSTRAP_STATIC, FIXTURES, FPLApiClient, FPLApiError
from fpl_forecast.ingest.season import SeasonIdentityError, infer_and_validate_current_season
from fpl_forecast.ingest.snapshots import latest_snapshot_path, read_json_snapshot
from fpl_forecast.operations.state import OperationalStateName, OperationalStatus, now_utc


@dataclass(frozen=True)
class LaunchCheck:
    status: OperationalStatus
    bootstrap_path: Path | None
    fixtures_path: Path | None
    rule_diff: dict[str, Any]


def check_season_launch(
    *,
    season: str,
    raw_dir: Path,
    offline: bool = True,
    refresh: bool = False,
) -> LaunchCheck:
    bootstrap_path = latest_snapshot_path(raw_dir, season=season, endpoint_name=BOOTSTRAP_STATIC, content_type="json")
    fixtures_path = latest_snapshot_path(raw_dir, season=season, endpoint_name=FIXTURES, content_type="json")
    if (bootstrap_path is None or fixtures_path is None) and not offline:
        client = FPLApiClient(raw_dir=raw_dir)
        try:
            client.snapshot_current(season=season, refresh=refresh, offline=False)
        except FPLApiError:
            pass
        bootstrap_path = latest_snapshot_path(raw_dir, season=season, endpoint_name=BOOTSTRAP_STATIC, content_type="json")
        fixtures_path = latest_snapshot_path(raw_dir, season=season, endpoint_name=FIXTURES, content_type="json")
    if bootstrap_path is None or fixtures_path is None:
        return LaunchCheck(
            status=OperationalStatus(
                state=OperationalStateName.WAITING_FOR_SEASON_LAUNCH,
                target_season=season,
                inferred_official_season=None,
                checked_at=now_utc(),
                reason="No cached official bootstrap/fixtures payloads are available for this season.",
                retry_automatically=True,
            ),
            bootstrap_path=bootstrap_path,
            fixtures_path=fixtures_path,
            rule_diff={},
        )

    bootstrap = read_json_snapshot(bootstrap_path)
    fixtures = read_json_snapshot(fixtures_path)
    try:
        identity = infer_and_validate_current_season(
            requested_season=season,
            bootstrap_payload=bootstrap,
            fixtures_payload=fixtures,
        )
    except SeasonIdentityError:
        inferred = _infer_without_validation(bootstrap, fixtures)
        return LaunchCheck(
            status=OperationalStatus(
                state=OperationalStateName.WAITING_FOR_SEASON_LAUNCH,
                target_season=season,
                inferred_official_season=inferred,
                checked_at=now_utc(),
                latest_official_deadline=_latest_deadline(bootstrap),
                latest_completed_gameweek=_latest_completed_gameweek(bootstrap),
                latest_completed_fixture=_latest_completed_fixture(fixtures),
                reason=f"Official payload currently identifies as {inferred}, not {season}.",
                retry_automatically=True,
            ),
            bootstrap_path=bootstrap_path,
            fixtures_path=fixtures_path,
            rule_diff={},
        )

    rule_diff = compare_rule_schema(bootstrap)
    if rule_diff["material_changes"]:
        return LaunchCheck(
            status=OperationalStatus(
                state=OperationalStateName.NEEDS_RULE_REVIEW,
                target_season=season,
                inferred_official_season=identity.inferred_season,
                checked_at=now_utc(),
                latest_official_deadline=identity.first_event_deadline,
                reason="Official rule schema differs from the verified Phase 7 configuration.",
                retry_automatically=False,
                review_artifact="rule_diff.json",
            ),
            bootstrap_path=bootstrap_path,
            fixtures_path=fixtures_path,
            rule_diff=rule_diff,
        )

    return LaunchCheck(
        status=OperationalStatus(
            state=OperationalStateName.READY_TO_REFRESH,
            target_season=season,
            inferred_official_season=identity.inferred_season,
            checked_at=now_utc(),
            latest_official_deadline=identity.first_event_deadline,
            latest_completed_gameweek=_latest_completed_gameweek(bootstrap),
            latest_completed_fixture=_latest_completed_fixture(fixtures),
            reason="Official payload matches the requested season and verified rule schema.",
            retry_automatically=False,
        ),
        bootstrap_path=bootstrap_path,
        fixtures_path=fixtures_path,
        rule_diff=rule_diff,
    )


def compare_rule_schema(bootstrap: dict[str, Any]) -> dict[str, Any]:
    settings = bootstrap.get("game_settings", {})
    element_types = bootstrap.get("element_types", [])
    expected = {
        "squad_squadsize": 15,
        "squad_squadplay": 11,
        "squad_team_limit": 3,
        "squad_total_spend": 1000,
        "sys_vice_captain_enabled": True,
    }
    observed = {key: settings.get(key) for key in expected}
    quota_expected = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    quota_observed = {
        item.get("singular_name_short"): item.get("squad_select")
        for item in element_types
        if isinstance(item, dict)
    }
    material = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    for key, value in quota_expected.items():
        if quota_observed.get(key) != value:
            material[f"quota_{key}"] = {"expected": value, "observed": quota_observed.get(key)}
    extra_positions = sorted(set(quota_observed).difference(quota_expected))
    if extra_positions:
        material["extra_positions"] = {"expected": [], "observed": extra_positions}
    return {"material_changes": material, "observed": observed, "observed_position_quotas": quota_observed}


def _infer_without_validation(bootstrap: dict[str, Any], fixtures: list[dict[str, Any]]) -> str | None:
    try:
        kickoff = min(value["kickoff_time"] for value in fixtures if value.get("kickoff_time"))
    except (ValueError, TypeError):
        return None
    year = int(str(kickoff)[:4])
    month = int(str(kickoff)[5:7])
    start = year if month >= 6 else year - 1
    return f"{start}-{(start + 1) % 100:02d}" if bootstrap else None


def _latest_deadline(bootstrap: dict[str, Any]) -> str | None:
    values = [event.get("deadline_time") for event in bootstrap.get("events", []) if event.get("deadline_time")]
    return min(values) if values else None


def _latest_completed_gameweek(bootstrap: dict[str, Any]) -> int | None:
    completed = [event.get("id") for event in bootstrap.get("events", []) if event.get("finished")]
    return max(completed) if completed else None


def _latest_completed_fixture(fixtures: list[dict[str, Any]]) -> int | None:
    completed = [fixture.get("id") for fixture in fixtures if fixture.get("finished")]
    return max(completed) if completed else None
