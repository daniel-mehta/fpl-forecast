from __future__ import annotations

import json

import httpx
import pytest

from conftest import fixture_bytes
from fpl_forecast.ingest.fpl_api import (
    BOOTSTRAP_STATIC,
    FPLApiClient,
    FPLApiError,
    load_latest_fpl_snapshot,
)


def test_fpl_response_parsing_and_snapshot_current(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=_standard_bootstrap())
        if request.url.path.endswith("/fixtures/"):
            return httpx.Response(200, json=_standard_fixtures())
        raise AssertionError(f"unexpected URL {request.url}")

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    records = client.snapshot_current(season="2025-26", refresh=True)

    assert {record.endpoint_name for record in records} == {"bootstrap_static", "fixtures"}
    loaded = load_latest_fpl_snapshot(
        raw_dir=tmp_path,
        season="2025-26",
        endpoint_name=BOOTSTRAP_STATIC,
    )
    assert loaded.payload["elements"][0]["id"] == 11
    assert loaded.metadata["source"] == "fpl_api"
    assert loaded.metadata["inferred_season"] == "2025-26"


def test_fpl_http_errors_are_informative(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="maintenance", request=request)

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(FPLApiError, match="HTTP 503"):
        client.fetch_endpoint(
            season="2026-27",
            endpoint_name="bootstrap_static",
            url=client.bootstrap_static_url(),
            refresh=True,
        )


def test_fpl_invalid_json_fails_without_snapshot(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not-json")

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(FPLApiError, match="Invalid JSON"):
        client.fetch_endpoint(
            season="2026-27",
            endpoint_name="bootstrap_static",
            url=client.bootstrap_static_url(),
            refresh=True,
        )
    assert not list(tmp_path.glob("**/*.json"))


def test_fpl_missing_required_top_level_fields_fails(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"elements": []})

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(FPLApiError, match="missing required fields"):
        client.fetch_endpoint(
            season="2026-27",
            endpoint_name="bootstrap_static",
            url=client.bootstrap_static_url(),
            refresh=True,
        )


def test_fpl_cache_and_refresh_behavior(tmp_path):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, content=fixture_bytes("fpl_api/bootstrap_static.json"))

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    first = client.fetch_endpoint(
        season="2026-27",
        endpoint_name="bootstrap_static",
        url=client.bootstrap_static_url(),
        refresh=True,
    )
    second = client.fetch_endpoint(
        season="2026-27",
        endpoint_name="bootstrap_static",
        url=client.bootstrap_static_url(),
        refresh=False,
    )
    third = client.fetch_endpoint(
        season="2026-27",
        endpoint_name="bootstrap_static",
        url=client.bootstrap_static_url(),
        refresh=True,
    )

    assert calls["count"] == 2
    assert first.raw_path == second.raw_path
    assert third.raw_path != first.raw_path


def test_fpl_snapshot_current_reuses_valid_cache_offline(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=_standard_bootstrap())
        if request.url.path.endswith("/fixtures/"):
            return httpx.Response(200, json=_standard_fixtures())
        raise AssertionError(f"unexpected URL {request.url}")

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    online_records = client.snapshot_current(season="2025-26", refresh=True)
    offline_client = FPLApiClient(
        raw_dir=tmp_path,
        transport=httpx.MockTransport(lambda request: pytest.fail("network should not be used")),
    )
    offline_records = offline_client.snapshot_current(season="2025-26", offline=True)

    assert [record.raw_path for record in offline_records] == [record.raw_path for record in online_records]


def test_fpl_offline_without_cache_fails(tmp_path):
    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(lambda request: None))
    with pytest.raises(FPLApiError, match="offline mode"):
        client.snapshot_current(
            season="2026-27",
            offline=True,
        )


def test_fpl_timeout_error_is_informative(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("too slow", request=request)

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(FPLApiError, match="request failed"):
        client.fetch_endpoint(
            season="2026-27",
            endpoint_name="bootstrap_static",
            url=client.bootstrap_static_url(),
            refresh=True,
        )


def test_snapshot_current_rejects_mismatched_season(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=_standard_bootstrap())
        if request.url.path.endswith("/fixtures/"):
            return httpx.Response(200, json=_standard_fixtures())
        raise AssertionError(f"unexpected URL {request.url}")

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(FPLApiError, match="conflicts with inferred payload season 2025-26"):
        client.snapshot_current(season="2026-27", refresh=True)


def test_snapshot_current_rejects_malformed_dates(tmp_path):
    bootstrap = _standard_bootstrap()
    bootstrap["events"][0]["deadline_time"] = "not-a-date"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=bootstrap)
        if request.url.path.endswith("/fixtures/"):
            return httpx.Response(200, json=_standard_fixtures())
        raise AssertionError(f"unexpected URL {request.url}")

    client = FPLApiClient(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(FPLApiError, match="malformed"):
        client.snapshot_current(season="2025-26", refresh=True)


def _standard_bootstrap() -> dict:
    payload = json.loads(fixture_bytes("fpl_api/bootstrap_static.json"))
    payload["events"] = [
        {"id": gameweek, "deadline_time": f"2025-08-{min(10 + gameweek, 28):02d}T11:00:00Z"}
        for gameweek in range(1, 39)
    ]
    payload["events"][-1]["deadline_time"] = "2026-05-24T13:30:00Z"
    payload["teams"] = [
        {"id": team_id, "code": team_id, "name": f"Team {team_id}", "short_name": f"T{team_id}", "strength": 3}
        for team_id in range(1, 21)
    ]
    return payload


def _standard_fixtures() -> list[dict]:
    fixtures = []
    fixture_id = 1
    for gameweek in range(1, 39):
        for match in range(10):
            fixtures.append(
                {
                    "id": fixture_id,
                    "code": 10_000 + fixture_id,
                    "event": gameweek,
                    "team_h": (match % 20) + 1,
                    "team_a": ((match + 1) % 20) + 1,
                    "kickoff_time": "2025-08-15T19:00:00Z"
                    if gameweek == 1
                    else "2026-05-24T15:00:00Z"
                    if gameweek == 38
                    else "2025-12-15T15:00:00Z",
                    "finished": False,
                    "started": False,
                    "team_h_score": None,
                    "team_a_score": None,
                }
            )
            fixture_id += 1
    return fixtures
