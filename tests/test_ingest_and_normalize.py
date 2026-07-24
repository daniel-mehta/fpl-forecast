from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest

from conftest import fixture_bytes
from fpl_forecast.ingest.snapshots import write_raw_snapshot
from fpl_forecast.ingest.season import SeasonIdentityError
from fpl_forecast.ingest.vaastav import VaastavDataError, VaastavIngestor
from fpl_forecast.normalize.current import normalize_current
from fpl_forecast.normalize.historical import normalize_historical
from test_fpl_api import _standard_bootstrap, _standard_fixtures


def test_current_normalization_end_to_end_from_cached_snapshots(tmp_path):
    raw_dir = tmp_path / "raw" / "fpl_api"
    normalized_dir = tmp_path / "normalized"
    retrieved_at = datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)
    write_raw_snapshot(
        raw_dir,
        season="2025-26",
        endpoint_name="bootstrap_static",
        content=json.dumps(_standard_bootstrap()).encode(),
        source_url="https://fantasy.premierleague.com/api/bootstrap-static/",
        http_status=200,
        response_headers={},
        source="fpl_api",
        retrieved_at=retrieved_at,
    )
    write_raw_snapshot(
        raw_dir,
        season="2025-26",
        endpoint_name="fixtures",
        content=json.dumps(_standard_fixtures()).encode(),
        source_url="https://fantasy.premierleague.com/api/fixtures/",
        http_status=200,
        response_headers={},
        source="fpl_api",
        retrieved_at=retrieved_at,
    )

    outputs = normalize_current(
        season="2025-26",
        raw_dir=raw_dir,
        normalized_dir=normalized_dir,
    )

    assert {path.name for path in outputs} == {
        "current_players.parquet",
        "current_teams.parquet",
        "current_fixtures.parquet",
    }
    players = pd.read_parquet(normalized_dir / "2025-26" / "current_players.parquet")
    assert players.loc[0, "player_id"] == 11
    assert players.loc[0, "player_code"] == 1001
    assert players.loc[0, "source"] == "fpl_api"
    assert players.loc[0, "raw_snapshot_path"].endswith(".json")


def test_current_normalization_rejects_mismatched_cached_season(tmp_path):
    raw_dir = tmp_path / "raw" / "fpl_api"
    normalized_dir = tmp_path / "normalized"
    retrieved_at = datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)
    write_raw_snapshot(
        raw_dir,
        season="2026-27",
        endpoint_name="bootstrap_static",
        content=json.dumps(_standard_bootstrap()).encode(),
        source_url="https://fantasy.premierleague.com/api/bootstrap-static/",
        http_status=200,
        response_headers={},
        source="fpl_api",
        retrieved_at=retrieved_at,
    )
    write_raw_snapshot(
        raw_dir,
        season="2026-27",
        endpoint_name="fixtures",
        content=json.dumps(_standard_fixtures()).encode(),
        source_url="https://fantasy.premierleague.com/api/fixtures/",
        http_status=200,
        response_headers={},
        source="fpl_api",
        retrieved_at=retrieved_at,
    )

    with pytest.raises(SeasonIdentityError, match="conflicts"):
        normalize_current(
            season="2026-27",
            raw_dir=raw_dir,
            normalized_dir=normalized_dir,
        )


def test_vaastav_ingestion_warns_on_xp_and_normalizes_without_xp(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/merged_gw.csv"):
            return httpx.Response(200, content=fixture_bytes("vaastav/merged_gw.csv"))
        if request.url.path.endswith("/players_raw.csv"):
            return httpx.Response(200, content=fixture_bytes("vaastav/players_raw.csv"))
        raise AssertionError(f"unexpected URL {request.url}")

    raw_dir = tmp_path / "raw" / "vaastav"
    normalized_dir = tmp_path / "normalized"
    ingestor = VaastavIngestor(raw_dir=raw_dir, transport=httpx.MockTransport(handler))
    result = ingestor.ingest_season(season="2024-25", revision="abc123", refresh=True)

    assert len(result.records) == 2
    assert result.source_version == "abc123"
    assert any("xP" in warning for warning in result.warnings)

    outputs = normalize_historical(
        season="2024-25",
        raw_dir=raw_dir,
        normalized_dir=normalized_dir,
    )
    history = pd.read_parquet(outputs[0])
    assert outputs[0].name == "historical_player_fixtures.parquet"
    assert len(history) == 2
    assert not any(column.lower() == "xp" for column in history.columns)
    assert "source_position" in history.columns
    assert "fpl_position" in history.columns
    assert history.loc[0, "player_code"] == 1001
    assert history.loc[0, "source"] == "vaastav"
    assert history.loc[0, "source_version"] == "abc123"


def test_historical_normalization_preserves_double_and_blank_gameweek_grain(tmp_path):
    raw_dir = tmp_path / "raw" / "vaastav"
    normalized_dir = tmp_path / "normalized"
    merged = (
        "name,position,team,xP,element,fixture,round,minutes,total_points,value,"
        "was_home,opponent_team,kickoff_time\n"
        "Double Player,MID,Arsenal,5.4,11,101,2,90,10,75,true,2,2025-09-01T12:00:00Z\n"
        "Double Player,MID,Arsenal,4.4,11,102,2,0,0,75,false,3,2025-09-04T12:00:00Z\n"
        "Single Player,DEF,Chelsea,2.1,12,103,4,90,2,55,false,1,2025-09-20T12:00:00Z\n"
    )
    players = (
        "id,code,first_name,second_name,web_name,team,element_type,now_cost\n"
        "11,1001,Double,Player,D Player,1,3,75\n"
        "12,1002,Single,Player,S Player,2,2,55\n"
    )
    write_raw_snapshot(
        raw_dir,
        season="2024-25",
        endpoint_name="merged_gw",
        content=merged.encode(),
        source_url="https://example.test/merged_gw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
    )
    write_raw_snapshot(
        raw_dir,
        season="2024-25",
        endpoint_name="players_raw",
        content=players.encode(),
        source_url="https://example.test/players_raw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
    )

    outputs = normalize_historical(
        season="2024-25",
        raw_dir=raw_dir,
        normalized_dir=normalized_dir,
    )
    history = pd.read_parquet(outputs[0])

    assert history.duplicated(["season", "gameweek", "player_id"]).sum() == 1
    assert history.duplicated(["season", "gameweek", "player_id", "fixture_id"]).sum() == 0
    assert 3 not in set(history["gameweek"])
    assert (history["minutes"] == 0).sum() == 1


def test_historical_normalization_maps_assistant_manager_element_type(tmp_path):
    raw_dir = tmp_path / "raw" / "vaastav"
    normalized_dir = tmp_path / "normalized"
    merged = (
        "name,position,team,element,fixture,round,minutes,total_points,value,"
        "was_home,opponent_team,kickoff_time\n"
        "Mikel Arteta,AM,Arsenal,735,201,23,0,9,15,true,2,2025-02-01T12:00:00Z\n"
    )
    players = (
        "id,code,first_name,second_name,web_name,team,element_type,now_cost\n"
        "735,100051017,Mikel,Arteta,Arteta,1,5,15\n"
    )
    write_raw_snapshot(
        raw_dir,
        season="2024-25",
        endpoint_name="merged_gw",
        content=merged.encode(),
        source_url="https://example.test/merged_gw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
    )
    write_raw_snapshot(
        raw_dir,
        season="2024-25",
        endpoint_name="players_raw",
        content=players.encode(),
        source_url="https://example.test/players_raw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
    )

    outputs = normalize_historical(
        season="2024-25",
        raw_dir=raw_dir,
        normalized_dir=normalized_dir,
    )
    history = pd.read_parquet(outputs[0])

    assert history.loc[0, "source_position"] == "AM"
    assert history.loc[0, "element_type"] == 5
    assert history.loc[0, "fpl_position"] == "AM"


def test_historical_normalization_collapses_exact_duplicate_player_fixture_rows(tmp_path):
    raw_dir = tmp_path / "raw" / "vaastav"
    normalized_dir = tmp_path / "normalized"
    merged = (
        "name,position,team,element,fixture,round,minutes,total_points,value,"
        "was_home,opponent_team,kickoff_time\n"
        "Duplicate Player,MID,Arsenal,11,101,1,90,5,75,true,2,2025-08-01T12:00:00Z\n"
        "Duplicate Player,MID,Arsenal,11,101,1,90,5,75,true,2,2025-08-01T12:00:00Z\n"
    )
    players = "id,code,first_name,second_name,web_name,team,element_type,now_cost\n11,1001,Duplicate,Player,Dup,1,3,75\n"
    write_raw_snapshot(
        raw_dir,
        season="2025-26",
        endpoint_name="merged_gw",
        content=merged.encode(),
        source_url="https://example.test/merged_gw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
    )
    write_raw_snapshot(
        raw_dir,
        season="2025-26",
        endpoint_name="players_raw",
        content=players.encode(),
        source_url="https://example.test/players_raw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
    )

    outputs = normalize_historical(season="2025-26", raw_dir=raw_dir, normalized_dir=normalized_dir)
    history = pd.read_parquet(outputs[0])

    assert len(history) == 1
    assert history.duplicated(["season", "fixture_id", "player_id"]).sum() == 0


def test_historical_normalization_rejects_conflicting_duplicate_player_fixture_rows(tmp_path):
    raw_dir = tmp_path / "raw" / "vaastav"
    normalized_dir = tmp_path / "normalized"
    merged = (
        "name,position,team,element,fixture,round,minutes,total_points,value,"
        "was_home,opponent_team,kickoff_time\n"
        "Duplicate Player,MID,Arsenal,11,101,1,90,5,75,true,2,2025-08-01T12:00:00Z\n"
        "Duplicate Player,MID,Arsenal,11,101,1,45,2,75,true,2,2025-08-01T12:00:00Z\n"
    )
    players = "id,code,first_name,second_name,web_name,team,element_type,now_cost\n11,1001,Duplicate,Player,Dup,1,3,75\n"
    write_raw_snapshot(
        raw_dir,
        season="2025-26",
        endpoint_name="merged_gw",
        content=merged.encode(),
        source_url="https://example.test/merged_gw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
    )
    write_raw_snapshot(
        raw_dir,
        season="2025-26",
        endpoint_name="players_raw",
        content=players.encode(),
        source_url="https://example.test/players_raw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
    )

    with pytest.raises(ValueError, match="conflicting duplicate keys"):
        normalize_historical(season="2025-26", raw_dir=raw_dir, normalized_dir=normalized_dir)


def test_vaastav_missing_required_columns_fails(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/merged_gw.csv"):
            return httpx.Response(200, content=b"element,fixture\n11,101\n")
        return httpx.Response(200, content=fixture_bytes("vaastav/players_raw.csv"))

    ingestor = VaastavIngestor(raw_dir=tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(VaastavDataError, match="missing columns"):
        ingestor.ingest_season(season="2024-25", revision="abc123", refresh=True)
