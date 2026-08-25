from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from fpl_forecast.ingest.fpl_api import (
    BOOTSTRAP_STATIC,
    ELEMENT_SUMMARY,
    EVENT_LIVE,
    FIXTURES,
    FPLApiClient,
)
from fpl_forecast.ingest.snapshots import read_metadata, write_raw_snapshot
from fpl_forecast.operations.current_panel import reconstruct_completed_current_season
from fpl_forecast.operations.model_chain import (
    _completed_minutes_training_frame,
    _official_snapshot_metadata,
    _official_target_players,
    _target_player_fixture_rows,
)
from fpl_forecast.operations.temporal_clubs import (
    HistoricalClubResolutionError,
    HistoricalClubResolver,
)


SEASON = "2026-27"
CUTOFF = "2026-08-29T11:00:00Z"


def test_transfer_zero_minutes_uses_fixture_club_and_preserves_current_club(
    monkeypatch, tmp_path
) -> None:
    case = _case(tmp_path, transferred=True)
    calls: list[str] = []
    client = _network_client(case, calls)
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case["kwargs"], client=client, refresh=True)

    history = pd.read_parquet(result.player_history_path)
    transferred = history.loc[history["player_id"].eq(7)].iloc[0]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert transferred["player_uid"] == "player_code_700"
    assert transferred["fixture_id"] == 11
    assert transferred["historical_team_id"] == 1
    assert transferred["player_team_uid"] == "team_1"
    assert transferred["minutes"] == 0
    assert transferred["current_team_id"] == 3
    assert transferred["current_team_uid"] == "team_3"
    assert transferred["historical_club_resolution_method"] == "element_summary_fixture_history"
    assert "kickoff_time" in history
    assert "kickoff_time_x" not in history
    assert "kickoff_time_y" not in history
    minutes_training = _completed_minutes_training_frame(history)
    assert minutes_training.loc[
        minutes_training["player_id"].eq(7), "player_team_uid"
    ].item() == "team_1"
    assert calls.count("/api/element-summary/7/") == 1
    assert "/api/element-summary/8/" not in calls
    assert manifest["element_summary_player_ids"] == [7]
    assert manifest["historical_club_resolutions"][0]["historical_team_id"] == 1


def test_element_summary_cache_is_reused_without_repeat_network(monkeypatch, tmp_path) -> None:
    case = _case(tmp_path, transferred=True)
    calls: list[str] = []
    _patch_team_identities(monkeypatch)
    first_client = _network_client(case, calls)
    first = reconstruct_completed_current_season(
        **case["kwargs"], client=first_client, refresh=True
    )

    offline_client = FPLApiClient(
        raw_dir=case["raw"],
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"network must not be used for cached replay: {request.url}")
        ),
    )
    second_kwargs = {**case["kwargs"], "run_id": "cached_replay"}
    second = reconstruct_completed_current_season(
        **second_kwargs,
        client=offline_client,
        refresh=False,
    )

    first_history = pd.read_parquet(first.player_history_path)
    second_history = pd.read_parquet(second.player_history_path)
    assert calls.count("/api/element-summary/7/") == 1
    assert first.source_hashes[f"{ELEMENT_SUMMARY}_7"] == second.source_hashes[
        f"{ELEMENT_SUMMARY}_7"
    ]
    assert second_history.loc[second_history["player_id"].eq(7), "historical_team_id"].item() == 1
    assert not first_history.empty


def test_stale_summary_cache_is_refreshed_once_when_fixture_is_missing(
    monkeypatch, tmp_path
) -> None:
    case = _case(tmp_path, transferred=True)
    _snapshot(
        case["raw"],
        f"{ELEMENT_SUMMARY}_7",
        _summary([]),
        datetime(2026, 8, 24, tzinfo=UTC),
    )
    calls: list[str] = []
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(
        **case["kwargs"],
        client=_network_client(case, calls),
        refresh=True,
    )

    history = pd.read_parquet(result.player_history_path)
    raw_snapshots = [
        path
        for path in (case["raw"] / SEASON / f"{ELEMENT_SUMMARY}_7").glob("*.json")
        if not path.name.endswith(".metadata.json")
    ]
    assert calls.count("/api/element-summary/7/") == 1
    assert len(raw_snapshots) == 2
    assert history.loc[history["player_id"].eq(7), "historical_team_id"].item() == 1


def test_element_summary_snapshot_has_checksum_retrieval_and_publication_lineage(
    monkeypatch, tmp_path
) -> None:
    case = _case(tmp_path, transferred=True)
    _patch_team_identities(monkeypatch)
    result = reconstruct_completed_current_season(
        **case["kwargs"], client=_network_client(case, []), refresh=True
    )

    metadata_paths = sorted(
        (case["raw"] / SEASON / f"{ELEMENT_SUMMARY}_7").glob("*.metadata.json")
    )
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    endpoint = f"{ELEMENT_SUMMARY}_7"
    assert metadata["element_id"] == 7
    assert metadata["evidence_purpose"] == "historical_fixture_club_resolution"
    assert metadata["publication_run_id"] == "temporal_case"
    assert len(metadata["sha256"]) == 64
    assert metadata["retrieved_at"]
    assert manifest["source_hashes"][endpoint] == metadata["sha256"]
    assert manifest["official_snapshots"][endpoint]["element_id"] == 7
    assert manifest["official_snapshots"][endpoint]["raw_snapshot_path"]
    forecast_snapshots = _official_snapshot_metadata(case["normalized"] / SEASON)
    assert forecast_snapshots[endpoint]["element_id"] == 7
    assert forecast_snapshots[endpoint]["sha256"] == metadata["sha256"]


def test_removed_player_uses_archived_identity_but_is_not_currently_selectable(
    monkeypatch, tmp_path
) -> None:
    case = _case(tmp_path, removed_player=True)
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case["kwargs"], refresh=False)

    history = pd.read_parquet(result.player_history_path)
    row = history.loc[history["player_id"].eq(7)].iloc[0]
    current_players = pd.read_parquet(
        case["normalized"] / SEASON / "current_players.parquet"
    )
    assert row["player_uid"] == "player_code_700"
    assert row["historical_team_id"] == 1
    assert pd.isna(row["current_team_id"])
    assert row["historical_club_resolution_method"] == "archived_bootstrap_fixture_compatible"
    assert 7 not in set(current_players["player_id"])
    assert "player_code_700" not in {
        f"player_code_{int(code)}" for code in current_players["player_code"].dropna()
    }
    monkeypatch.setattr(
        "fpl_forecast.operations.model_chain._latest_player_history",
        lambda normalized_dir: pd.DataFrame(
            columns=["player_uid", "historical_player_team_uid", "historical_fpl_position"]
        ),
    )
    teams = pd.read_parquet(case["normalized"] / SEASON / "current_teams.parquet")
    team_identity = teams.assign(
        team_uid=teams["team_id"].map(lambda value: f"team_{int(value)}")
    )
    selectable = _official_target_players(
        SEASON,
        normalized_dir=case["normalized"],
        price_variant="base",
        team_identity=team_identity,
        completed_player_fixtures=history,
    )
    assert "player_code_700" not in set(selectable["player_uid"])


def test_reused_element_id_in_cross_season_cache_is_excluded_from_identity_archive(
    monkeypatch, tmp_path
) -> None:
    case = _case(tmp_path)
    prior_season_payload = _bootstrap(current_team=1, season="2025-26")
    prior_season_payload["elements"][0].update(
        code=463748,
        web_name="Prior season player",
    )
    _snapshot(
        case["raw"],
        BOOTSTRAP_STATIC,
        prior_season_payload,
        datetime(2026, 7, 22, tzinfo=UTC),
    )
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case["kwargs"], refresh=False)

    history = pd.read_parquet(result.player_history_path)
    player = history.loc[history["player_id"].eq(7)].iloc[0]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    excluded = manifest["excluded_cross_season_bootstrap_snapshots"]
    assert player["player_uid"] == "player_code_700"
    assert len(excluded) == 1
    assert excluded[0]["requested_season"] == "2026-27"
    assert excluded[0]["inferred_season"] == "2025-26"
    assert excluded[0]["sha256"]
    assert excluded[0]["retrieved_at"]


def test_current_transfer_and_new_player_use_current_club_and_cold_start(
    monkeypatch, tmp_path
) -> None:
    case = _case(tmp_path, transferred=True)
    _patch_team_identities(monkeypatch)
    reconstruction = reconstruct_completed_current_season(
        **case["kwargs"],
        client=_network_client(case, []),
        refresh=True,
    )
    history = pd.read_parquet(reconstruction.player_history_path)
    monkeypatch.setattr(
        "fpl_forecast.operations.model_chain._latest_player_history",
        lambda normalized_dir: pd.DataFrame(
            columns=["player_uid", "historical_player_team_uid", "historical_fpl_position"]
        ),
    )
    teams = pd.read_parquet(case["normalized"] / SEASON / "current_teams.parquet")
    team_identity = teams.assign(
        team_uid=teams["team_id"].map(lambda value: f"team_{int(value)}")
    )

    target_players = _official_target_players(
        SEASON,
        normalized_dir=case["normalized"],
        price_variant="base",
        team_identity=team_identity,
        completed_player_fixtures=history,
    )
    transferred = target_players.loc[target_players["player_uid"].eq("player_code_700")].iloc[0]
    new_player = target_players.loc[target_players["player_uid"].eq("player_code_900")].iloc[0]
    target_fixture = pd.DataFrame(
        [
            {
                "season": SEASON,
                "gameweek": 2,
                "stable_fixture_uid": f"{SEASON}:official_fixture_12",
                "home_team_uid": "team_3",
                "away_team_uid": "team_4",
                "home_team_name": "Team 3",
                "away_team_name": "Team 4",
                "home_team_short_name": "T3",
                "away_team_short_name": "T4",
                "kickoff_time": "2026-08-29T15:00:00Z",
            }
        ]
    )
    target_rows = _target_player_fixture_rows(
        target_fixture,
        target_players,
        as_of=pd.Timestamp(CUTOFF),
    )

    assert transferred["player_team_uid"] == "team_3"
    assert transferred["transferred_player"]
    assert transferred["lineage_note"] == "transferred_player"
    assert not transferred["cold_start_no_history"]
    assert new_player["player_team_uid"] == "team_3"
    assert new_player["cold_start_no_history"]
    assert new_player["lineage_note"] == "new_player_cold_start"
    assert {"player_code_700", "player_code_900"}.issubset(set(target_rows["player_uid"]))
    assert target_rows.loc[
        target_rows["player_uid"].isin(["player_code_700", "player_code_900"]),
        "player_team_uid",
    ].eq("team_3").all()


def test_multiple_transfers_are_resolved_independently_per_fixture(tmp_path) -> None:
    raw = tmp_path / "raw"
    bootstrap = _bootstrap(current_team=5)
    fixtures = [
        _fixture(11, 1, 1, 2, finished=True),
        _fixture(12, 2, 4, 3, finished=True),
        _fixture(13, 3, 5, 6, finished=False),
    ]
    bootstrap_record = _snapshot(raw, BOOTSTRAP_STATIC, bootstrap, datetime(2026, 8, 25, tzinfo=UTC))
    summary = _summary(
        [
            _history(11, opponent=2, was_home=True, gameweek=1),
            _history(12, opponent=4, was_home=False, gameweek=2),
        ]
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/element-summary/7/"):
            return httpx.Response(200, json=summary)
        raise AssertionError(f"unexpected URL {request.url}")

    client = FPLApiClient(raw_dir=raw, transport=httpx.MockTransport(handler))
    resolver = HistoricalClubResolver(
        season=SEASON,
        bootstrap_payload=bootstrap,
        bootstrap_metadata=read_metadata(bootstrap_record.raw_path),
        bootstrap_raw_path=str(bootstrap_record.raw_path),
        fixtures_payload=fixtures,
        raw_fpl_dir=raw,
        normalized_dir=tmp_path / "normalized",
        client=client,
        information_cutoff="2026-09-12T11:00:00Z",
        allow_network=True,
        source_context={"publication_run_id": "multiple_transfers"},
    )
    first = resolver.resolve_event(
        gameweek=1,
        payload=_event_live([_event_element(7, 11, minutes=90)]),
        eligible_fixture_ids={11},
    )
    second = resolver.resolve_event(
        gameweek=2,
        payload=_event_live([_event_element(7, 12, minutes=90)]),
        eligible_fixture_ids={12},
    )

    assert first.club_assignments[(7, 11)].historical_team_id == 1
    assert second.club_assignments[(7, 12)].historical_team_id == 3
    assert first.club_assignments[(7, 11)].current_team_id == 5
    assert second.club_assignments[(7, 12)].current_team_id == 5
    assert calls == ["/api/element-summary/7/"]


def test_contradictory_summary_fails_with_player_fixture_and_candidates(monkeypatch, tmp_path) -> None:
    case = _case(tmp_path, transferred=True)
    bad_summary = _summary([_history(11, opponent=1, was_home=True, gameweek=1)])
    _patch_team_identities(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/event/1/live/"):
            return httpx.Response(200, json=case["event_live"])
        if request.url.path.endswith("/element-summary/7/"):
            return httpx.Response(200, json=bad_summary)
        raise AssertionError(f"unexpected URL {request.url}")

    client = FPLApiClient(raw_dir=case["raw"], transport=httpx.MockTransport(handler))
    with pytest.raises(
        HistoricalClubResolutionError,
        match=r"player=7, fixture=11, candidate_clubs=\[1, 2\].*contradicts",
    ):
        reconstruct_completed_current_season(
            **case["kwargs"], client=client, refresh=True
        )


def test_partial_postponed_event_includes_only_available_fixture(monkeypatch, tmp_path) -> None:
    case = _case(tmp_path, partial_postponement=True)
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case["kwargs"], refresh=False)

    players = pd.read_parquet(result.player_history_path)
    teams = pd.read_parquet(result.team_history_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert set(players["fixture_id"]) == {11}
    assert set(teams["source_fixture_id"]) == {11}
    assert manifest["deferred_fixture_ids"] == [14]
    assert manifest["events"][0]["included_fixture_ids"] == [11]
    assert manifest["events"][0]["deferred_fixture_ids"] == [14]
    assert manifest["events"][0]["event_finished"] is False


def test_postponed_fixture_becomes_history_only_after_final_and_available(
    monkeypatch, tmp_path
) -> None:
    before = _case(tmp_path / "before", all_prior_postponed=True)
    after = _case(tmp_path / "after", played_postponement=True)
    _patch_team_identities(monkeypatch)

    excluded = reconstruct_completed_current_season(**before["kwargs"], refresh=False)
    included = reconstruct_completed_current_season(**after["kwargs"], refresh=False)

    assert excluded.player_rows == 0
    assert json.loads(excluded.manifest_path.read_text(encoding="utf-8"))[
        "deferred_fixture_ids"
    ] == [11]
    included_players = pd.read_parquet(included.player_history_path)
    assert set(included_players["fixture_id"]) == {11}
    assert included_players["source_available_time"].lt(pd.Timestamp(CUTOFF)).all()


def _case(
    root: Path,
    *,
    transferred: bool = False,
    removed_player: bool = False,
    partial_postponement: bool = False,
    all_prior_postponed: bool = False,
    played_postponement: bool = False,
) -> dict[str, object]:
    raw = root / "raw"
    normalized = root / "normalized"
    season_dir = normalized / SEASON
    season_dir.mkdir(parents=True)
    current_team = 3 if transferred else 1
    current_bootstrap = _bootstrap(current_team=current_team, include_player=not removed_player)
    archive_bootstrap = _bootstrap(current_team=1)
    if removed_player:
        _snapshot(
            raw,
            BOOTSTRAP_STATIC,
            archive_bootstrap,
            datetime(2026, 8, 20, tzinfo=UTC),
        )
    _snapshot(
        raw,
        BOOTSTRAP_STATIC,
        current_bootstrap,
        datetime(2026, 8, 25, tzinfo=UTC),
    )
    first_finished = not all_prior_postponed
    if played_postponement:
        first_finished = True
    fixtures = [
        _fixture(11, 1, 1, 2, finished=first_finished),
        _fixture(12, 2, 3, 4, finished=False),
    ]
    if partial_postponement:
        fixtures.append(_fixture(14, 1, 3, 4, finished=False))
    _snapshot(raw, FIXTURES, fixtures, datetime(2026, 8, 25, tzinfo=UTC))
    event_elements = [
        _event_element(7, 11, minutes=0 if transferred else 90),
        _event_element(8, 11, minutes=90),
    ]
    if partial_postponement:
        event_elements.append(_event_element(9, 14, minutes=0))
    event_live = _event_live(event_elements)
    _snapshot(
        raw,
        f"{EVENT_LIVE}_1",
        event_live,
        datetime(2026, 8, 25, 1, tzinfo=UTC),
    )
    _write_normalized(
        season_dir,
        bootstrap=current_bootstrap,
        fixtures=fixtures,
        event_finished=not partial_postponement and not all_prior_postponed,
    )
    return {
        "raw": raw,
        "normalized": normalized,
        "event_live": event_live,
        "summary": _summary([_history(11, opponent=2, was_home=True, gameweek=1)]),
        "kwargs": {
            "season": SEASON,
            "target_gameweek": 2,
            "information_cutoff": CUTOFF,
            "raw_fpl_dir": raw,
            "normalized_dir": normalized,
            "run_id": "temporal_case",
            "requested_gameweek": 2,
            "git_commit": "a" * 40,
            "clean_source": True,
        },
    }


def _network_client(case: dict[str, object], calls: list[str]) -> FPLApiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/event/1/live/"):
            return httpx.Response(200, json=case["event_live"])
        if request.url.path.endswith("/element-summary/7/"):
            return httpx.Response(200, json=case["summary"])
        raise AssertionError(f"unexpected URL {request.url}")

    return FPLApiClient(
        raw_dir=case["raw"],
        transport=httpx.MockTransport(handler),
    )


def _bootstrap(
    *,
    current_team: int,
    include_player: bool = True,
    season: str = SEASON,
) -> dict:
    start_year = int(season[:4])
    elements = [
        {
            "id": 8,
            "code": 800,
            "web_name": "Compatible",
            "team": 2,
            "element_type": 2,
            "now_cost": 50,
            "status": "a",
            "can_select": True,
            "removed": False,
        },
        {
            "id": 9,
            "code": 900,
            "web_name": "Current only",
            "team": 3,
            "element_type": 3,
            "now_cost": 50,
            "status": "a",
            "can_select": True,
            "removed": False,
        },
    ]
    if include_player:
        elements.insert(
            0,
            {
                "id": 7,
                "code": 700,
                "web_name": "Transferred",
                "team": current_team,
                "element_type": 3,
                "now_cost": 65,
                "status": "a",
                "can_select": True,
                "removed": False,
            },
        )
    return {
        "events": [
            {
                "id": gameweek,
                "deadline_time": (
                    f"{start_year}-08-{min(10 + gameweek, 28):02d}T11:00:00Z"
                    if gameweek < 20
                    else f"{start_year + 1}-05-{min(gameweek, 28):02d}T11:00:00Z"
                ),
            }
            for gameweek in range(1, 39)
        ],
        "game_settings": {},
        "phases": [],
        "teams": [
            {"id": team_id, "code": team_id, "name": f"Team {team_id}", "short_name": f"T{team_id}", "strength": 3}
            for team_id in range(1, 7)
        ],
        "total_players": len(elements),
        "elements": elements,
        "element_stats": [],
        "element_types": [],
    }


def _fixture(
    fixture_id: int,
    gameweek: int,
    home: int,
    away: int,
    *,
    finished: bool,
) -> dict:
    kickoff = pd.Timestamp("2026-08-22T15:00:00Z") + pd.Timedelta(days=7 * (gameweek - 1))
    return {
        "id": fixture_id,
        "code": fixture_id,
        "event": gameweek,
        "team_h": home,
        "team_a": away,
        "kickoff_time": kickoff.isoformat(),
        "finished": finished,
        "started": finished,
        "finished_provisional": finished,
        "team_h_score": 1 if finished else None,
        "team_a_score": 0 if finished else None,
    }


def _event_element(player_id: int, fixture_id: int, *, minutes: int) -> dict:
    points = 2 if minutes >= 60 else 0
    stats = _zero_stats()
    stats.update(
        {
            "minutes": minutes,
            "starts": int(minutes >= 60),
            "total_points": points,
            "played": minutes > 0,
        }
    )
    return {
        "id": player_id,
        "stats": stats,
        "explain": [
            {
                "fixture": fixture_id,
                "stats": [
                    {
                        "identifier": "minutes",
                        "value": minutes,
                        "points": points,
                        "points_modification": 0,
                    }
                ],
            }
        ],
    }


def _event_live(elements: list[dict]) -> dict:
    return {"elements": elements}


def _summary(history: list[dict]) -> dict:
    return {
        "fixtures": [],
        "history": history,
        "history_past": [{"season_name": "2025/26", "element_code": 700}],
    }


def _history(
    fixture_id: int,
    *,
    opponent: int,
    was_home: bool,
    gameweek: int,
) -> dict:
    kickoff = pd.Timestamp("2026-08-22T15:00:00Z") + pd.Timedelta(days=7 * (gameweek - 1))
    return {
        "element": 7,
        "fixture": fixture_id,
        "opponent_team": opponent,
        "was_home": was_home,
        "round": gameweek,
        "kickoff_time": kickoff.isoformat(),
        "minutes": 0,
        "starts": 0,
        "total_points": 0,
    }


def _zero_stats() -> dict:
    return {
        "minutes": 0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "saves": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "own_goals": 0,
        "bonus": 0,
        "bps": 0,
        "defensive_contribution": 0,
        "starts": 0,
        "total_points": 0,
        "played": False,
        "in_dreamteam": False,
    }


def _snapshot(raw: Path, endpoint: str, payload: object, retrieved_at: datetime):
    return write_raw_snapshot(
        raw,
        season=SEASON,
        endpoint_name=endpoint,
        content=json.dumps(payload, sort_keys=True).encode(),
        source_url=(
            "https://fantasy.premierleague.com/api/element-summary/7/"
            if endpoint == f"{ELEMENT_SUMMARY}_7"
            else f"https://fantasy.premierleague.com/api/{endpoint}/"
        ),
        http_status=200,
        target_season=SEASON,
        source="fpl_api",
        source_version=SEASON,
        retrieved_at=retrieved_at,
    )


def _write_normalized(
    season_dir: Path,
    *,
    bootstrap: dict,
    fixtures: list[dict],
    event_finished: bool,
) -> None:
    pd.DataFrame(
        [
            {
                "gameweek": 1,
                "name": "Gameweek 1",
                "deadline_time": "2026-08-21T17:30:00Z",
                "finished": event_finished,
                "data_checked": event_finished,
                "is_current": True,
                "is_next": False,
                "released": True,
            },
            {
                "gameweek": 2,
                "name": "Gameweek 2",
                "deadline_time": CUTOFF,
                "finished": False,
                "data_checked": False,
                "is_current": False,
                "is_next": True,
                "released": True,
            },
        ]
    ).to_parquet(season_dir / "current_events.parquet", index=False)
    pd.DataFrame(fixtures).rename(
        columns={
            "id": "fixture_id",
            "code": "fixture_code",
            "event": "gameweek",
            "team_h": "home_team_id",
            "team_a": "away_team_id",
        }
    ).assign(
        season=SEASON,
        source="fpl_api",
        source_version=SEASON,
        retrieved_at="2026-08-25T00:00:00Z",
        raw_snapshot_path="raw/fixtures.json",
    ).to_parquet(season_dir / "current_fixtures.parquet", index=False)
    pd.DataFrame(bootstrap["elements"]).rename(
        columns={
            "id": "player_id",
            "code": "player_code",
            "team": "team_id",
            "element_type": "position_id",
            "now_cost": "price_tenths",
        }
    ).assign(
        position=lambda frame: frame["position_id"].map({2: "DEF", 3: "MID"}),
        entity_type="player",
        season=SEASON,
        source="fpl_api",
        source_version=SEASON,
        retrieved_at="2026-08-25T00:00:00Z",
        raw_snapshot_path="raw/bootstrap.json",
    ).to_parquet(season_dir / "current_players.parquet", index=False)
    pd.DataFrame(bootstrap["teams"]).rename(
        columns={"id": "team_id", "code": "team_code", "name": "team_name"}
    ).assign(
        season=SEASON,
        source="fpl_api",
        source_version=SEASON,
        retrieved_at="2026-08-25T00:00:00Z",
        raw_snapshot_path="raw/bootstrap.json",
    ).to_parquet(season_dir / "current_teams.parquet", index=False)


def _patch_team_identities(monkeypatch) -> None:
    monkeypatch.setattr(
        "fpl_forecast.operations.current_panel._current_team_identities",
        lambda teams, **kwargs: teams.assign(
            team_uid=teams["team_id"].map(lambda value: f"team_{int(value)}")
        ),
    )
