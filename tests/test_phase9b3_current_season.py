from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from fpl_forecast.ingest.fpl_api import BOOTSTRAP_STATIC, EVENT_LIVE, FIXTURES
from fpl_forecast.ingest.snapshots import write_raw_snapshot
from fpl_forecast.operations.current_panel import reconstruct_completed_current_season
from fpl_forecast.operations.live_results import normalize_event_live
from fpl_forecast.operations.publication_pipeline import PublicationError, resolve_target_gameweek


def test_gw2_reconstructs_one_completed_normal_event(monkeypatch, tmp_path) -> None:
    case = _official_case(tmp_path, target_gameweek=2)
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case)

    players = pd.read_parquet(result.player_history_path)
    teams = pd.read_parquet(result.team_history_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.event_count == 1
    assert len(players) == 2
    assert len(teams) == 1
    assert set(players["gameweek"]) == {1}
    assert set(teams["gameweek"]) == {1}
    assert manifest["temporal_policy"] == "source_available_time < information_cutoff"
    assert manifest["publication_run_id"] == "official_synthetic_gw2"
    assert set(result.source_hashes) == {BOOTSTRAP_STATIC, FIXTURES, f"{EVENT_LIVE}_1"}


def test_gw3_reconstructs_two_completed_events_without_target_leakage(monkeypatch, tmp_path) -> None:
    case = _official_case(tmp_path, target_gameweek=3)
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case)
    players = pd.read_parquet(result.player_history_path)

    assert result.event_count == 2
    assert set(players["gameweek"]) == {1, 2}
    assert players["gameweek"].max() < 3
    assert f"{EVENT_LIVE}_3" not in result.source_hashes


def test_reconstruction_rejects_event_live_retrieved_at_target_cutoff(monkeypatch, tmp_path) -> None:
    case = _official_case(
        tmp_path,
        target_gameweek=2,
        event_retrieved_at=datetime(2026, 8, 29, 11, tzinfo=UTC),
    )
    _patch_team_identities(monkeypatch)

    with pytest.raises(ValueError, match="not before target cutoff"):
        reconstruct_completed_current_season(**case)


def test_reconstruction_rejects_missing_event_live(monkeypatch, tmp_path) -> None:
    case = _official_case(tmp_path, target_gameweek=2, omit_event_live={1})
    _patch_team_identities(monkeypatch)

    with pytest.raises(Exception, match="event_live_1"):
        reconstruct_completed_current_season(**case)


def test_reconstruction_rejects_prior_event_not_data_checked(monkeypatch, tmp_path) -> None:
    case = _official_case(tmp_path, target_gameweek=2)
    _patch_team_identities(monkeypatch)
    events_path = case["normalized_dir"] / "2026-27" / "current_events.parquet"
    events = pd.read_parquet(events_path)
    events.loc[events["gameweek"].eq(1), "data_checked"] = False
    events.to_parquet(events_path, index=False)

    with pytest.raises(ValueError, match="not finished and data-checked"):
        reconstruct_completed_current_season(**case)


def test_reconstruction_rejects_prior_fixture_not_provisional(monkeypatch, tmp_path) -> None:
    case = _official_case(tmp_path, target_gameweek=2)
    _patch_team_identities(monkeypatch)
    fixtures_path = case["normalized_dir"] / "2026-27" / "current_fixtures.parquet"
    fixtures = pd.read_parquet(fixtures_path)
    fixtures.loc[fixtures["gameweek"].eq(1), "finished_provisional"] = False
    fixtures.to_parquet(fixtures_path, index=False)

    with pytest.raises(ValueError, match="not fully finalized"):
        reconstruct_completed_current_season(**case)


def test_event_live_rejects_fixture_assigned_to_another_gameweek(tmp_path) -> None:
    bootstrap = _bootstrap(2)
    fixtures = _fixtures(2)
    fixtures[0]["event"] = 2

    with pytest.raises(ValueError, match="belongs to gameweek 2, not 1"):
        normalize_event_live(
            season="2026-27",
            gameweek=1,
            payload=_event_live(1),
            retrieved_at="2026-08-17T00:00:00Z",
            raw_snapshot_path=str(tmp_path / "event.json"),
            bootstrap_payload=bootstrap,
            fixtures_payload=fixtures,
        )


def test_event_live_rejects_duplicate_player_fixture_keys(tmp_path) -> None:
    payload = _event_live(1)
    payload["elements"][0]["explain"].append(payload["elements"][0]["explain"][0].copy())

    with pytest.raises(ValueError, match="duplicate player-fixture keys"):
        normalize_event_live(
            season="2026-27",
            gameweek=1,
            payload=payload,
            retrieved_at="2026-08-17T00:00:00Z",
            raw_snapshot_path=str(tmp_path / "event.json"),
            bootstrap_payload=_bootstrap(2),
            fixtures_payload=_fixtures(2),
        )


def test_completed_double_gameweek_preserves_fixture_grain(monkeypatch, tmp_path) -> None:
    case = _official_case(tmp_path, target_gameweek=2, double_gameweek=True)
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case)
    players = pd.read_parquet(result.player_history_path)
    player = players.loc[players["player_id"].eq(7)]

    assert len(player) == 2
    assert player["fixture_id"].nunique() == 2
    assert int(player["total_points"].sum()) == 4


def test_assistant_managers_are_archived_but_excluded_from_training(monkeypatch, tmp_path) -> None:
    case = _official_case(tmp_path, target_gameweek=2, include_assistant=True)
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case)
    players = pd.read_parquet(result.player_history_path)

    assert set(players["entity_type"]) == {"player"}
    assert 99 not in set(players["player_id"])


def test_legitimate_prior_blank_event_is_recorded_without_synthetic_rows(monkeypatch, tmp_path) -> None:
    case = _official_case(tmp_path, target_gameweek=3, blank_events={2})
    _patch_team_identities(monkeypatch)

    result = reconstruct_completed_current_season(**case)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.blank_events == (2,)
    assert set(pd.read_parquet(result.player_history_path)["gameweek"]) == {1}
    assert manifest["events"][1]["blank_event"] is True
    assert manifest["events"][1]["player_fixture_rows"] == 0


def test_globally_blank_target_fails_with_precise_unsupported_status() -> None:
    events = _event_frame(2)
    fixtures = pd.DataFrame(_fixtures(2))
    fixtures = fixtures.loc[fixtures["event"].ne(2)].rename(
        columns={"event": "gameweek", "id": "fixture_id"}
    )

    with pytest.raises(PublicationError, match="globally blank official event"):
        resolve_target_gameweek(
            season="2026-27",
            events=events,
            fixtures=fixtures,
            requested_gameweek=2,
            now=datetime(2026, 8, 18, tzinfo=UTC),
        )


def test_reconstruction_fingerprint_inputs_change_with_event_snapshot(monkeypatch, tmp_path) -> None:
    first_case = _official_case(tmp_path / "first", target_gameweek=2, event_points=2)
    second_case = _official_case(tmp_path / "second", target_gameweek=2, event_points=3)
    _patch_team_identities(monkeypatch)

    first = reconstruct_completed_current_season(**first_case)
    second = reconstruct_completed_current_season(**second_case)

    assert first.source_hashes[f"{EVENT_LIVE}_1"] != second.source_hashes[f"{EVENT_LIVE}_1"]


def test_identical_official_inputs_reconstruct_deterministically(monkeypatch, tmp_path) -> None:
    first_case = _official_case(tmp_path / "first", target_gameweek=2)
    second_case = _official_case(tmp_path / "second", target_gameweek=2)
    _patch_team_identities(monkeypatch)

    first = reconstruct_completed_current_season(**first_case)
    second = reconstruct_completed_current_season(**second_case)
    first_players = pd.read_parquet(first.player_history_path).drop(
        columns=["raw_snapshot_path", "fixtures_raw_snapshot_path"]
    )
    second_players = pd.read_parquet(second.player_history_path).drop(
        columns=["raw_snapshot_path", "fixtures_raw_snapshot_path"]
    )
    first_teams = pd.read_parquet(first.team_history_path).drop(columns=["raw_snapshot_path"])
    second_teams = pd.read_parquet(second.team_history_path).drop(columns=["raw_snapshot_path"])

    pd.testing.assert_frame_equal(first_players, second_players)
    pd.testing.assert_frame_equal(first_teams, second_teams)
    assert first.source_hashes == second.source_hashes


def test_workflow_requires_plain_integer_gameweek_and_has_no_schedule() -> None:
    workflow = Path(".github/workflows/deploy-pages.yml").read_text(encoding="utf-8")

    assert "Target gameweek must be a plain positive integer" in workflow
    assert 'args+=(--target-gameweek "$TARGET_GAMEWEEK")' in workflow
    assert "schedule:" not in workflow
    assert "--run-id \"${{ steps.inputs.outputs.run_id }}\"" in workflow


def _official_case(
    root: Path,
    *,
    target_gameweek: int,
    event_retrieved_at: datetime = datetime(2026, 8, 17, tzinfo=UTC),
    omit_event_live: set[int] | None = None,
    double_gameweek: bool = False,
    include_assistant: bool = False,
    blank_events: set[int] | None = None,
    event_points: int = 2,
) -> dict:
    omit_event_live = omit_event_live or set()
    blank_events = blank_events or set()
    raw = root / "raw"
    normalized = root / "normalized"
    season_dir = normalized / "2026-27"
    season_dir.mkdir(parents=True)
    bootstrap = _bootstrap(target_gameweek, include_assistant=include_assistant)
    fixtures = _fixtures(target_gameweek, double_gameweek=double_gameweek, blank_events=blank_events)
    base_retrieved = datetime(2026, 8, 16, tzinfo=UTC)
    _snapshot(raw, BOOTSTRAP_STATIC, bootstrap, base_retrieved)
    _snapshot(raw, FIXTURES, fixtures, base_retrieved)
    _normalized_current(
        season_dir,
        bootstrap=bootstrap,
        fixtures=fixtures,
        target_gameweek=target_gameweek,
        blank_events=blank_events,
    )
    for gameweek in range(1, target_gameweek):
        if gameweek in omit_event_live:
            continue
        payload = (
            {"elements": [{**element, "stats": _zero_stats(), "explain": []} for element in bootstrap["elements"]]}
            if gameweek in blank_events
            else _event_live(
                gameweek,
                double_gameweek=double_gameweek and gameweek == 1,
                include_assistant=include_assistant,
                event_points=event_points,
            )
        )
        _snapshot(
            raw,
            f"{EVENT_LIVE}_{gameweek}",
            payload,
            event_retrieved_at.replace(day=event_retrieved_at.day + gameweek - 1),
        )
    return {
        "season": "2026-27",
        "target_gameweek": target_gameweek,
        "information_cutoff": _deadline(target_gameweek),
        "raw_fpl_dir": raw,
        "normalized_dir": normalized,
        "run_id": f"official_synthetic_gw{target_gameweek}",
        "requested_gameweek": target_gameweek,
        "git_commit": "a" * 40,
        "clean_source": True,
        "refresh": False,
    }


def _snapshot(raw: Path, endpoint: str, payload, retrieved_at: datetime) -> None:
    write_raw_snapshot(
        raw,
        season="2026-27",
        endpoint_name=endpoint,
        content=json.dumps(payload, sort_keys=True).encode(),
        source_url=f"https://fantasy.premierleague.com/api/{endpoint}/",
        http_status=200,
        target_season="2026-27",
        source="fpl_api",
        source_version="2026-27",
        retrieved_at=retrieved_at,
    )


def _bootstrap(target_gameweek: int, *, include_assistant: bool = False) -> dict:
    elements = [
        {"id": 7, "code": 700, "web_name": "Home", "team": 1, "element_type": 3},
        {"id": 8, "code": 800, "web_name": "Away", "team": 2, "element_type": 2},
    ]
    if include_assistant:
        elements.append({"id": 99, "code": 9900, "web_name": "Assistant", "team": 1, "element_type": 5})
    return {
        "events": _event_frame(target_gameweek).rename(columns={"gameweek": "id"}).to_dict("records"),
        "game_settings": {},
        "phases": [],
        "teams": [
            {"id": 1, "code": 1, "name": "Home FC", "short_name": "HOM", "strength": 3},
            {"id": 2, "code": 2, "name": "Away FC", "short_name": "AWY", "strength": 3},
        ],
        "total_players": len(elements),
        "elements": elements,
        "element_stats": [],
        "element_types": [],
    }


def _fixtures(
    target_gameweek: int,
    *,
    double_gameweek: bool = False,
    blank_events: set[int] | None = None,
) -> list[dict]:
    blank_events = blank_events or set()
    rows = []
    fixture_id = 10
    for gameweek in range(1, target_gameweek + 1):
        if gameweek in blank_events:
            continue
        count = 2 if double_gameweek and gameweek == 1 else 1
        for _ in range(count):
            fixture_id += 1
            completed = gameweek < target_gameweek
            rows.append(
                {
                    "id": fixture_id,
                    "code": fixture_id,
                    "event": gameweek,
                    "team_h": 1,
                    "team_a": 2,
                    "kickoff_time": _kickoff(gameweek),
                    "finished": completed,
                    "started": completed,
                    "finished_provisional": completed,
                    "team_h_score": 1 if completed else None,
                    "team_a_score": 0 if completed else None,
                }
            )
    return rows


def _event_live(
    gameweek: int,
    *,
    double_gameweek: bool = False,
    include_assistant: bool = False,
    event_points: int = 2,
) -> dict:
    first_fixture = 11 + (gameweek - 1) * (2 if double_gameweek else 1)
    fixture_ids = [first_fixture, first_fixture + 1] if double_gameweek else [first_fixture]
    elements = []
    for player_id in ([7, 8, 99] if include_assistant else [7, 8]):
        explains = [
            {
                "fixture": fixture_id,
                "stats": [
                    {
                        "identifier": "minutes",
                        "value": 90,
                        "points": event_points,
                        "points_modification": 0,
                    }
                ],
            }
            for fixture_id in fixture_ids
        ]
        stats = _zero_stats()
        stats["minutes"] = 90 * len(fixture_ids)
        stats["starts"] = len(fixture_ids)
        stats["total_points"] = event_points * len(fixture_ids)
        elements.append({"id": player_id, "stats": stats, "explain": explains})
    return {"elements": elements}


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


def _event_frame(target_gameweek: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameweek": gameweek,
                "name": f"Gameweek {gameweek}",
                "deadline_time": _deadline(gameweek),
                "finished": gameweek < target_gameweek,
                "data_checked": gameweek < target_gameweek,
                "is_current": gameweek == target_gameweek - 1,
                "is_next": gameweek == target_gameweek,
                "released": True,
            }
            for gameweek in range(1, target_gameweek + 1)
        ]
    )


def _normalized_current(
    season_dir: Path,
    *,
    bootstrap: dict,
    fixtures: list[dict],
    target_gameweek: int,
    blank_events: set[int],
) -> None:
    _event_frame(target_gameweek).to_parquet(season_dir / "current_events.parquet", index=False)
    pd.DataFrame(fixtures).rename(
        columns={
            "id": "fixture_id",
            "code": "fixture_code",
            "event": "gameweek",
            "team_h": "home_team_id",
            "team_a": "away_team_id",
        }
    ).assign(
        season="2026-27",
        source="fpl_api",
        source_version="2026-27",
        raw_snapshot_path="raw/fixtures.json",
    ).to_parquet(season_dir / "current_fixtures.parquet", index=False)
    pd.DataFrame(bootstrap["elements"]).rename(
        columns={
            "id": "player_id",
            "code": "player_code",
            "team": "team_id",
            "element_type": "position_id",
        }
    ).assign(
        position=lambda frame: frame["position_id"].map({2: "DEF", 3: "MID", 5: "AM"}),
        price_tenths=50,
        entity_type=lambda frame: frame["position_id"].map(
            lambda value: "assistant_manager" if value == 5 else "player"
        ),
        season="2026-27",
    ).to_parquet(season_dir / "current_players.parquet", index=False)
    pd.DataFrame(bootstrap["teams"]).rename(
        columns={"id": "team_id", "code": "team_code", "name": "team_name"}
    ).assign(
        season="2026-27",
        source="fpl_api",
        source_version="2026-27",
        retrieved_at="2026-08-16T00:00:00Z",
        raw_snapshot_path="raw/bootstrap.json",
    ).to_parquet(season_dir / "current_teams.parquet", index=False)


def _patch_team_identities(monkeypatch) -> None:
    monkeypatch.setattr(
        "fpl_forecast.operations.current_panel._current_team_identities",
        lambda teams, **kwargs: teams.assign(
            team_uid=teams["team_id"].map({1: "team_home", 2: "team_away"})
        ),
    )


def _deadline(gameweek: int) -> str:
    return (pd.Timestamp("2026-08-15T11:00:00Z") + pd.Timedelta(days=7 * gameweek)).isoformat()


def _kickoff(gameweek: int) -> str:
    return (pd.Timestamp(_deadline(gameweek)) + pd.Timedelta(hours=4)).isoformat()
