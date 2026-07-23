from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from fpl_forecast.dashboard.app import run_dashboard
from fpl_forecast.ingest.fpl_api import BOOTSTRAP_STATIC, FIXTURES
from fpl_forecast.ingest.snapshots import write_raw_snapshot
from fpl_forecast.operations.config import LATEST_SUCCESSFUL_PATH, LOCK_PATH, STATUS_PATH
from fpl_forecast.operations.current_panel import build_current_player_fixture_history
from fpl_forecast.operations.launch import check_season_launch
from fpl_forecast.operations.live_results import finalized_fixture_ids, normalize_event_live
from fpl_forecast.operations.locking import RefreshLock
from fpl_forecast.operations.model_chain import run_operational_model_chain
from fpl_forecast.operations.orchestrator import refresh_operational
from fpl_forecast.operations.state import OperationalStateName


def test_old_season_payload_returns_waiting_and_target_payload_transitions(tmp_path) -> None:
    _write_current_snapshots(tmp_path, stored_season="2026-27", payload_season="2025-26")

    waiting = check_season_launch(season="2026-27", raw_dir=tmp_path)

    assert waiting.status.state == OperationalStateName.WAITING_FOR_SEASON_LAUNCH
    assert waiting.status.inferred_official_season == "2025-26"

    _write_current_snapshots(tmp_path, stored_season="2026-27", payload_season="2026-27", stamp=2)
    ready = check_season_launch(season="2026-27", raw_dir=tmp_path)

    assert ready.status.state == OperationalStateName.READY_TO_REFRESH
    assert ready.status.inferred_official_season == "2026-27"


def test_changed_rules_enter_review_state(tmp_path) -> None:
    _write_current_snapshots(tmp_path, stored_season="2026-27", payload_season="2026-27", budget=995)

    result = check_season_launch(season="2026-27", raw_dir=tmp_path)

    assert result.status.state == OperationalStateName.NEEDS_RULE_REVIEW
    assert "squad_total_spend" in result.rule_diff["material_changes"]


def test_live_result_normalization_preserves_fixture_grain_and_unknown_start() -> None:
    payload = {
        "elements": [
            {
                "id": 7,
                "explain": [
                    {
                        "fixture": 11,
                        "stats": [
                            {"identifier": "minutes", "value": 90, "points": 2},
                            {"identifier": "goals_scored", "value": 1, "points": 5},
                            {"identifier": "total_points", "value": 8, "points": 8},
                        ],
                    },
                    {
                        "fixture": 12,
                        "stats": [
                            {"identifier": "minutes", "value": 30, "points": 1},
                            {"identifier": "total_points", "value": 1, "points": 1},
                        ],
                    },
                ],
            }
        ]
    }

    frame = normalize_event_live(
        season="2026-27",
        gameweek=1,
        payload=payload,
        retrieved_at="2026-08-16T20:00:00Z",
        raw_snapshot_path="mock/event_live.json",
    )

    assert len(frame) == 2
    assert set(frame["fixture_id"]) == {11, 12}
    assert frame["exact_start"].isna().all()
    assert int(frame.loc[frame["fixture_id"].eq(11), "total_points"].iloc[0]) == 8


def test_finalized_fixture_policy_requires_finished_and_provisional() -> None:
    fixtures = [
        {"id": 1, "finished": True, "finished_provisional": True},
        {"id": 2, "finished": True, "finished_provisional": False},
        {"id": 3, "finished": False, "finished_provisional": False},
    ]

    assert finalized_fixture_ids(fixtures) == {1}


def test_current_panel_rebuild_is_idempotent_and_excludes_assistant_managers() -> None:
    fixtures = pd.DataFrame(
        [{"fixture_id": 11, "gameweek": 1, "kickoff_time": "2026-08-15T12:00:00Z", "finished": True, "finished_provisional": True}]
    )
    players = pd.DataFrame(
        [
            {"player_id": 7, "player_code": 77, "web_name": "Player", "team_id": 1, "position": "MID", "price_tenths": 75},
            {"player_id": 8, "player_code": 88, "web_name": "Assistant", "team_id": 1, "position": "AM", "price_tenths": 50},
        ]
    )
    live = pd.DataFrame(
        [
            {"season": "2026-27", "gameweek": 1, "player_id": 7, "fixture_id": 11, "minutes": 90, "total_points": 5},
            {"season": "2026-27", "gameweek": 1, "player_id": 7, "fixture_id": 11, "minutes": 90, "total_points": 5},
            {"season": "2026-27", "gameweek": 1, "player_id": 8, "fixture_id": 11, "minutes": 0, "total_points": 0},
        ]
    )

    panel = build_current_player_fixture_history(fixtures=fixtures, live_results=live, players=players)

    assert len(panel) == 1
    assert panel.iloc[0]["fpl_position"] == "MID"


def test_mock_operational_refresh_noop_failure_and_lock_behaviors() -> None:
    first = refresh_operational(season="2026-27", mock_launch=True, force=True, run_id="phase8_test_success")
    assert first.status.state == OperationalStateName.SUCCEEDED
    before = json.loads(LATEST_SUCCESSFUL_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_lineage"]["team_model"] == "T2_REGULARIZED_ATTACK_DEFENCE"
    assert manifest["model_lineage"]["decision_run_id"] == "phase8_test_success_decision_current"
    freshness = json.loads((first.run_dir / "data_freshness.json").read_text(encoding="utf-8"))
    assert freshness["source"] == "mocked target-season production model chain"

    second = refresh_operational(season="2026-27", mock_launch=True)
    assert second.no_op
    assert second.status.state == OperationalStateName.SUCCEEDED

    failed = refresh_operational(
        season="2026-27",
        mock_launch=True,
        force=True,
        run_id="phase8_test_fail",
        fail_stage="modeling",
    )
    after = json.loads(LATEST_SUCCESSFUL_PATH.read_text(encoding="utf-8"))
    assert failed.status.state == OperationalStateName.FAILED_USING_LAST_SUCCESS
    assert before["run_id"] == after["run_id"]

    with RefreshLock():
        locked = refresh_operational(season="2026-27", mock_launch=True)
    assert locked.status.state == OperationalStateName.FAILED_USING_LAST_SUCCESS
    assert not LOCK_PATH.exists()


def test_dashboard_smoke_builds_html_without_server() -> None:
    refresh_operational(season="2026-27", mock_launch=True, force=True, run_id="phase8_dashboard_smoke")

    path = run_dashboard(smoke=True)
    html = path.read_text(encoding="utf-8")

    assert path.exists()
    assert "FPL Forecast" in html
    assert "mocked target-season production model chain" in html
    assert "phase8_dashboard_smoke_team_current" in html


def test_target_fixture_change_moves_team_and_player_forecasts(tmp_path) -> None:
    base = run_operational_model_chain(season="2026-27", run_id="phase8_fixture_base", output_dir=tmp_path / "base")
    changed = run_operational_model_chain(
        season="2026-27",
        run_id="phase8_fixture_changed",
        output_dir=tmp_path / "changed",
        fixture_variant="opponent_swap",
    )

    teams = base.team_predictions.merge(
        changed.team_predictions,
        on="stable_fixture_uid",
        suffixes=("_base", "_changed"),
    )
    team_delta = (
        teams["expected_home_goals_base"].sub(teams["expected_home_goals_changed"]).abs()
        + teams["expected_away_goals_base"].sub(teams["expected_away_goals_changed"]).abs()
    )
    assert team_delta.gt(1e-9).any()

    players = base.player_gameweek_predictions.merge(
        changed.player_gameweek_predictions,
        on=["player_uid", "model_name"],
        suffixes=("_base", "_changed"),
    )
    player_delta = players["expected_points_base"].sub(players["expected_points_changed"]).abs()
    assert player_delta.gt(1e-9).any()


def test_price_change_can_move_optimized_squad_without_changing_performance_forecasts(tmp_path) -> None:
    base = run_operational_model_chain(season="2026-27", run_id="phase8_price_base", output_dir=tmp_path / "base")
    changed = run_operational_model_chain(
        season="2026-27",
        run_id="phase8_price_changed",
        output_dir=tmp_path / "changed",
        price_variant="premium_target",
    )

    performance = base.player_gameweek_predictions.merge(
        changed.player_gameweek_predictions,
        on=["player_uid", "model_name"],
        suffixes=("_base", "_changed"),
    )
    max_delta = performance["expected_points_base"].sub(performance["expected_points_changed"]).abs().max()
    assert max_delta == 0
    assert set(base.optimized_squad["player_uid"]) != set(changed.optimized_squad["player_uid"])


def test_full_chain_handles_new_transferred_position_change_and_promoted_fallback(tmp_path) -> None:
    result = run_operational_model_chain(season="2026-27", run_id="phase8_special_cases", output_dir=tmp_path)
    candidates = result.decision_candidates.loc[
        result.decision_candidates["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M3")
    ].copy()

    assert {
        "new_player_cold_start",
        "transferred_player",
        "position_change",
        "promoted_team_neutral_fallback",
    }.issubset(set(candidates["lineage_note"]))
    special = candidates.loc[candidates["lineage_note"].isin(["new_player_cold_start", "promoted_team_neutral_fallback"])]
    assert special["cold_start_no_history"].astype(bool).all()
    transferred = candidates.loc[candidates["lineage_note"].eq("transferred_player")].iloc[0]
    assert transferred["player_team_uid"] in set(result.team_predictions["home_team_uid"]) | set(
        result.team_predictions["away_team_uid"]
    )

    promoted_fixture = result.team_predictions.loc[
        result.team_predictions["home_team_uid"].eq("team_promoted_neutral_fallback")
        | result.team_predictions["away_team_uid"].eq("team_promoted_neutral_fallback")
    ]
    assert not promoted_fixture.empty
    assert (
        promoted_fixture["home_unseen_or_promoted_flag"].astype(bool)
        | promoted_fixture["away_unseen_or_promoted_flag"].astype(bool)
    ).any()


def test_real_cached_2026_27_status_is_waiting() -> None:
    result = refresh_operational(season="2026-27", offline=True, status_only=True)

    assert result.status.state == OperationalStateName.WAITING_FOR_SEASON_LAUNCH
    assert result.status.inferred_official_season == "2025-26"
    assert STATUS_PATH.exists()


def _write_current_snapshots(
    raw_dir,
    *,
    stored_season: str,
    payload_season: str,
    stamp: int = 1,
    budget: int = 1000,
) -> None:
    bootstrap = _bootstrap_payload(payload_season, budget=budget)
    fixtures = _fixtures_payload(payload_season)
    retrieved = datetime(2026, 7, stamp, tzinfo=UTC)
    write_raw_snapshot(
        raw_dir,
        season=stored_season,
        endpoint_name=BOOTSTRAP_STATIC,
        content=json.dumps(bootstrap).encode(),
        source_url="mock://bootstrap",
        http_status=200,
        target_season=stored_season,
        source="fpl_api",
        source_version=payload_season,
        retrieved_at=retrieved,
    )
    write_raw_snapshot(
        raw_dir,
        season=stored_season,
        endpoint_name=FIXTURES,
        content=json.dumps(fixtures).encode(),
        source_url="mock://fixtures",
        http_status=200,
        target_season=stored_season,
        source="fpl_api",
        source_version=payload_season,
        retrieved_at=retrieved,
    )


def _bootstrap_payload(season: str, *, budget: int) -> dict:
    start = int(season[:4])
    return {
        "events": [
            {"id": event, "deadline_time": f"{start if event < 20 else start + 1}-{8 if event < 20 else 1:02d}-01T11:00:00Z", "finished": event < 2}
            for event in range(1, 39)
        ],
        "game_settings": {
            "squad_squadsize": 15,
            "squad_squadplay": 11,
            "squad_team_limit": 3,
            "squad_total_spend": budget,
            "sys_vice_captain_enabled": True,
        },
        "phases": [],
        "teams": [{"id": team, "code": team, "name": f"Team {team}", "short_name": f"T{team}", "strength": 3} for team in range(1, 21)],
        "total_players": 1,
        "elements": [],
        "element_stats": [],
        "element_types": [
            {"id": 1, "singular_name_short": "GKP", "squad_select": 2, "squad_min_play": 1, "squad_max_play": 1},
            {"id": 2, "singular_name_short": "DEF", "squad_select": 5, "squad_min_play": 3, "squad_max_play": 5},
            {"id": 3, "singular_name_short": "MID", "squad_select": 5, "squad_min_play": 2, "squad_max_play": 5},
            {"id": 4, "singular_name_short": "FWD", "squad_select": 3, "squad_min_play": 1, "squad_max_play": 3},
        ],
    }


def _fixtures_payload(season: str) -> list[dict]:
    start = int(season[:4])
    return [
        {
            "id": fixture,
            "code": fixture,
            "event": (fixture - 1) // 10 + 1,
            "team_h": fixture % 20 + 1,
            "team_a": (fixture + 1) % 20 + 1,
            "kickoff_time": f"{start if fixture < 200 else start + 1}-{8 if fixture < 200 else 1:02d}-15T15:00:00Z",
            "finished": fixture < 5,
            "finished_provisional": fixture < 5,
        }
        for fixture in range(1, 381)
    ]
