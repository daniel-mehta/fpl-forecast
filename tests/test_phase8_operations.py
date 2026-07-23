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
from fpl_forecast.operations.live_results import (
    audit_event_live_scoring,
    finalized_fixture_ids,
    normalize_event_live,
    validate_event_live_for_forecast,
)
from fpl_forecast.operations.locking import RefreshLock
from fpl_forecast.operations.model_chain import run_operational_model_chain
from fpl_forecast.operations.orchestrator import refresh_operational
from fpl_forecast.operations.state import OperationalStateName
from fpl_forecast.operations.transition import run_mock_gw1_to_gw2_transition


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
    payload = _event_live_payload(
        [
            _element(
                7,
                total=8,
                explain=[
                    _fixture_explain(11, [("minutes", 90, 2), ("goals_scored", 1, 5)]),
                    _fixture_explain(12, [("minutes", 30, 1)]),
                ],
            )
        ]
    )

    frame = normalize_event_live(
        season="2026-27",
        gameweek=1,
        payload=payload,
        retrieved_at="2026-08-16T20:00:00Z",
        raw_snapshot_path="mock/event_live.json",
        bootstrap_payload=_bootstrap_payload("2026-27", budget=1000),
        fixtures_payload=_fixtures_payload("2026-27")[:20],
    )

    assert len(frame) == 2
    assert set(frame["fixture_id"]) == {11, 12}
    assert int(frame.loc[frame["fixture_id"].eq(11), "total_points"].iloc[0]) == 7
    assert int(frame.loc[frame["fixture_id"].eq(12), "total_points"].iloc[0]) == 1
    assert frame.loc[frame["fixture_id"].eq(11), "official_event_total_points"].iloc[0] == 8


def test_event_live_uses_awarded_points_not_raw_values() -> None:
    payload = _event_live_payload(
        [_element(7, total=7, explain=[_fixture_explain(11, [("minutes", 83, 2), ("goals_scored", 1, 5)])])]
    )

    frame = _normalize_test_live(payload)

    row = frame.iloc[0]
    assert row["goals_scored"] == 1
    assert row["points_goals_scored"] == 5
    assert row["reconstructed_points"] == 7
    assert audit_event_live_scoring(frame)["status_counts"].set_index("audit_status").loc["exact_match", "rows"] == 1


def test_event_live_reconstructs_dnp_sub_clean_sheet_gc_saves_bonus_and_cards() -> None:
    payload = _event_live_payload(
        [
            _element(7, total=0, minutes=0, explain=[_fixture_explain(11, [("minutes", 0, 0)])]),
            _element(8, total=1, minutes=30, explain=[_fixture_explain(11, [("minutes", 30, 1)])]),
            _element(9, total=2, minutes=59, explain=[_fixture_explain(11, [("minutes", 59, 1), ("clean_sheets", 1, 0), ("yellow_cards", 1, -1), ("bonus", 2, 2)])]),
            _element(10, total=1, minutes=90, explain=[_fixture_explain(11, [("minutes", 90, 2), ("goals_conceded", 2, -1)])]),
            _element(11, total=3, minutes=90, explain=[_fixture_explain(11, [("minutes", 90, 2), ("saves", 5, 1)])]),
        ]
    )

    frame = _normalize_test_live(payload)
    audit = audit_event_live_scoring(frame)["player_event_reconciliation"]

    assert set(audit["audit_status"]) == {"exact_match"}
    assert int(frame.loc[frame["player_id"].eq(8), "starts"].iloc[0]) == 0
    assert int(frame.loc[frame["player_id"].eq(9), "points_clean_sheets"].iloc[0]) == 0
    assert int(frame.loc[frame["player_id"].eq(10), "points_goals_conceded"].iloc[0]) == -1
    assert int(frame.loc[frame["player_id"].eq(11), "points_saves"].iloc[0]) == 1


def test_event_live_double_gameweek_does_not_repeat_event_totals() -> None:
    payload = _event_live_payload(
        [
            _element(
                7,
                total=9,
                explain=[
                    _fixture_explain(11, [("minutes", 90, 2), ("goals_scored", 1, 5)]),
                    _fixture_explain(12, [("minutes", 30, 1), ("bonus", 1, 1)]),
                ],
            )
        ]
    )

    frame = _normalize_test_live(payload)
    audit = audit_event_live_scoring(frame)["player_event_reconciliation"].iloc[0]

    assert list(frame.sort_values("fixture_id")["total_points"]) == [7, 2]
    assert not frame["total_points"].eq(9).any()
    assert audit["reconstructed_points"] == 9
    assert audit["difference"] == 0


def test_event_live_duplicate_components_and_unfinished_fixtures_fail_safety_gate() -> None:
    payload = _event_live_payload(
        [_element(7, total=4, explain=[_fixture_explain(11, [("minutes", 90, 2), ("minutes", 90, 2)])])]
    )
    fixtures = _fixtures_payload("2026-27")[:20]
    fixtures[10]["finished"] = False
    fixtures[10]["finished_provisional"] = False
    frame = normalize_event_live(
        season="2026-27",
        gameweek=1,
        payload=payload,
        retrieved_at="2026-08-16T20:00:00Z",
        raw_snapshot_path="mock/event_live.json",
        bootstrap_payload=_bootstrap_payload("2026-27", budget=1000),
        fixtures_payload=fixtures,
    )

    issues = validate_event_live_for_forecast(frame, information_cutoff="2026-08-17T10:00:00Z")

    assert frame["unresolved_source_limitation"].astype(bool).any()
    assert "incomplete fixture is being treated as final" in issues
    assert "unresolved source limitation rows cannot enter model training" in issues


def test_event_live_source_after_cutoff_and_duplicate_keys_fail_safety_gate() -> None:
    frame = _normalize_test_live(
        _event_live_payload([_element(7, total=2, explain=[_fixture_explain(11, [("minutes", 90, 2)])])])
    )
    duplicated = pd.concat([frame, frame], ignore_index=True)

    issues = validate_event_live_for_forecast(duplicated, information_cutoff="2026-08-16T19:00:00Z")

    assert "duplicate player-fixture keys" in issues
    assert "source availability is after the forecast cutoff" in issues


def test_event_live_preserves_assistant_managers_but_excludes_them_from_player_audit() -> None:
    bootstrap = _bootstrap_payload("2026-27", budget=1000)
    bootstrap["elements"].append({"id": 99, "code": 9900, "web_name": "Assistant", "team": 1, "element_type": 5})
    payload = _event_live_payload(
        [
            _element(7, total=2, explain=[_fixture_explain(11, [("minutes", 90, 2)])]),
            _element(99, total=6, explain=[_fixture_explain(11, [("minutes", 90, 2), ("bonus", 4, 4)])]),
        ]
    )

    frame = normalize_event_live(
        season="2026-27",
        gameweek=1,
        payload=payload,
        retrieved_at="2026-08-16T20:00:00Z",
        raw_snapshot_path="mock/event_live.json",
        bootstrap_payload=bootstrap,
        fixtures_payload=_fixtures_payload("2026-27")[:20],
    )
    reconciliation = audit_event_live_scoring(frame)["player_event_reconciliation"]

    assert frame.loc[frame["player_id"].eq(99), "entity_type"].iloc[0] == "assistant_manager"
    assert set(reconciliation["player_id"]) == {7}


def test_event_live_reingestion_is_idempotent_and_revised_snapshot_replaces_by_key() -> None:
    first = _normalize_test_live(
        _event_live_payload([_element(7, total=2, explain=[_fixture_explain(11, [("minutes", 90, 2)])])])
    )
    second = _normalize_test_live(
        _event_live_payload([_element(7, total=3, explain=[_fixture_explain(11, [("minutes", 90, 2), ("bonus", 1, 1)])])]),
        retrieved_at="2026-08-16T21:00:00Z",
    )

    assert first.equals(_normalize_test_live(_event_live_payload([_element(7, total=2, explain=[_fixture_explain(11, [("minutes", 90, 2)])])])))
    revised = (
        pd.concat([first, second], ignore_index=True)
        .sort_values("source_available_time")
        .drop_duplicates(["season", "fixture_id", "player_uid"], keep="last")
    )
    assert len(revised) == 1
    assert int(revised.iloc[0]["total_points"]) == 3


def test_mock_gw1_to_gw2_transition_publishes_and_preserves_latest_on_failure() -> None:
    result = run_mock_gw1_to_gw2_transition(season="2026-27", run_id="phase8_transition_test")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.no_op
    assert result.failure_preserved_latest
    assert summary["completed_rows_entered_gw2_history"] > 0
    assert summary["gw2_projection_rows"] > 0
    assert summary["gw2_optimized_squad_rows"] == 15
    assert summary["gw2_targets_absent_from_inputs"]


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
        "total_players": 6,
        "elements": [
            {"id": 7, "code": 700, "web_name": "Player 7", "team": 1, "element_type": 3},
            {"id": 8, "code": 800, "web_name": "Player 8", "team": 1, "element_type": 3},
            {"id": 9, "code": 900, "web_name": "Player 9", "team": 1, "element_type": 3},
            {"id": 10, "code": 1000, "web_name": "Player 10", "team": 1, "element_type": 2},
            {"id": 11, "code": 1100, "web_name": "Player 11", "team": 1, "element_type": 1},
        ],
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


def _event_live_payload(elements: list[dict]) -> dict:
    return {"elements": elements}


def _element(
    player_id: int,
    *,
    total: int,
    explain: list[dict],
    minutes: int | None = None,
) -> dict:
    stats = {
        "minutes": minutes if minutes is not None else sum(_stat_value(block, "minutes") for block in explain),
        "goals_scored": sum(_stat_value(block, "goals_scored") for block in explain),
        "assists": sum(_stat_value(block, "assists") for block in explain),
        "clean_sheets": sum(_stat_value(block, "clean_sheets") for block in explain),
        "goals_conceded": sum(_stat_value(block, "goals_conceded") for block in explain),
        "own_goals": sum(_stat_value(block, "own_goals") for block in explain),
        "penalties_saved": sum(_stat_value(block, "penalties_saved") for block in explain),
        "penalties_missed": sum(_stat_value(block, "penalties_missed") for block in explain),
        "yellow_cards": sum(_stat_value(block, "yellow_cards") for block in explain),
        "red_cards": sum(_stat_value(block, "red_cards") for block in explain),
        "saves": sum(_stat_value(block, "saves") for block in explain),
        "bonus": sum(_stat_value(block, "bonus") for block in explain),
        "bps": 0,
        "defensive_contribution": sum(_stat_value(block, "defensive_contribution") for block in explain),
        "starts": int((minutes if minutes is not None else sum(_stat_value(block, "minutes") for block in explain)) >= 60),
        "total_points": total,
        "played": total > 0,
        "in_dreamteam": False,
    }
    return {"id": player_id, "stats": stats, "explain": explain, "modified": False}


def _fixture_explain(fixture_id: int, stats: list[tuple[str, int, int]]) -> dict:
    return {
        "fixture": fixture_id,
        "stats": [
            {"identifier": identifier, "value": value, "points": points, "points_modification": 0}
            for identifier, value, points in stats
        ],
    }


def _stat_value(block: dict, identifier: str) -> int:
    return sum(int(stat.get("value", 0)) for stat in block.get("stats", []) if stat.get("identifier") == identifier)


def _normalize_test_live(payload: dict, *, retrieved_at: str = "2026-08-16T20:00:00Z") -> pd.DataFrame:
    bootstrap = _bootstrap_payload("2026-27", budget=1000)
    fixtures = _fixtures_payload("2026-27")
    for fixture in fixtures:
        if fixture["id"] in {11, 12}:
            fixture["event"] = 1
            fixture["team_h"] = 1
            fixture["team_a"] = 2
            fixture["finished"] = True
            fixture["finished_provisional"] = True
    return normalize_event_live(
        season="2026-27",
        gameweek=1,
        payload=payload,
        retrieved_at=retrieved_at,
        raw_snapshot_path="mock/event_live.json",
        bootstrap_payload=bootstrap,
        fixtures_payload=fixtures,
    )
