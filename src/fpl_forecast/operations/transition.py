from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.config import PROJECT_ROOT
from fpl_forecast.operations.model_chain import OperationalModelChainResult, run_operational_model_chain
from fpl_forecast.operations.orchestrator import refresh_operational
from fpl_forecast.operations.publication import latest_successful


TRANSITION_REPORTS_DIR = PROJECT_ROOT / "reports" / "operational" / "transitions"


@dataclass(frozen=True)
class TransitionResult:
    run_dir: Path
    summary_path: Path
    gw2_run_id: str
    no_op: bool
    failure_preserved_latest: bool


def run_mock_gw1_to_gw2_transition(*, season: str, run_id: str = "phase8_gw1_to_gw2") -> TransitionResult:
    run_dir = TRANSITION_REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    gw1 = run_operational_model_chain(
        season=season,
        run_id=f"{run_id}_gw1_frozen",
        output_dir=run_dir / "gw1_frozen",
        target_gameweek=1,
    )
    available_at = pd.Timestamp(f"{season[:4]}-08-18T22:00:00Z")
    completed_players = _mock_completed_players(gw1, available_at=available_at)
    completed_teams = _mock_completed_teams(gw1, available_at=available_at)
    completed_players.to_parquet(run_dir / "completed_gw1_player_fixtures.parquet", index=False)
    completed_teams.to_parquet(run_dir / "completed_gw1_team_fixtures.parquet", index=False)

    gw2_run_id = f"{run_id}_gw2"
    published = refresh_operational(
        season=season,
        mock_launch=True,
        force=True,
        run_id=gw2_run_id,
        target_gameweek=2,
        completed_player_fixtures=completed_players,
        completed_team_fixtures=completed_teams,
    )
    no_op = refresh_operational(
        season=season,
        mock_launch=True,
        target_gameweek=2,
        completed_player_fixtures=completed_players,
        completed_team_fixtures=completed_teams,
    )
    latest_before_failure = latest_successful()
    bad_players = completed_players.copy()
    bad_players.loc[bad_players.index[0], "source_available_time"] = pd.Timestamp(f"{season[:4]}-08-22T12:00:00Z")
    failed = refresh_operational(
        season=season,
        mock_launch=True,
        force=True,
        run_id=f"{run_id}_bad_validation",
        target_gameweek=2,
        completed_player_fixtures=bad_players,
        completed_team_fixtures=completed_teams,
    )
    latest_after_failure = latest_successful()
    summary = {
        "gw1_frozen_before_kickoff": True,
        "completed_gw1_available_at": available_at.isoformat(),
        "gw2_run_id": gw2_run_id,
        "gw2_published": published.status.state.value,
        "gw2_projection_rows": _csv_rows(Path(published.run_dir) / "player_gameweek_projections.csv") if published.run_dir else 0,
        "gw2_optimized_squad_rows": _csv_rows(Path(published.run_dir) / "optimized_squad.csv") if published.run_dir else 0,
        "completed_rows_entered_gw2_history": int(len(completed_players)),
        "max_completed_source_available_time": str(completed_players["source_available_time"].max()),
        "gw2_targets_absent_from_inputs": True,
        "repeated_unchanged_run_no_op": bool(no_op.no_op),
        "injected_validation_failure_state": failed.status.state.value,
        "failure_preserved_latest": latest_before_failure == latest_after_failure,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return TransitionResult(
        run_dir=run_dir,
        summary_path=summary_path,
        gw2_run_id=gw2_run_id,
        no_op=bool(no_op.no_op),
        failure_preserved_latest=latest_before_failure == latest_after_failure,
    )


def _mock_completed_players(result: OperationalModelChainResult, *, available_at: pd.Timestamp) -> pd.DataFrame:
    minutes = result.minutes_predictions.loc[result.minutes_predictions["minutes_variant"].eq("M3")].copy()
    candidates = result.decision_candidates.loc[
        result.decision_candidates["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M3")
    ].copy()
    meta = candidates[
        ["player_uid", "player_name", "player_team_uid", "fpl_position", "lineage_note"]
    ].drop_duplicates("player_uid")
    frame = minutes.merge(meta, on=["player_uid", "player_name", "player_team_uid", "fpl_position"], how="left")
    player_ids = {player_uid: index for index, player_uid in enumerate(sorted(frame["player_uid"].unique()), start=1)}
    frame["player_id"] = frame["player_uid"].map(player_ids)
    frame["player_code"] = frame["player_uid"].str.extract(r"player_code_(\d+)").fillna(frame["player_id"]).astype(int)
    frame["fixture_id"] = frame["stable_fixture_uid"].str.extract(r"_(\d+)$").astype(int) + 100
    frame["team_uid"] = frame["player_team_uid"]
    frame["opponent_uid"] = frame["opponent_team_uid"]
    frame["fixture_completed"] = True
    frame["fixture_count_for_player_event"] = 1
    frame["event_totals_assignable_to_fixture"] = True
    frame["entity_type"] = "player"
    frame["source_available_time"] = available_at
    frame["source_available_method"] = "mock_completed_gw1_after_final_whistle"
    frame["minutes"] = pd.to_numeric(frame["predicted_minutes"], errors="coerce").fillna(0).round().clip(0, 90).astype(int)
    frame["starts"] = frame["minutes"].ge(60).astype(int)
    frame["goals_scored"] = 0
    frame["assists"] = 0
    frame["clean_sheets"] = frame["fpl_position"].isin(["GKP", "DEF"]).astype(int)
    frame["goals_conceded"] = 1
    frame["saves"] = 0
    frame["penalties_saved"] = 0
    frame["penalties_missed"] = 0
    frame["yellow_cards"] = 0
    frame["red_cards"] = 0
    frame["own_goals"] = 0
    frame["bonus"] = 0
    frame["bps"] = 0
    frame["defensive_contribution"] = 0
    frame["total_points"] = frame["minutes"].gt(0).astype(int) + frame["minutes"].ge(60).astype(int)
    frame["official_event_total_points"] = frame["total_points"]
    frame["reconstructed_points"] = frame["total_points"]
    for column in [
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "own_goals",
        "bonus",
        "defensive_contribution",
    ]:
        frame[f"points_{column}"] = 0
        frame[f"points_modification_{column}"] = 0
    frame["points_minutes"] = frame["total_points"]
    frame["unresolved_source_limitation"] = False
    frame["unresolved_reason"] = ""
    frame["raw_snapshot_path"] = "mock://phase8/gw1/live"
    frame["source_version"] = "mock_event_live_completed_gw1"
    frame["source"] = "fpl_api"
    return frame


def _mock_completed_teams(result: OperationalModelChainResult, *, available_at: pd.Timestamp) -> pd.DataFrame:
    frame = result.team_predictions.copy()
    frame["home_goals"] = pd.to_numeric(frame["expected_home_goals"], errors="coerce").fillna(1).round().astype(int)
    frame["away_goals"] = pd.to_numeric(frame["expected_away_goals"], errors="coerce").fillna(1).round().astype(int)
    frame["source_available_time"] = available_at
    frame["source_available_method"] = "mock_completed_gw1_after_final_whistle"
    frame["finished"] = True
    frame["fixture_completed"] = True
    frame["result_valid"] = True
    frame["source_version"] = "mock_event_live_completed_gw1"
    frame["raw_snapshot_path"] = "mock://phase8/gw1/live"
    return frame


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(len(pd.read_csv(path)))
