from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from fpl_forecast.operations.model_chain import run_operational_model_chain


def test_official_mode_uses_current_inputs_and_contains_no_mock_markers(tmp_path, phase8_normalized_dir) -> None:
    normalized_dir = _copy_phase8_normalized(tmp_path, phase8_normalized_dir)
    _write_official_current_tables(normalized_dir)

    result = run_operational_model_chain(
        season="2026-27",
        run_id="phase9b_official_test",
        output_dir=tmp_path / "official",
        normalized_dir=normalized_dir,
        source_mode="official_current_season",
    )

    assert result.lineage["source_mode"] == "official_current_season"
    assert result.lineage["target_fixture_count"] == 10
    assert result.lineage["target_deadline"] == "2026-08-21T17:30:00+00:00"
    assert not result.team_predictions["stable_fixture_uid"].str.contains("mock").any()
    assert not result.decision_candidates["player_name"].str.contains("New Player").any()
    assert "Official Player" in result.decision_candidates["player_name"].iloc[0]
    assert len(result.optimized_squad) == 15
    assert set(result.optimized_squad["fpl_position"].value_counts().to_dict().items()) == {
        ("GKP", 2),
        ("DEF", 5),
        ("MID", 5),
        ("FWD", 3),
    }
    assert "official_bootstrap_static.now_cost" == result.lineage["price_source"]


def test_mock_mode_remains_explicitly_mocked(tmp_path, phase8_normalized_dir) -> None:
    result = run_operational_model_chain(
        season="2026-27",
        run_id="phase9b_mock_test",
        output_dir=tmp_path / "mock",
        normalized_dir=phase8_normalized_dir,
    )

    assert result.lineage["source_mode"] == "mock"
    assert result.team_predictions["stable_fixture_uid"].str.contains("mock").any()
    assert result.lineage["price_source"] == "mock_adapter"


def test_official_mode_excludes_assistant_managers(tmp_path, phase8_normalized_dir) -> None:
    normalized_dir = _copy_phase8_normalized(tmp_path, phase8_normalized_dir)
    _write_official_current_tables(normalized_dir, include_assistant_manager=True)

    result = run_operational_model_chain(
        season="2026-27",
        run_id="phase9b_assistant_manager_test",
        output_dir=tmp_path / "official",
        normalized_dir=normalized_dir,
        source_mode="official_current_season",
    )

    assert "player_code_99999999" not in set(result.decision_candidates["player_uid"])


def test_official_mode_fails_closed_on_invalid_prices(tmp_path, phase8_normalized_dir) -> None:
    normalized_dir = _copy_phase8_normalized(tmp_path, phase8_normalized_dir)
    _write_official_current_tables(normalized_dir, invalid_price=True)

    with pytest.raises(ValueError, match="invalid price"):
        run_operational_model_chain(
            season="2026-27",
            run_id="phase9b_invalid_price_test",
            output_dir=tmp_path / "official",
            normalized_dir=normalized_dir,
            source_mode="official_current_season",
        )


def test_official_mode_fails_closed_on_ambiguous_player_identity(tmp_path, phase8_normalized_dir) -> None:
    normalized_dir = _copy_phase8_normalized(tmp_path, phase8_normalized_dir)
    _write_official_current_tables(normalized_dir, duplicate_player_code=True)

    with pytest.raises(ValueError, match="Ambiguous current player identities"):
        run_operational_model_chain(
            season="2026-27",
            run_id="phase9b_ambiguous_player_test",
            output_dir=tmp_path / "official",
            normalized_dir=normalized_dir,
            source_mode="official_current_season",
        )


def test_official_mode_fails_closed_on_ambiguous_team_identity(tmp_path, phase8_normalized_dir) -> None:
    normalized_dir = _copy_phase8_normalized(tmp_path, phase8_normalized_dir)
    _write_official_current_tables(normalized_dir, duplicate_team_name=True)

    with pytest.raises(ValueError, match="Ambiguous current team identities"):
        run_operational_model_chain(
            season="2026-27",
            run_id="phase9b_ambiguous_team_test",
            output_dir=tmp_path / "official",
            normalized_dir=normalized_dir,
            source_mode="official_current_season",
        )


def _copy_phase8_normalized(tmp_path: Path, source: Path) -> Path:
    target = tmp_path / "normalized"
    shutil.copytree(source, target)
    return target


def _write_official_current_tables(
    normalized_dir: Path,
    *,
    include_assistant_manager: bool = False,
    invalid_price: bool = False,
    duplicate_player_code: bool = False,
    duplicate_team_name: bool = False,
) -> None:
    season_dir = normalized_dir / "2026-27"
    season_dir.mkdir(parents=True, exist_ok=True)
    teams = _current_teams(duplicate_team_name=duplicate_team_name)
    players = _current_players(
        include_assistant_manager=include_assistant_manager,
        invalid_price=invalid_price,
        duplicate_player_code=duplicate_player_code,
    )
    fixtures = _current_fixtures()
    events = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "gameweek": gameweek,
                "name": f"Gameweek {gameweek}",
                "deadline_time": "2026-08-21T17:30:00Z" if gameweek == 1 else "2026-08-28T17:30:00Z",
                "finished": False,
                "data_checked": False,
                "is_current": False,
                "is_next": gameweek == 1,
                "released": True,
                "retrieved_at": "2026-07-24T00:00:00Z",
                "source_version": "official_test",
                "raw_snapshot_path": "mock-official/bootstrap.json",
            }
            for gameweek in range(1, 39)
        ]
    )
    teams.to_parquet(season_dir / "current_teams.parquet", index=False)
    players.to_parquet(season_dir / "current_players.parquet", index=False)
    fixtures.to_parquet(season_dir / "current_fixtures.parquet", index=False)
    events.to_parquet(season_dir / "current_events.parquet", index=False)


def _current_teams(*, duplicate_team_name: bool) -> pd.DataFrame:
    rows = []
    for team_id in range(1, 21):
        name = f"Phase8 Team {team_id:02d}" if team_id < 20 else "Official Promoted"
        if duplicate_team_name and team_id == 2:
            name = "Phase8 Team 01"
        rows.append(
            {
                "season": "2026-27",
                "team_id": team_id,
                "team_code": team_id,
                "team_name": name,
                "short_name": f"T{team_id:02d}",
                "strength": pd.NA,
                "source": "fpl_api",
                "source_version": "official_test",
                "retrieved_at": "2026-07-24T00:00:00Z",
                "raw_snapshot_path": "mock-official/bootstrap.json",
            }
        )
    return pd.DataFrame(rows)


def _current_players(
    *,
    include_assistant_manager: bool,
    invalid_price: bool,
    duplicate_player_code: bool,
) -> pd.DataFrame:
    rows = []
    player_id = 1
    positions = [("GKP", 1, 2), ("DEF", 2, 5), ("MID", 3, 5), ("FWD", 4, 3)]
    for team_id in range(1, 21):
        for position, position_id, count in positions:
            for slot in range(count):
                code = 300000 + player_id
                if duplicate_player_code and player_id == 2:
                    code = 300001
                rows.append(
                    {
                        "season": "2026-27",
                        "player_id": player_id,
                        "player_code": code,
                        "first_name": "Official",
                        "second_name": f"Player {player_id}",
                        "web_name": f"Official Player {player_id}",
                        "team_id": team_id,
                        "position_id": position_id,
                        "position": position,
                        "entity_type": "player",
                        "price_tenths": 0 if invalid_price and player_id == 1 else 45 + position_id,
                        "price": 4.5 + position_id / 10,
                        "status": "a",
                        "news": "",
                        "chance_of_playing_next_round": pd.NA,
                        "chance_of_playing_this_round": pd.NA,
                        "selected_by_percent": 0.0,
                        "form": 0.0,
                        "minutes": 0,
                        "total_points": 0,
                        "source": "fpl_api",
                        "source_version": "official_test",
                        "retrieved_at": "2026-07-24T00:00:00Z",
                        "raw_snapshot_path": "mock-official/bootstrap.json",
                    }
                )
                player_id += 1
    if include_assistant_manager:
        rows.append(
            {
                **rows[0],
                "player_id": 9999,
                "player_code": 99999999,
                "web_name": "Official Manager",
                "position_id": 5,
                "position": "AM",
                "entity_type": "assistant_manager",
            }
        )
    return pd.DataFrame(rows)


def _current_fixtures() -> pd.DataFrame:
    rows = []
    for index in range(10):
        rows.append(
            {
                "season": "2026-27",
                "fixture_id": index + 1,
                "fixture_code": 1000 + index,
                "gameweek": 1,
                "home_team_id": index * 2 + 1,
                "away_team_id": index * 2 + 2,
                "kickoff_time": f"2026-08-{21 + index // 5:02d}T15:00:00Z",
                "finished": False,
                "started": False,
                "team_h_score": pd.NA,
                "team_a_score": pd.NA,
                "finished_provisional": False,
                "provisional_start_time": False,
                "minutes": 0,
                "home_team_difficulty": 3,
                "away_team_difficulty": 3,
                "pulse_id": 0,
                "source": "fpl_api",
                "source_version": "official_test",
                "retrieved_at": "2026-07-24T00:00:00Z",
                "raw_snapshot_path": "mock-official/fixtures.json",
            }
        )
    return pd.DataFrame(rows)
