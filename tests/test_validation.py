from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from typer.testing import CliRunner

from fpl_forecast.cli import app
from fpl_forecast.ingest.snapshots import write_raw_snapshot
from fpl_forecast.validation.data_quality import validate_all


def test_validation_flags_duplicate_keys_invalid_minutes_xp_warning_and_provenance(tmp_path):
    normalized_dir = tmp_path / "normalized"
    raw_vaastav_dir = tmp_path / "raw" / "vaastav"
    table_dir = normalized_dir / "2024-25"
    table_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            _history_row(player_id=11, fixture_id=101, minutes=131),
            _history_row(player_id=11, fixture_id=101, minutes=90),
        ]
    ).to_parquet(table_dir / "historical_player_fixtures.parquet", index=False)
    write_raw_snapshot(
        raw_vaastav_dir,
        season="2024-25",
        endpoint_name="merged_gw",
        content=b"element,fixture,round,minutes,total_points,xP\n11,101,1,90,10,4.4\n",
        source_url="https://raw.githubusercontent.com/example/merged_gw.csv",
        http_status=200,
        response_headers={},
        source="vaastav",
        source_version="abc123",
        content_type="csv",
        retrieved_at=datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC),
    )

    result = validate_all(normalized_dir=normalized_dir, raw_vaastav_dir=raw_vaastav_dir)

    error_messages = [issue.message for issue in result.errors]
    warning_messages = [issue.message for issue in result.warnings]
    assert any("duplicate" in message for message in error_messages)
    assert any("plausible single-fixture range" in message for message in error_messages)
    assert any("xP" in message for message in warning_messages)


def test_validation_rejects_synthetic_demo_contamination(tmp_path):
    normalized_dir = tmp_path / "normalized" / "2026-27"
    normalized_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "player_id": 9000001,
                "player_code": 1,
                "team_id": 1,
                "position_id": 3,
                "position": "MID",
                "price_tenths": 75,
                "minutes": 90,
                "source": "synthetic_demo",
                "source_version": "demo",
                "retrieved_at": "2026-07-22T12:00:00Z",
                "season": "2026-27",
                "raw_snapshot_path": "outputs/synthetic_demo/not_a_real_input.csv",
            }
        ]
    ).to_parquet(normalized_dir / "current_players.parquet", index=False)

    result = validate_all(
        normalized_dir=tmp_path / "normalized",
        raw_vaastav_dir=tmp_path / "raw" / "vaastav",
    )

    messages = [issue.message for issue in result.errors]
    assert any("Synthetic source marker" in message for message in messages)
    assert any("synthetic demo output" in message for message in messages)
    assert any("Synthetic demo player IDs" in message for message in messages)


def test_validation_rejects_case_insensitive_xp_leakage(tmp_path):
    normalized_dir = tmp_path / "normalized" / "2024-25"
    normalized_dir.mkdir(parents=True)
    row = _history_row(player_id=11, fixture_id=101, minutes=90)
    row["XP"] = 4.2
    pd.DataFrame([row]).to_parquet(
        normalized_dir / "historical_player_fixtures.parquet",
        index=False,
    )

    result = validate_all(
        normalized_dir=tmp_path / "normalized",
        raw_vaastav_dir=tmp_path / "raw" / "vaastav",
    )

    assert any("xP is present" in issue.message for issue in result.errors)


def test_validation_rejects_unparseable_timestamps(tmp_path):
    normalized_dir = tmp_path / "normalized" / "2024-25"
    normalized_dir.mkdir(parents=True)
    row = _history_row(player_id=11, fixture_id=101, minutes=90)
    row["retrieved_at"] = "not-a-time"
    pd.DataFrame([row]).to_parquet(
        normalized_dir / "historical_player_fixtures.parquet",
        index=False,
    )

    result = validate_all(
        normalized_dir=tmp_path / "normalized",
        raw_vaastav_dir=tmp_path / "raw" / "vaastav",
    )

    assert any("retrieved_at" in issue.message for issue in result.errors)


def test_validation_reports_missing_required_columns(tmp_path):
    normalized_dir = tmp_path / "normalized" / "2026-27"
    normalized_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "team_name": "Arsenal",
                "source": "fpl_api",
                "source_version": "sha",
                "retrieved_at": "2026-07-22T12:00:00Z",
                "season": "2026-27",
                "raw_snapshot_path": "/tmp/raw.json",
            }
        ]
    ).to_parquet(normalized_dir / "current_teams.parquet", index=False)

    result = validate_all(
        normalized_dir=tmp_path / "normalized",
        raw_vaastav_dir=tmp_path / "raw" / "vaastav",
    )

    assert any("Missing required columns" in issue.message for issue in result.errors)


def test_cli_smoke_validate_data_reports_error_for_empty_directory(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate-data",
            "--normalized-dir",
            str(tmp_path / "normalized"),
            "--raw-vaastav-dir",
            str(tmp_path / "raw" / "vaastav"),
        ],
    )

    assert result.exit_code == 1
    assert "No normalized Parquet tables found" in result.output


def test_phase1_validation_ignores_phase2_outputs(tmp_path):
    phase2_dir = tmp_path / "normalized" / "phase2"
    phase2_dir.mkdir(parents=True)
    pd.DataFrame([{"team_uid": "team_a", "canonical_name": "Team A"}]).to_parquet(
        phase2_dir / "dim_team.parquet",
        index=False,
    )
    season_dir = tmp_path / "normalized" / "2026-27"
    season_dir.mkdir()
    pd.DataFrame(
        [
            {
                "team_id": 1,
                "team_code": 1,
                "team_name": "Team A",
                "source": "fpl_api",
                "source_version": "sha",
                "retrieved_at": "2026-07-22T12:00:00Z",
                "season": "2026-27",
                "raw_snapshot_path": "/tmp/raw.json",
            }
        ]
    ).to_parquet(season_dir / "current_teams.parquet", index=False)

    result = validate_all(
        normalized_dir=tmp_path / "normalized",
        raw_vaastav_dir=tmp_path / "raw" / "vaastav",
    )

    assert result.ok


def _history_row(*, player_id: int, fixture_id: int, minutes: int) -> dict[str, object]:
    return {
        "player_id": player_id,
        "player_code": 1001,
        "fixture_id": fixture_id,
        "gameweek": 1,
        "player_name": "Real Player",
        "team_name": "Arsenal",
        "source_position": "MID",
        "element_type": 3,
        "fpl_position": "MID",
        "kickoff_time": "2025-08-15T12:00:00Z",
        "minutes": minutes,
        "total_points": 10,
        "price_tenths": 75,
        "expected_goals": 0.4,
        "source": "vaastav",
        "source_version": "abc123",
        "retrieved_at": "2026-07-22T12:00:00Z",
        "season": "2024-25",
        "raw_snapshot_path": "/tmp/merged_gw.csv",
    }
