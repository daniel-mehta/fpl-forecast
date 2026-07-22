from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fpl_forecast.cli import app
from fpl_forecast.features.build import build_features_from_frames
from fpl_forecast.features.leakage import audit_leakage_frames
from fpl_forecast.panel.build import build_identities, build_panel
from fpl_forecast.panel.cutoffs import build_gameweek_cutoffs
from fpl_forecast.panel.facts import standard_player_fact
from fpl_forecast.panel.fixtures import build_fixture_dimension_from_frames


def test_fixture_identity_double_blank_and_team_consistency(tmp_path):
    normalized_dir = _write_history(tmp_path, _panel_rows())
    build_identities(seasons=["2022-23"], normalized_dir=normalized_dir)
    team_map = pd.read_parquet(normalized_dir / "phase2" / "team_season_map.parquet")
    history = pd.read_parquet(normalized_dir / "2022-23" / "historical_player_fixtures.parquet")

    fixtures = build_fixture_dimension_from_frames(history, team_map)

    assert fixtures["fixture_key"].is_unique
    assert fixtures["source_fixture_id"].nunique() == 3
    assert set(fixtures["gameweek"]) == {1, 2}
    assert 3 not in set(fixtures["gameweek"])


def test_fact_features_gw1_prior_season_and_shift_correctness(tmp_path):
    normalized_dir = _write_history(tmp_path, _panel_rows(include_prior=True))
    build_identities(seasons=["2021-22", "2022-23"], normalized_dir=normalized_dir)
    result = build_panel(seasons=["2021-22", "2022-23"], normalized_dir=normalized_dir)
    fact = result.facts.fact
    features = result.features.features

    assert fact.duplicated(["season", "player_uid", "fixture_key"]).sum() == 0
    gw1 = features.loc[(features["season"] == "2022-23") & (features["gameweek"] == 1)]
    assert pd.to_numeric(gw1["season_to_date_minutes"], errors="coerce").fillna(0).max() == 0
    assert gw1["prior_season_minutes"].notna().any()
    later = features.loc[(features["season"] == "2022-23") & (features["gameweek"] == 2)]
    assert later["prev_fixture_minutes"].notna().any()
    later_with_lineage = later.dropna(subset=["player_max_source_available_time"])
    assert later_with_lineage["player_max_source_available_time"].lt(
        later_with_lineage["information_cutoff"]
    ).all()


def test_leakage_auditor_adversarial_failures():
    fact = pd.DataFrame(
        [
            {
                "season": "2022-23",
                "player_uid": "player_code_1",
                "fixture_key": "2022-23:1",
            }
        ]
    )
    features = pd.DataFrame(
        [
            {
                "season": "2022-23",
                "player_uid": "player_code_1",
                "fixture_key": "2022-23:1",
                "gameweek": 1,
                "kickoff_time": "2022-08-01T12:00:00Z",
                "information_cutoff": "2022-08-01T12:00:00Z",
                "player_team_uid": "team_a",
                "opponent_team_uid": "team_b",
                "was_home": True,
                "target_total_points": 5,
                "prev3_minutes_sum": 90,
                "player_max_source_kickoff": "2022-08-01T12:00:00Z",
                "player_max_source_available_time": "2022-08-01T15:00:00Z",
                "team_max_source_kickoff": "2022-08-02T12:00:00Z",
                "team_max_source_available_time": "2022-08-02T15:00:00Z",
                "opponent_max_source_available_time": "2022-08-01T15:00:00Z",
                "prior_source_season": "2023-24",
                "XP": 1.2,
                "season_total_points": 200,
                "feature_registry_version": "registry.json",
            }
        ]
    )

    result = audit_leakage_frames(features, fact)

    messages = " ".join(issue.message for issue in result.errors)
    assert "xP" in messages
    assert "information_cutoff" in messages
    assert "GW1" in messages
    assert "Prior-season" in messages
    assert "season cumulative" in messages or "cumulative" in messages


def test_leakage_auditor_rejects_source_unavailable_at_cutoff_even_if_kickoff_is_before():
    fact = pd.DataFrame(
        [{"season": "2022-23", "player_uid": "player_code_1", "fixture_key": "2022-23:2"}]
    )
    features = pd.DataFrame(
        [
            {
                "season": "2022-23",
                "player_uid": "player_code_1",
                "fixture_key": "2022-23:2",
                "gameweek": 2,
                "kickoff_time": "2022-08-01T14:00:00Z",
                "information_cutoff": "2022-08-01T13:00:00Z",
                "player_team_uid": "team_a",
                "opponent_team_uid": "team_b",
                "was_home": True,
                "target_total_points": 5,
                "prev3_minutes_sum": 90,
                "home_flag": 1,
                "player_max_source_kickoff": "2022-08-01T11:00:00Z",
                "player_max_source_available_time": "2022-08-01T14:00:00Z",
                "team_max_source_kickoff": "2022-08-01T11:00:00Z",
                "team_max_source_available_time": "2022-08-01T14:00:00Z",
                "opponent_max_source_kickoff": "2022-08-01T11:00:00Z",
                "opponent_max_source_available_time": "2022-08-01T14:00:00Z",
                "feature_registry_version": "registry.json",
            }
        ]
    )

    result = audit_leakage_frames(features, fact)

    assert any("not available before information_cutoff" in issue.message for issue in result.errors)


def test_standard_player_fact_excludes_assistant_managers(tmp_path):
    normalized_dir = _write_history(tmp_path, _panel_rows(include_assistant_manager=True))
    build_identities(seasons=["2022-23"], normalized_dir=normalized_dir)
    result = build_panel(seasons=["2022-23"], normalized_dir=normalized_dir)

    fact = result.facts.fact
    players_only = standard_player_fact(fact)

    assert "assistant_manager" in set(fact["entity_type"])
    assert set(players_only["entity_type"]) == {"player"}


def test_feature_registry_completeness_and_cli_failure(tmp_path):
    normalized_dir = _write_history(tmp_path, _panel_rows())
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["audit-leakage", "--seasons", "2022-23", "--normalized-dir", str(normalized_dir)],
    )

    assert result.exit_code == 1
    assert "Missing feature table" in result.output


def test_missing_kickoff_fails_fixture_build(tmp_path):
    rows = _panel_rows()
    rows[0]["kickoff_time"] = None
    normalized_dir = _write_history(tmp_path, rows)
    build_identities(seasons=["2022-23"], normalized_dir=normalized_dir)
    team_map = pd.read_parquet(normalized_dir / "phase2" / "team_season_map.parquet")
    history = pd.read_parquet(normalized_dir / "2022-23" / "historical_player_fixtures.parquet")

    with pytest.raises(ValueError, match="kickoff_time"):
        build_fixture_dimension_from_frames(history, team_map)


def test_exact_and_inferred_cutoff_semantics_from_feature_builder():
    fact = pd.DataFrame(_fact_rows_for_features())
    fixtures = pd.DataFrame(
        [
            {
                "season": "2022-23",
                "source_fixture_id": 1,
                "fixture_key": "2022-23:1",
                "gameweek": 1,
                "home_team_uid": "team_a",
                "away_team_uid": "team_b",
                "kickoff_time": "2022-08-01T12:00:00Z",
                "home_goals": 1,
                "away_goals": 0,
            },
            {
                "season": "2022-23",
                "source_fixture_id": 2,
                "fixture_key": "2022-23:2",
                "gameweek": 2,
                "home_team_uid": "team_a",
                "away_team_uid": "team_b",
                "kickoff_time": "2022-08-08T12:00:00Z",
                "home_goals": 2,
                "away_goals": 1,
            },
        ]
    )

    features = build_features_from_frames(fact, fixtures)

    assert set(features["information_cutoff"].astype(str)) == {
        "2022-08-01 12:00:00+00:00",
        "2022-08-08 12:00:00+00:00",
    }
    second = features.loc[features["fixture_key"] == "2022-23:2"].iloc[0]
    assert second["prev_fixture_minutes"] == 90
    assert second["player_max_source_available_time"] < second["information_cutoff"]


def test_cutoff_builder_uses_exact_deadline_when_available_and_infers_otherwise():
    fixtures = pd.DataFrame(
        [
            {"season": "2022-23", "gameweek": 1, "kickoff_time": "2022-08-05T20:00:00Z"},
            {"season": "2022-23", "gameweek": 1, "kickoff_time": "2022-08-06T12:30:00Z"},
            {"season": "2022-23", "gameweek": 2, "kickoff_time": "2022-08-13T12:30:00Z"},
        ]
    )
    deadlines = pd.DataFrame(
        [
            {
                "season": "2022-23",
                "gameweek": 1,
                "deadline_time": "2022-08-05T18:30:00Z",
                "cutoff_source": "official_fixture_history",
            }
        ]
    )

    cutoffs = build_gameweek_cutoffs(fixtures, exact_deadlines=deadlines)

    gw1 = cutoffs.loc[cutoffs["gameweek"] == 1].iloc[0]
    gw2 = cutoffs.loc[cutoffs["gameweek"] == 2].iloc[0]
    assert gw1["cutoff_is_exact"]
    assert gw1["cutoff_method"] == "exact_recorded_fpl_deadline"
    assert not gw2["cutoff_is_exact"]
    assert gw2["information_cutoff"].isoformat() == "2022-08-13T12:30:00+00:00"


def test_cutoff_builder_rejects_missing_kickoff():
    fixtures = pd.DataFrame([{"season": "2022-23", "gameweek": 1, "kickoff_time": None}])

    with pytest.raises(ValueError, match="kickoff_time"):
        build_gameweek_cutoffs(fixtures)


def _write_history(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    normalized_dir = tmp_path / "normalized"
    frame = pd.DataFrame(rows)
    for season, season_frame in frame.groupby("season"):
        season_dir = normalized_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        season_frame.to_parquet(season_dir / "historical_player_fixtures.parquet", index=False)
    return normalized_dir


def _panel_rows(
    *,
    include_prior: bool = False,
    include_assistant_manager: bool = False,
) -> list[dict[str, object]]:
    rows = []
    if include_prior:
        rows.extend(
            [
                _row("2021-22", 1, 1001, "Return Player", "Team A", "MID", 1, 1, True, 2, 90),
                _row("2021-22", 2, 1002, "Other Player", "Team B", "DEF", 1, 1, False, 1, 90),
            ]
        )
    rows.extend(
        [
            _row("2022-23", 11, 1001, "Return Player", "Team A", "MID", 1, 1, True, 2, 90),
            _row("2022-23", 12, 1002, "Other Player", "Team B", "DEF", 1, 1, False, 1, 90),
            _row("2022-23", 11, 1001, "Return Player", "Team A", "MID", 3, 2, True, 2, 0),
            _row("2022-23", 12, 1002, "Other Player", "Team B", "DEF", 3, 2, False, 1, 90),
            _row("2022-23", 11, 1001, "Return Player", "Team A", "MID", 4, 2, False, 2, 90),
            _row("2022-23", 13, 1003, "Promoted Player", "Team C", "FWD", 4, 2, True, 1, 90),
        ]
    )
    if include_assistant_manager:
        rows.append(
            _row("2022-23", 99, 9999, "Assistant Boss", "Team A", "AM", 1, 1, True, 2, 0)
        )
    return rows


def _row(
    season: str,
    player_id: int,
    code: int,
    name: str,
    team: str,
    position: str,
    fixture_id: int,
    gameweek: int,
    was_home: bool,
    opponent_team: int,
    minutes: int,
) -> dict[str, object]:
    year = int(season[:4])
    kickoff_day = 1 + fixture_id
    return {
        "season": season,
        "player_id": player_id,
        "player_code": code,
        "fixture_id": fixture_id,
        "gameweek": gameweek,
        "player_name": name,
        "team_name": team,
        "source_position": position,
        "element_type": {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4, "AM": 5}[position],
        "fpl_position": position,
        "kickoff_time": f"{year}-08-{kickoff_day:02d}T12:00:00Z",
        "source_available_time": f"{year}-08-{kickoff_day:02d}T15:00:00Z",
        "source_available_method": "kickoff_plus_3h_conservative_match_completion",
        "minutes": minutes,
        "starts": int(minutes > 0),
        "goals_scored": 1 if minutes > 0 and position == "FWD" else 0,
        "assists": 0,
        "total_points": 2,
        "price_tenths": 55,
        "was_home": was_home,
        "opponent_team": opponent_team,
        "team_h_score": 1,
        "team_a_score": 0,
        "source": "vaastav",
        "source_version": "abc123",
        "retrieved_at": "2026-07-22T00:00:00Z",
        "raw_snapshot_path": "/tmp/raw.csv",
    }


def _fact_rows_for_features() -> list[dict[str, object]]:
    base = []
    for fixture_key, gameweek, kickoff, minutes in (
        ("2022-23:1", 1, "2022-08-01T12:00:00Z", 90),
        ("2022-23:2", 2, "2022-08-08T12:00:00Z", 30),
    ):
        base.append(
            {
                "season": "2022-23",
                "player_uid": "player_code_1",
                "fixture_key": fixture_key,
                "gameweek": gameweek,
                "kickoff_time": kickoff,
                "information_cutoff": kickoff,
                "player_team_uid": "team_a",
                "opponent_team_uid": "team_b",
                "was_home": True,
                "entity_type": "player",
                "source_available_time": (
                    pd.Timestamp(kickoff, tz="UTC") + pd.Timedelta(hours=3)
                ),
                "minutes": minutes,
                "starts": int(minutes >= 60),
                "total_points": 2,
                "goals_scored": 0,
                "assists": 0,
                "target_total_points": 2,
            }
        )
    return base
