from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_forecast.panel.players import build_player_identities
from fpl_forecast.panel.teams import build_team_identities, read_team_aliases


def test_team_aliases_promoted_clubs_and_id_reuse(tmp_path):
    normalized_dir = _write_history(tmp_path, _history_rows())
    alias_path = tmp_path / "manual" / "team_aliases.csv"
    alias_path.parent.mkdir()
    alias_path.write_text(
        "source_name,canonical_name,note\n"
        "Man Utd,Manchester United,short name\n"
        "Spurs,Tottenham Hotspur,short name\n",
        encoding="utf-8",
    )

    result = build_team_identities(
        seasons=["2022-23", "2023-24"],
        normalized_dir=normalized_dir,
        alias_path=alias_path,
    )

    assert "team_manchester_united" in set(result.dim_team["team_uid"])
    assert "team_burnley" in set(result.dim_team["team_uid"])
    reused_id = result.team_season_map.loc[result.team_season_map["source_team_id"] == 1]
    assert reused_id["team_uid"].nunique() == 2


def test_team_alias_ambiguity_fails(tmp_path):
    alias_path = tmp_path / "team_aliases.csv"
    alias_path.write_text(
        "source_name,canonical_name,note\n"
        "United,Manchester United,a\n"
        "United,Newcastle United,b\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Ambiguous duplicate team aliases"):
        read_team_aliases(alias_path)


def test_player_identity_stable_code_transfer_position_change_and_duplicate_names(tmp_path):
    normalized_dir = _write_history(tmp_path, _history_rows(include_duplicate_name=True))
    result = build_player_identities(
        seasons=["2022-23", "2023-24"],
        normalized_dir=normalized_dir,
        override_path=tmp_path / "manual" / "player_identity_overrides.csv",
        review_dir=tmp_path / "review",
    )

    mapped = result.player_season_map
    alpha = mapped.loc[mapped["source_player_name"] == "Alex Alpha"]
    assert alpha["player_uid"].nunique() == 1
    assert set(alpha["source_team"]) == {"Man Utd", "Spurs"}
    assert set(alpha["fpl_position"]) == {"MID", "FWD"}
    duplicate_names = mapped.loc[mapped["source_player_name"] == "Sam Same"]
    assert duplicate_names["player_uid"].nunique() == 2


def test_player_missing_code_and_code_collision_require_review(tmp_path):
    rows = _history_rows()
    rows.extend(
        [
            _row("2022-23", 31, None, "No Code", "Man Utd", "MID", 201),
            _row("2022-23", 32, 9999, "Collision One", "Man Utd", "MID", 201),
            _row("2022-23", 33, 9999, "Collision Two", "Man Utd", "MID", 201),
        ]
    )
    normalized_dir = _write_history(tmp_path, rows)

    result = build_player_identities(
        seasons=["2022-23"],
        normalized_dir=normalized_dir,
        override_path=tmp_path / "manual" / "player_identity_overrides.csv",
        review_dir=tmp_path / "review",
    )

    review = result.review_candidates
    assert set(review["match_method"]) >= {"missing_code", "ambiguous_code"}
    assert (tmp_path / "review" / "player_identity_review_candidates.csv").exists()


def test_player_manual_override_resolves_missing_code(tmp_path):
    rows = [_row("2022-23", 31, None, "No Code", "Man Utd", "MID", 201)]
    normalized_dir = _write_history(tmp_path, rows)
    override_path = tmp_path / "manual" / "player_identity_overrides.csv"
    override_path.parent.mkdir()
    override_path.write_text(
        "season,source_player_id,source_player_name,intended_player_uid,reason,evidence,reviewed_by,review_timestamp\n"
        "2022-23,31,No Code,player_manual_no_code,fixture source,manual review,tester,2026-07-22T00:00:00Z\n",
        encoding="utf-8",
    )

    result = build_player_identities(
        seasons=["2022-23"],
        normalized_dir=normalized_dir,
        override_path=override_path,
        review_dir=tmp_path / "review",
    )

    row = result.player_season_map.loc[result.player_season_map["source_player_id"] == 31].iloc[0]
    assert row["player_uid"] == "player_manual_no_code"
    assert row["manual_review_status"] == "reviewed"


def _write_history(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    normalized_dir = tmp_path / "normalized"
    frame = pd.DataFrame(rows)
    for season, season_frame in frame.groupby("season"):
        season_dir = normalized_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        season_frame.to_parquet(season_dir / "historical_player_fixtures.parquet", index=False)
    return normalized_dir


def _history_rows(*, include_duplicate_name: bool = False) -> list[dict[str, object]]:
    rows = [
        _row("2022-23", 11, 1001, "Alex Alpha", "Man Utd", "MID", 201, was_home=True, opponent_team=2),
        _row("2022-23", 12, 1002, "Sam Same", "Spurs", "DEF", 201, was_home=False, opponent_team=1),
        _row("2023-24", 21, 1001, "Alex Alpha", "Spurs", "FWD", 301, was_home=True, opponent_team=1),
        _row("2023-24", 22, 1002, "Sam Same", "Burnley", "DEF", 301, was_home=False, opponent_team=2),
    ]
    if include_duplicate_name:
        rows.append(
            _row("2023-24", 23, 1003, "Sam Same", "Burnley", "MID", 301, was_home=False, opponent_team=2)
        )
    return rows


def _row(
    season: str,
    player_id: int,
    player_code: int | None,
    name: str,
    team: str,
    position: str,
    fixture_id: int,
    *,
    was_home: bool = True,
    opponent_team: int = 2,
) -> dict[str, object]:
    return {
        "season": season,
        "player_id": player_id,
        "player_code": player_code,
        "fixture_id": fixture_id,
        "gameweek": 1,
        "player_name": name,
        "team_name": team,
        "source_position": position,
        "element_type": {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4, "AM": 5}[position],
        "fpl_position": position,
        "kickoff_time": f"{season[:4]}-08-15T12:00:00Z",
        "minutes": 90,
        "starts": 1,
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
