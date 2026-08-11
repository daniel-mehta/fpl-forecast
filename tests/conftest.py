from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_forecast.xpoints.rules import goal_points_for_position

from fpl_forecast.panel.build import build_panel
from fpl_forecast.panel.players import build_player_identities
from fpl_forecast.panel.teams import build_team_identities


FIXTURES_DIR = Path(__file__).parent / "fixtures"
PHASE8_SEASONS = ["2022-23", "2023-24", "2024-25"]
PHASE8_POSITIONS = (
    ["GKP", "GKP"]
    + ["DEF"] * 5
    + ["MID"] * 5
    + ["FWD"] * 3
)
PHASE8_SPECIAL_CODES = {
    (1, "MID", 0): 223094,
    (2, "FWD", 0): 164511,
    (3, "DEF", 0): 233420,
}


def fixture_bytes(relative_path: str) -> bytes:
    return (FIXTURES_DIR / relative_path).read_bytes()


def fixture_text(relative_path: str) -> str:
    return (FIXTURES_DIR / relative_path).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def phase8_normalized_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    normalized_dir = tmp_path_factory.mktemp("phase8_normalized") / "normalized"
    _write_phase8_historical_player_fixtures(normalized_dir)
    manual_dir = normalized_dir / "manual"
    build_team_identities(
        seasons=PHASE8_SEASONS,
        normalized_dir=normalized_dir,
        alias_path=manual_dir / "team_aliases.csv",
    )
    build_player_identities(
        seasons=PHASE8_SEASONS,
        normalized_dir=normalized_dir,
        override_path=manual_dir / "player_identity_overrides.csv",
        review_dir=normalized_dir / "review",
    )
    build_panel(seasons=PHASE8_SEASONS, normalized_dir=normalized_dir)
    return normalized_dir


def _write_phase8_historical_player_fixtures(normalized_dir: Path) -> None:
    players = _phase8_players()
    for season in PHASE8_SEASONS:
        frame = pd.DataFrame(_phase8_history_rows(season, players))
        season_dir = normalized_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(season_dir / "historical_player_fixtures.parquet", index=False)


def _phase8_players() -> list[dict[str, object]]:
    players: list[dict[str, object]] = []
    source_player_id = 1
    for team_id in range(1, 21):
        for position_slot, position in enumerate(PHASE8_POSITIONS):
            position_index = PHASE8_POSITIONS[:position_slot].count(position)
            player_code = PHASE8_SPECIAL_CODES.get(
                (team_id, position, position_index),
                300000 + source_player_id,
            )
            players.append(
                {
                    "player_id": source_player_id,
                    "player_code": player_code,
                    "player_name": f"Phase8 T{team_id:02d} {position} {position_index + 1}",
                    "team_id": team_id,
                    "team_name": f"Phase8 Team {team_id:02d}",
                    "source_position": position,
                    "element_type": {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[position],
                    "fpl_position": position,
                    "position_slot": position_slot,
                    "price_tenths": {"GKP": 45, "DEF": 48, "MID": 55, "FWD": 60}[position] + (team_id % 5),
                    "is_price_target": player_code in set(PHASE8_SPECIAL_CODES.values()),
                }
            )
            source_player_id += 1
    return players


def _phase8_history_rows(season: str, players: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    season_start = int(season[:4])
    players_by_team = {
        team_id: [player for player in players if int(player["team_id"]) == team_id]
        for team_id in range(1, 21)
    }
    for gameweek in range(1, 7):
        for match_index in range(10):
            home_team = match_index * 2 + 1
            away_team = match_index * 2 + 2
            if gameweek % 2 == 0:
                home_team, away_team = away_team, home_team
            fixture_id = (gameweek - 1) * 10 + match_index + 1
            kickoff = f"{season_start}-08-{gameweek + 1:02d}T15:00:00Z"
            home_goals = (home_team + gameweek) % 4
            away_goals = (away_team + match_index) % 3
            for team_id, opponent_id, was_home in (
                (home_team, away_team, True),
                (away_team, home_team, False),
            ):
                for player in players_by_team[team_id]:
                    rows.append(
                        _phase8_player_fixture_row(
                            season=season,
                            player=player,
                            fixture_id=fixture_id,
                            gameweek=gameweek,
                            kickoff=kickoff,
                            was_home=was_home,
                            opponent_id=opponent_id,
                            home_goals=home_goals,
                            away_goals=away_goals,
                        )
                    )
    return rows


def _phase8_player_fixture_row(
    *,
    season: str,
    player: dict[str, object],
    fixture_id: int,
    gameweek: int,
    kickoff: str,
    was_home: bool,
    opponent_id: int,
    home_goals: int,
    away_goals: int,
) -> dict[str, object]:
    position = str(player["fpl_position"])
    slot = int(player["position_slot"])
    minutes = 90 if slot < 11 else 25
    starts = int(minutes >= 60)
    goals = int(bool(player["is_price_target"]) and position in {"MID", "FWD"}) + int(position == "FWD" and slot == 12)
    assists = int(bool(player["is_price_target"]) and position in {"DEF", "MID"})
    clean_sheets = int(position in {"GKP", "DEF"} and ((was_home and away_goals == 0) or (not was_home and home_goals == 0)))
    goals_conceded = away_goals if was_home else home_goals
    total_points = (
        int(minutes > 0)
        + int(minutes >= 60)
        + goals * goal_points_for_position(season=season, position=position)
        + assists * 3
        + clean_sheets * (4 if position in {"GKP", "DEF"} else 1)
        - int(goals_conceded >= 2 and position in {"GKP", "DEF"})
    )
    return {
        "season": season,
        "player_id": player["player_id"],
        "player_code": player["player_code"],
        "fixture_id": fixture_id,
        "gameweek": gameweek,
        "player_name": player["player_name"],
        "team_name": player["team_name"],
        "source_position": player["source_position"],
        "element_type": player["element_type"],
        "fpl_position": player["fpl_position"],
        "kickoff_time": kickoff,
        "source_available_time": pd.Timestamp(kickoff) + pd.Timedelta(hours=3),
        "source_available_method": "kickoff_plus_3h_conservative_match_completion",
        "minutes": minutes,
        "starts": starts,
        "goals_scored": goals,
        "assists": assists,
        "clean_sheets": clean_sheets,
        "goals_conceded": goals_conceded,
        "saves": 3 if position == "GKP" else 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": int(slot == 10),
        "red_cards": 0,
        "own_goals": 0,
        "bonus": int(bool(player["is_price_target"])),
        "bps": 20 + goals * 10 + assists * 5,
        "defensive_contribution": 0,
        "total_points": total_points,
        "price_tenths": player["price_tenths"],
        "was_home": was_home,
        "opponent_team": opponent_id,
        "team_h_score": home_goals,
        "team_a_score": away_goals,
        "source": "pytest_phase8_fixture",
        "source_version": "phase8_hermetic_v1",
        "retrieved_at": "2026-07-23T00:00:00Z",
        "raw_snapshot_path": f"mock://phase8/{season}/fixture_{fixture_id}.json",
    }
