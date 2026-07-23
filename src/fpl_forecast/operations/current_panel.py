from __future__ import annotations

import pandas as pd


def build_current_player_fixture_history(
    *,
    fixtures: pd.DataFrame,
    live_results: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    if live_results.empty:
        return pd.DataFrame()
    player_meta = players[["player_id", "player_code", "web_name", "team_id", "position", "price_tenths"]].rename(
        columns={"web_name": "player_name", "position": "fpl_position"}
    )
    fixture_meta = fixtures[["fixture_id", "gameweek", "kickoff_time", "finished", "finished_provisional"]]
    frame = live_results.merge(player_meta, on="player_id", how="left")
    frame = frame.merge(fixture_meta, on=["fixture_id", "gameweek"], how="left")
    frame = frame.loc[frame["fpl_position"].ne("AM")].copy()
    return frame.drop_duplicates(["season", "player_id", "fixture_id"], keep="last")
