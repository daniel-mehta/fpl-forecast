from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.team_model.data import load_current_fixture_frame


def validate_current_minutes_inputs(
    *,
    season: str,
    gameweek: int | None,
    as_of: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> pd.DataFrame:
    """Validate official current-season fixtures before any player-level forecast is attempted."""
    fixtures = load_current_fixture_frame(season=season, as_of=as_of, normalized_dir=normalized_dir)
    if gameweek is not None:
        fixtures = fixtures.loc[pd.to_numeric(fixtures["gameweek"], errors="coerce").eq(gameweek)].copy()
    if fixtures.empty:
        raise ValueError("No forecastable future fixtures match the requested filters.")
    players_path = Path(normalized_dir) / season / "current_players.parquet"
    if not players_path.exists():
        raise FileNotFoundError(f"Missing official normalized current players for {season}.")
    players = pd.read_parquet(players_path)
    if players.empty:
        raise ValueError(f"No current players are available for {season}.")
    if not players["season"].eq(season).all():
        raise ValueError(f"Current player season mismatch for requested {season}.")
    if "player_code" not in players.columns or players["player_code"].isna().all():
        raise ValueError("Current players do not contain usable FPL player codes.")
    return fixtures
