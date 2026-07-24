from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.panel.common import phase2_dir


SEASON_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def resolve_historical_training_seasons(
    *,
    target_season: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> list[str]:
    base = Path(normalized_dir)
    if not SEASON_PATTERN.match(target_season):
        raise ValueError(f"Invalid target season label: {target_season}")
    if not base.exists():
        raise FileNotFoundError(
            "Missing normalized data directory for operational training: "
            f"{base}. Required Phase 2 artifacts include {phase2_dir(base) / 'dim_team.parquet'}."
        )
    candidates = sorted(
        path.name
        for path in base.iterdir()
        if path.is_dir() and SEASON_PATTERN.match(path.name) and _season_start(path.name) < _season_start(target_season)
    )
    if not candidates:
        raise FileNotFoundError(f"No completed historical seasons found before {target_season}.")
    _validate_required_phase2_seasons(candidates, normalized_dir=base)
    return candidates


def _validate_required_phase2_seasons(seasons: list[str], *, normalized_dir: Path) -> None:
    for season in seasons:
        history_path = normalized_dir / season / "historical_player_fixtures.parquet"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing normalized historical player-fixture table: {history_path}")
    phase2 = phase2_dir(normalized_dir)
    required = {
        "dim_fixture.parquet": "season",
        "fact_player_fixture.parquet": "season",
        "features_player_fixture.parquet": "season",
        "team_season_map.parquet": "season",
        "player_season_map.parquet": "season",
    }
    for filename, season_column in required.items():
        path = phase2 / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 2 artifact required for operational training: {path}")
        frame = pd.read_parquet(path, columns=[season_column])
        present = set(frame[season_column].dropna().astype(str))
        missing = sorted(set(seasons).difference(present))
        if missing:
            raise ValueError(f"{path} is missing required training season(s): {', '.join(missing)}")


def _season_start(season: str) -> int:
    return int(season.split("-", maxsplit=1)[0])
