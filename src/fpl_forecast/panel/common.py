from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR


def parse_seasons(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        seasons = [season.strip() for season in value.split(",") if season.strip()]
    else:
        seasons = [season.strip() for season in value if season.strip()]
    if not seasons:
        raise ValueError("At least one season is required.")
    return seasons


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def uid_from_slug(prefix: str, value: object) -> str:
    slug = normalize_name(value).replace(" ", "_")
    if not slug:
        raise ValueError(f"Cannot build {prefix} uid from empty value.")
    return f"{prefix}_{slug}"


def season_dir(normalized_dir: Path | str, season: str) -> Path:
    return Path(normalized_dir) / season


def phase2_dir(normalized_dir: Path | str) -> Path:
    output_dir = Path(normalized_dir) / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_historical_player_fixtures(
    *,
    normalized_dir: Path | str = NORMALIZED_DIR,
    seasons: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = season_dir(normalized_dir, season) / "historical_player_fixtures.parquet"
        if not path.exists():
            legacy_path = season_dir(normalized_dir, season) / "historical_player_gameweeks.parquet"
            if legacy_path.exists():
                raise FileNotFoundError(
                    f"{legacy_path} exists but Phase 2 requires player-fixture grain at {path}."
                )
            raise FileNotFoundError(f"Missing normalized historical player-fixture table: {path}")
        frame = pd.read_parquet(path)
        if "season" not in frame.columns:
            frame["season"] = season
        frames.append(frame)
    if not frames:
        raise ValueError("No historical player-fixture tables loaded.")
    return pd.concat(frames, ignore_index=True)


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
