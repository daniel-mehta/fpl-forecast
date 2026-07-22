from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.panel.common import phase2_dir


@dataclass(frozen=True)
class PanelInspection:
    lines: list[str]


def inspect_panel(
    *,
    seasons: list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> PanelInspection:
    output_dir = phase2_dir(normalized_dir)
    paths = {
        "dim_team": output_dir / "dim_team.parquet",
        "team_season_map": output_dir / "team_season_map.parquet",
        "dim_player": output_dir / "dim_player.parquet",
        "player_season_map": output_dir / "player_season_map.parquet",
        "dim_fixture": output_dir / "dim_fixture.parquet",
        "fact_player_fixture": output_dir / "fact_player_fixture.parquet",
        "features_player_fixture": output_dir / "features_player_fixture.parquet",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 2 tables: {', '.join(missing)}")
    fixture = pd.read_parquet(paths["dim_fixture"])
    fact = pd.read_parquet(paths["fact_player_fixture"])
    features = pd.read_parquet(paths["features_player_fixture"])
    player_map = pd.read_parquet(paths["player_season_map"])
    team_map = pd.read_parquet(paths["team_season_map"])

    lines: list[str] = []
    for season in seasons:
        season_fact = fact.loc[fact["season"] == season]
        season_fixture = fixture.loc[fixture["season"] == season]
        season_features = features.loc[features["season"] == season]
        lines.append(
            f"{season}: fact_rows={len(season_fact)} unique_players={season_fact['player_uid'].nunique()} "
            f"fixtures={season_fixture['fixture_key'].nunique()} teams={team_map.loc[team_map['season'] == season, 'team_uid'].nunique()}"
        )
        if not season_fact.empty:
            dupes = int(season_fact.duplicated(["season", "player_uid", "fixture_key"]).sum())
            zero_minutes = int((pd.to_numeric(season_fact["minutes"], errors="coerce") == 0).sum())
            entity_counts = season_fact["entity_type"].value_counts().to_dict()
            dgw_players = int(
                season_fact.loc[
                    season_fact.duplicated(["season", "gameweek", "player_uid"], keep=False),
                    ["gameweek", "player_uid"],
                ].drop_duplicates().shape[0]
            )
            gw1 = season_features.loc[season_features["gameweek"] == 1]
            gw1_minutes = (
                pd.to_numeric(gw1.get("season_to_date_minutes"), errors="coerce").fillna(0).max()
                if not gw1.empty
                else 0
            )
            lines.append(
                f"{season}: duplicate_fact_keys={dupes} zero_minute_rows={zero_minutes} "
                f"double_gameweek_player_count={dgw_players} gw1_max_season_to_date_minutes={gw1_minutes} "
                f"player_rows={entity_counts.get('player', 0)} "
                f"assistant_manager_rows={entity_counts.get('assistant_manager', 0)}"
            )
    method_counts = (
        player_map.groupby(["season", "match_method"]).size().reset_index(name="count").to_string(index=False)
    )
    lines.append("player_identity_method_counts:")
    lines.append(method_counts)
    missingness = (
        features.groupby("season")
        .agg(
            prior_minutes_missing=("prior_season_minutes", lambda series: int(series.isna().sum())),
            prev_fixture_minutes_missing=("prev_fixture_minutes", lambda series: int(series.isna().sum())),
        )
        .reset_index()
        .to_string(index=False)
    )
    lines.append("feature_missingness:")
    lines.append(missingness)
    return PanelInspection(lines=lines)
