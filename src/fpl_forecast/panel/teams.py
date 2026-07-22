from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.config import MANUAL_DIR, NORMALIZED_DIR
from fpl_forecast.panel.common import (
    load_historical_player_fixtures,
    normalize_name,
    phase2_dir,
    uid_from_slug,
    write_parquet,
)


TEAM_ALIAS_TEMPLATE = MANUAL_DIR / "team_aliases.csv"


@dataclass(frozen=True)
class TeamIdentityResult:
    dim_team_path: Path
    team_season_map_path: Path
    dim_team: pd.DataFrame
    team_season_map: pd.DataFrame


def ensure_team_alias_template(path: Path = TEAM_ALIAS_TEMPLATE) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "source_name,canonical_name,note\n"
        "Nott'm Forest,Nottingham Forest,Vaastav short apostrophe form\n"
        "Spurs,Tottenham Hotspur,Vaastav common short name\n"
        "Man Utd,Manchester United,Vaastav common short name\n"
        "Man City,Manchester City,Vaastav common short name\n",
        encoding="utf-8",
    )


def build_team_identities(
    *,
    seasons: list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
    alias_path: Path = TEAM_ALIAS_TEMPLATE,
) -> TeamIdentityResult:
    ensure_team_alias_template(alias_path)
    history = load_historical_player_fixtures(normalized_dir=normalized_dir, seasons=seasons)
    alias_map = read_team_aliases(alias_path)
    records = _source_team_records(history)
    records["canonical_name"] = records["source_team_name"].map(
        lambda value: alias_map.get(normalize_name(value), value)
    )
    records["team_uid"] = records["canonical_name"].map(lambda value: uid_from_slug("team", value))
    records["match_method"] = records["source_team_name"].map(
        lambda value: "manual_alias" if normalize_name(value) in alias_map else "exact_normalized_name"
    )
    records["match_confidence"] = records["match_method"].map(
        {"manual_alias": 0.95, "exact_normalized_name": 1.0}
    )
    records["evidence"] = records.apply(
        lambda row: f"source_name={row.source_team_name}; normalized={normalize_name(row.source_team_name)}",
        axis=1,
    )
    team_season_map = records[
        [
            "season",
            "source_team_id",
            "source_team_name",
            "team_uid",
            "match_method",
            "match_confidence",
            "evidence",
        ]
    ].sort_values(["season", "team_uid", "source_team_id"], na_position="last")

    dim_team = (
        team_season_map[["team_uid", "source_team_name"]]
        .merge(records[["team_uid", "canonical_name"]].drop_duplicates(), on="team_uid", how="left")
        .drop_duplicates("team_uid")
        .rename(columns={"source_team_name": "first_source_name"})
    )
    dim_team["normalized_short_name"] = dim_team["canonical_name"].map(normalize_name)
    dim_team = dim_team[["team_uid", "canonical_name", "normalized_short_name"]].sort_values(
        "team_uid"
    )

    duplicate_aliases = (
        pd.DataFrame(
            [{"normalized": key, "canonical": value} for key, value in alias_map.items()]
        )
        .groupby("normalized")["canonical"]
        .nunique()
    )
    ambiguous = duplicate_aliases[duplicate_aliases > 1]
    if not ambiguous.empty:
        raise ValueError(f"Ambiguous team aliases found: {', '.join(ambiguous.index)}")

    output_dir = phase2_dir(normalized_dir)
    dim_path = write_parquet(dim_team, output_dir / "dim_team.parquet")
    map_path = write_parquet(team_season_map, output_dir / "team_season_map.parquet")
    return TeamIdentityResult(dim_path, map_path, dim_team, team_season_map)


def read_team_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    aliases = pd.read_csv(path)
    required = {"source_name", "canonical_name"}
    missing = required.difference(aliases.columns)
    if missing:
        raise ValueError(f"Team alias file is missing columns: {', '.join(sorted(missing))}")
    normalized = aliases["source_name"].map(normalize_name)
    if normalized.duplicated().any():
        dupes = aliases.loc[normalized.duplicated(keep=False), "source_name"].tolist()
        raise ValueError(f"Ambiguous duplicate team aliases: {', '.join(map(str, dupes))}")
    return dict(zip(normalized, aliases["canonical_name"], strict=False))


def _source_team_records(history: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "fixture_id", "team_name", "opponent_team", "was_home"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Historical player-fixture data missing columns: {', '.join(sorted(missing))}")

    rows: list[dict[str, object]] = []
    for (season, fixture_id), fixture_rows in history.groupby(["season", "fixture_id"], dropna=False):
        sides = fixture_rows[["team_name", "was_home", "opponent_team"]].drop_duplicates()
        for _, side in sides.iterrows():
            other_side = sides.loc[sides["was_home"] != side["was_home"]]
            source_team_id = pd.NA
            if len(other_side) == 1:
                source_team_id = other_side.iloc[0]["opponent_team"]
            rows.append(
                {
                    "season": season,
                    "source_team_id": source_team_id,
                    "source_team_name": side["team_name"],
                }
            )
    records = pd.DataFrame(rows).dropna(subset=["source_team_name"]).drop_duplicates()
    records["source_team_id"] = pd.to_numeric(records["source_team_id"], errors="coerce").astype(
        "Int64"
    )
    return records
