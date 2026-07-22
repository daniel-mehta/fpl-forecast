from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.config import MANUAL_DIR, NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.panel.common import (
    load_historical_player_fixtures,
    phase2_dir,
    uid_from_slug,
    write_parquet,
)


PLAYER_OVERRIDE_TEMPLATE = MANUAL_DIR / "player_identity_overrides.csv"
REVIEW_DIR = PROJECT_ROOT / "data" / "review"


@dataclass(frozen=True)
class PlayerIdentityResult:
    dim_player_path: Path
    player_season_map_path: Path
    review_path: Path
    dim_player: pd.DataFrame
    player_season_map: pd.DataFrame
    review_candidates: pd.DataFrame


def ensure_player_override_template(path: Path = PLAYER_OVERRIDE_TEMPLATE) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "season,source_player_id,source_player_name,intended_player_uid,reason,evidence,"
        "reviewed_by,review_timestamp\n",
        encoding="utf-8",
    )


def build_player_identities(
    *,
    seasons: list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
    override_path: Path = PLAYER_OVERRIDE_TEMPLATE,
    review_dir: Path = REVIEW_DIR,
) -> PlayerIdentityResult:
    ensure_player_override_template(override_path)
    history = load_historical_player_fixtures(normalized_dir=normalized_dir, seasons=seasons)
    source_players = _source_players(history)
    overrides = read_player_overrides(override_path)
    mapped = source_players.merge(
        overrides,
        on=["season", "source_player_id"],
        how="left",
        suffixes=("", "_override"),
    )

    code_counts = (
        mapped.dropna(subset=["player_code"])
        .groupby(["season", "player_code"])[["source_player_id", "source_player_name"]]
        .nunique()
        .reset_index()
    )
    code_collisions = code_counts.loc[
        (code_counts["source_player_id"] > 1) | (code_counts["source_player_name"] > 1)
    ]
    collision_keys = set(zip(code_collisions["season"], code_collisions["player_code"], strict=False))

    player_uids: list[str | None] = []
    methods: list[str] = []
    confidence: list[float] = []
    statuses: list[str] = []
    evidence: list[str] = []
    for row in mapped.itertuples(index=False):
        if isinstance(row.intended_player_uid, str) and row.intended_player_uid:
            player_uids.append(row.intended_player_uid)
            methods.append("manual_override")
            confidence.append(1.0)
            statuses.append("reviewed")
            evidence.append(f"manual override: {row.reason}; {row.evidence}")
        elif pd.notna(row.player_code) and (row.season, row.player_code) not in collision_keys:
            player_uids.append(f"player_code_{int(row.player_code)}")
            methods.append("fpl_code")
            confidence.append(0.99)
            statuses.append("automatic")
            evidence.append(f"stable FPL code {int(row.player_code)}")
        elif pd.notna(row.player_code):
            player_uids.append(None)
            methods.append("ambiguous_code")
            confidence.append(0.0)
            statuses.append("ambiguous")
            evidence.append(f"FPL code {row.player_code} collides within season")
        else:
            player_uids.append(None)
            methods.append("missing_code")
            confidence.append(0.0)
            statuses.append("unresolved")
            evidence.append("missing FPL code; name matching requires manual review")

    mapped["player_uid"] = player_uids
    mapped["match_method"] = methods
    mapped["match_confidence"] = confidence
    mapped["manual_review_status"] = statuses
    mapped["evidence"] = evidence

    player_season_map = mapped[
        [
            "season",
            "source_player_id",
            "player_code",
            "source_player_name",
            "source_team",
            "fpl_position",
            "player_uid",
            "match_method",
            "match_confidence",
            "manual_review_status",
            "evidence",
        ]
    ].sort_values(["season", "source_player_id"])

    conflicts = (
        player_season_map.dropna(subset=["player_uid"])
        .groupby(["season", "player_uid"])["source_player_id"]
        .nunique()
    )
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        keys = [f"{season}:{uid}" for season, uid in conflicts.index[:10]]
        raise ValueError(f"player_uid maps to multiple active players in a season: {', '.join(keys)}")

    dim_source = player_season_map.dropna(subset=["player_uid"]).copy()
    dim_source["season_order"] = dim_source["season"]
    dim_player = (
        dim_source.sort_values(["player_uid", "season_order"])
        .groupby("player_uid", as_index=False)
        .agg(
            canonical_display_name=("source_player_name", "first"),
            stable_fpl_code=("player_code", "first"),
            first_observed_season=("season", "min"),
            last_observed_season=("season", "max"),
        )
    )

    review_candidates = player_season_map.loc[
        player_season_map["manual_review_status"].isin(["unresolved", "ambiguous"])
    ].copy()
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / "player_identity_review_candidates.csv"
    review_candidates.to_csv(review_path, index=False)

    output_dir = phase2_dir(normalized_dir)
    dim_path = write_parquet(dim_player, output_dir / "dim_player.parquet")
    map_path = write_parquet(player_season_map, output_dir / "player_season_map.parquet")
    return PlayerIdentityResult(
        dim_path,
        map_path,
        review_path,
        dim_player,
        player_season_map,
        review_candidates,
    )


def read_player_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "season",
                "source_player_id",
                "source_player_name",
                "intended_player_uid",
                "reason",
                "evidence",
                "reviewed_by",
                "review_timestamp",
            ]
        )
    overrides = pd.read_csv(path)
    required = {"season", "source_player_id", "source_player_name", "intended_player_uid"}
    missing = required.difference(overrides.columns)
    if missing:
        raise ValueError(f"Player override file missing columns: {', '.join(sorted(missing))}")
    if not overrides.empty:
        duplicates = overrides.duplicated(["season", "source_player_id"]).sum()
        if duplicates:
            raise ValueError("Player override file contains duplicate season/source_player_id rows.")
    overrides["source_player_id"] = pd.to_numeric(overrides["source_player_id"], errors="coerce").astype(
        "Int64"
    )
    for column in ("reason", "evidence"):
        if column not in overrides.columns:
            overrides[column] = ""
    return overrides


def _source_players(history: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "player_id", "player_name", "team_name", "fpl_position"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Historical player-fixture data missing columns: {', '.join(sorted(missing))}")
    frame = history.copy()
    if "player_code" not in frame.columns:
        frame["player_code"] = pd.NA
    output = (
        frame.rename(
            columns={
                "player_id": "source_player_id",
                "player_name": "source_player_name",
                "team_name": "source_team",
            }
        )
        .sort_values(["season", "source_player_id", "fixture_id"])
        .groupby(["season", "source_player_id"], as_index=False)
        .agg(
            player_code=("player_code", "first"),
            source_player_name=("source_player_name", "first"),
            source_team=("source_team", "first"),
            fpl_position=("fpl_position", "first"),
        )
    )
    output["player_code"] = pd.to_numeric(output["player_code"], errors="coerce").astype("Int64")
    return output


def uid_for_manual_player(name: object, season: object, source_player_id: object) -> str:
    return uid_from_slug("player", f"{name}_{season}_{source_player_id}")
