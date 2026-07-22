from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.features.build import registered_feature_names
from fpl_forecast.panel.common import phase2_dir


FORBIDDEN_FEATURE_NAMES = {
    "total_points",
    "target_total_points",
    "minutes",
    "goals_scored",
    "assists",
    "bonus",
    "bps",
}
ALLOWED_NON_FEATURE_COLUMNS = {
    "season",
    "player_uid",
    "fixture_key",
    "gameweek",
    "kickoff_time",
    "source_available_time",
    "information_cutoff",
    "player_team_uid",
    "opponent_team_uid",
    "was_home",
    "target_total_points",
    "entity_type",
    "player_max_source_kickoff",
    "player_max_source_available_time",
    "team_max_source_kickoff",
    "team_max_source_available_time",
    "opponent_max_source_kickoff",
    "opponent_max_source_available_time",
    "prior_source_season",
    "feature_registry_version",
}


@dataclass(frozen=True)
class LeakageIssue:
    severity: str
    message: str


@dataclass(frozen=True)
class LeakageAuditResult:
    issues: list[LeakageIssue]

    @property
    def errors(self) -> list[LeakageIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


def audit_leakage(
    *,
    seasons: list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> LeakageAuditResult:
    output_dir = phase2_dir(normalized_dir)
    feature_path = output_dir / "features_player_fixture.parquet"
    fact_path = output_dir / "fact_player_fixture.parquet"
    if not feature_path.exists():
        return LeakageAuditResult([LeakageIssue("error", f"Missing feature table: {feature_path}")])
    if not fact_path.exists():
        return LeakageAuditResult([LeakageIssue("error", f"Missing fact table: {fact_path}")])
    features = pd.read_parquet(feature_path)
    fact = pd.read_parquet(fact_path)
    features = features.loc[features["season"].isin(seasons)].copy()
    fact = fact.loc[fact["season"].isin(seasons)].copy()
    return audit_leakage_frames(features, fact)


def audit_leakage_frames(features: pd.DataFrame, fact: pd.DataFrame) -> LeakageAuditResult:
    issues: list[LeakageIssue] = []
    required_keys = ["season", "player_uid", "fixture_key"]
    missing_keys = sorted(set(required_keys).difference(features.columns))
    if missing_keys:
        return LeakageAuditResult(
            [LeakageIssue("error", f"Feature table missing key columns: {', '.join(missing_keys)}")]
        )
    duplicate_features = int(features.duplicated(required_keys).sum())
    if duplicate_features:
        issues.append(LeakageIssue("error", f"Feature table has {duplicate_features} duplicate keys."))
    duplicate_fact = int(fact.duplicated(required_keys).sum()) if set(required_keys).issubset(fact.columns) else 0
    if duplicate_fact:
        issues.append(LeakageIssue("error", f"Fact table has {duplicate_fact} duplicate keys."))

    for column in features.columns:
        if column.lower() == "xp":
            issues.append(LeakageIssue("error", f"Forbidden xP feature column present: {column}"))
        if column in FORBIDDEN_FEATURE_NAMES and column != "target_total_points":
            issues.append(LeakageIssue("error", f"Target/same-fixture outcome appears as feature: {column}"))

    registry_names = registered_feature_names()
    produced_features = [
        column
        for column in features.columns
        if column not in ALLOWED_NON_FEATURE_COLUMNS and not column.endswith("_source_kickoff")
    ]
    unregistered = sorted(set(produced_features).difference(registry_names))
    if unregistered:
        issues.append(LeakageIssue("error", f"Unregistered feature columns: {', '.join(unregistered)}"))

    for timestamp_column in ("kickoff_time", "information_cutoff"):
        if timestamp_column in features.columns:
            parsed = pd.to_datetime(features[timestamp_column], utc=True, errors="coerce")
            if parsed.isna().any():
                issues.append(LeakageIssue("error", f"{timestamp_column} contains unparsable timestamps."))
    if "information_cutoff" in features.columns:
        cutoff = pd.to_datetime(features["information_cutoff"], utc=True, errors="coerce")
        for lineage_column in (
            "player_max_source_available_time",
            "team_max_source_available_time",
            "opponent_max_source_available_time",
        ):
            if lineage_column not in features.columns:
                issues.append(LeakageIssue("error", f"Missing lineage column: {lineage_column}"))
                continue
            source_time = pd.to_datetime(features[lineage_column], utc=True, errors="coerce")
            leaked = source_time.notna() & cutoff.notna() & (source_time >= cutoff)
            if leaked.any():
                issues.append(
                    LeakageIssue(
                        "error",
                        f"{lineage_column} has {int(leaked.sum())} rows not available before information_cutoff.",
                    )
                )

    gw1 = features.loc[pd.to_numeric(features.get("gameweek"), errors="coerce") == 1]
    for column in ("prev3_minutes_sum", "prev5_minutes_sum", "season_to_date_minutes"):
        if column in gw1.columns and pd.to_numeric(gw1[column], errors="coerce").fillna(0).ne(0).any():
            issues.append(LeakageIssue("error", f"GW1 {column} must be zero or missing."))

    if {"prior_source_season", "season"}.issubset(features.columns):
        bad_prior = features.loc[
            features["prior_source_season"].notna()
            & (features["prior_source_season"].astype(str) >= features["season"].astype(str))
        ]
        if not bad_prior.empty:
            issues.append(LeakageIssue("error", "Prior-season join points at same or later season."))

    suspicious = [column for column in features.columns if "season_total" in column.lower()]
    if suspicious:
        issues.append(
            LeakageIssue(
                "error",
                f"Naive end-of-season cumulative-looking columns found: {', '.join(suspicious)}",
            )
        )
    return LeakageAuditResult(issues)
