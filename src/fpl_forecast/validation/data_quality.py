from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, RAW_VAASTAV_DIR
from fpl_forecast.ingest.snapshots import latest_snapshot_path
from fpl_forecast.ingest.vaastav import MERGED_GW


ERROR = "error"
WARNING = "warning"

PROVENANCE_COLUMNS = ["source", "source_version", "retrieved_at", "season", "raw_snapshot_path"]
REAL_SOURCES = {"fpl_api", "vaastav"}
VALID_PLAYER_POSITION_VALUES = {"GKP", "GK", "DEF", "MID", "FWD"}
VALID_FPL_POSITION_VALUES = {"GKP", "DEF", "MID", "FWD", "AM"}
HISTORICAL_TABLES = {"historical_player_fixtures", "historical_player_gameweeks"}
SYNTHETIC_SOURCE_MARKERS = {"synthetic", "demo", "fake", "invented"}
SYNTHETIC_TEXT_MARKERS = {
    "Demo FC",
    "Synthetic United",
    "Synthetic FC",
    "Forecast XI",
    "Optimizer FC",
    "Test United",
}
SYNTHETIC_PLAYER_IDS = {-1, 0, 999001, 999002, 999003, 9000001, 9000002}


@dataclass(frozen=True)
class DataQualityIssue:
    severity: str
    table: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    issues: list[DataQualityIssue]

    @property
    def errors(self) -> list[DataQualityIssue]:
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> list[DataQualityIssue]:
        return [issue for issue in self.issues if issue.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_all(
    *,
    normalized_dir=NORMALIZED_DIR,
    raw_vaastav_dir=RAW_VAASTAV_DIR,
) -> ValidationResult:
    normalized_dir = Path(normalized_dir)
    issues: list[DataQualityIssue] = []
    tables = _load_tables(normalized_dir)
    if not tables:
        issues.append(DataQualityIssue(ERROR, "normalized", "No normalized Parquet tables found."))
        return ValidationResult(issues)

    for path, frame in tables.items():
        table = path.stem
        issues.extend(_validate_table(table, frame))

    issues.extend(_validate_fixture_team_references(tables))
    issues.extend(_validate_historical_xp(raw_vaastav_dir, tables))
    return ValidationResult(issues)


def _load_tables(normalized_dir: Path) -> dict[Path, pd.DataFrame]:
    return {
        path: pd.read_parquet(path)
        for path in sorted(normalized_dir.glob("**/*.parquet"))
        if "outputs" not in path.parts and "phase2" not in path.parts
    }


def _validate_table(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    required = _required_columns(table)
    if required:
        issues.extend(_require_columns(table, frame, required))
    issues.extend(_validate_provenance(table, frame))
    issues.extend(_validate_real_source_labels(table, frame))
    issues.extend(_validate_synthetic_contamination(table, frame))

    if table == "current_players":
        issues.extend(_require_unique(table, frame, ["player_id"]))
        issues.extend(_require_no_nulls(table, frame, ["player_id", "player_code", "team_id"]))
        issues.extend(_validate_positions(table, frame))
        issues.extend(_validate_nonnegative(table, frame, ["price_tenths", "minutes"]))
    elif table == "current_teams":
        issues.extend(_require_unique(table, frame, ["team_id"]))
        issues.extend(_require_no_nulls(table, frame, ["team_id"]))
    elif table == "current_fixtures":
        issues.extend(_require_unique(table, frame, ["fixture_id"]))
        issues.extend(_require_no_nulls(table, frame, ["fixture_id", "home_team_id", "away_team_id"]))
        issues.extend(_validate_gameweeks(table, frame, required=False))
    elif table in HISTORICAL_TABLES:
        issues.extend(_require_unique(table, frame, ["player_id", "fixture_id"]))
        issues.extend(_require_no_nulls(table, frame, ["player_id", "fixture_id", "gameweek"]))
        issues.extend(_validate_positions(table, frame))
        issues.extend(_validate_historical_positions(table, frame))
        issues.extend(_validate_nonnegative(table, frame, ["minutes"]))
        issues.extend(_validate_fixture_minutes(table, frame))
        issues.extend(_validate_gameweeks(table, frame, required=True))
        issues.extend(_validate_optional_expected_fields(table, frame))
        xp_columns = [column for column in frame.columns if column.lower() == "xp"]
        if xp_columns:
            issues.append(
                DataQualityIssue(
                    ERROR,
                    table,
                    "xP is present in normalized historical data; exclude it from default features.",
                )
            )
    issues.extend(_validate_timestamps(table, frame))
    return issues


def _required_columns(table: str) -> list[str]:
    base = list(PROVENANCE_COLUMNS)
    if table == "current_players":
        return [
            "player_id",
            "player_code",
            "team_id",
            "position_id",
            "position",
            "price_tenths",
            *base,
        ]
    if table == "current_teams":
        return ["team_id", "team_code", "team_name", *base]
    if table == "current_fixtures":
        return ["fixture_id", "gameweek", "home_team_id", "away_team_id", *base]
    if table in HISTORICAL_TABLES:
        return [
            "player_id",
            "player_code",
            "fixture_id",
            "gameweek",
            "minutes",
            "total_points",
            "opponent_team",
            "was_home",
            "kickoff_time",
            *base,
        ]
    return base


def _require_columns(table: str, frame: pd.DataFrame, columns: Iterable[str]) -> list[DataQualityIssue]:
    missing = sorted(set(columns).difference(frame.columns))
    if not missing:
        return []
    return [
        DataQualityIssue(
            ERROR,
            table,
            f"Missing required columns: {', '.join(missing)}",
        )
    ]


def _require_unique(table: str, frame: pd.DataFrame, columns: list[str]) -> list[DataQualityIssue]:
    if not set(columns).issubset(frame.columns) or frame.empty:
        return []
    duplicate_count = int(frame.duplicated(subset=columns).sum())
    if duplicate_count == 0:
        return []
    return [
        DataQualityIssue(
            ERROR,
            table,
            f"Primary key columns {', '.join(columns)} contain {duplicate_count} duplicate rows.",
        )
    ]


def _require_no_nulls(table: str, frame: pd.DataFrame, columns: list[str]) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    for column in columns:
        if column in frame.columns:
            null_count = int(frame[column].isna().sum())
            if null_count:
                issues.append(DataQualityIssue(ERROR, table, f"{column} contains {null_count} nulls."))
    return issues


def _validate_positions(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    if "position" not in frame.columns:
        return []
    invalid = frame["position"].dropna().loc[
        ~frame["position"].dropna().isin(VALID_PLAYER_POSITION_VALUES)
    ]
    if invalid.empty:
        return []
    examples = ", ".join(map(str, sorted(invalid.astype(str).unique())[:5]))
    return [DataQualityIssue(ERROR, table, f"Invalid position values found: {examples}")]


def _validate_historical_positions(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    if table not in HISTORICAL_TABLES:
        return []
    issues: list[DataQualityIssue] = []
    if "fpl_position" in frame.columns:
        values = frame["fpl_position"].dropna()
        invalid = values.loc[~values.isin(VALID_FPL_POSITION_VALUES)]
        if not invalid.empty:
            examples = ", ".join(map(str, sorted(invalid.astype(str).unique())[:5]))
            issues.append(DataQualityIssue(ERROR, table, f"Invalid fpl_position values found: {examples}"))
    if {"fpl_position", "element_type"}.issubset(frame.columns):
        bad_am = frame.loc[
            (frame["fpl_position"] == "AM")
            & (pd.to_numeric(frame["element_type"], errors="coerce") != 5)
        ]
        if not bad_am.empty:
            issues.append(
                DataQualityIssue(
                    ERROR,
                    table,
                    "AM fpl_position appears on rows whose element_type is not 5.",
                )
            )
    return issues


def _validate_nonnegative(table: str, frame: pd.DataFrame, columns: list[str]) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            bad_count = int((values < 0).sum())
            if bad_count:
                issues.append(DataQualityIssue(ERROR, table, f"{column} has {bad_count} negative rows."))
    return issues


def _validate_fixture_minutes(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    if "minutes" not in frame.columns:
        return []
    values = pd.to_numeric(frame["minutes"], errors="coerce")
    bad_count = int((values > 130).sum())
    if bad_count:
        return [
            DataQualityIssue(
                ERROR,
                table,
                f"minutes has {bad_count} rows above a plausible single-fixture range.",
            )
        ]
    return []


def _validate_gameweeks(table: str, frame: pd.DataFrame, *, required: bool) -> list[DataQualityIssue]:
    if "gameweek" not in frame.columns:
        return []
    values = pd.to_numeric(frame["gameweek"], errors="coerce")
    issues: list[DataQualityIssue] = []
    null_count = int(values.isna().sum())
    if null_count and required:
        issues.append(DataQualityIssue(ERROR, table, f"gameweek contains {null_count} nulls."))
    elif null_count:
        issues.append(DataQualityIssue(WARNING, table, f"gameweek contains {null_count} nulls."))
    outside = int(((values < 1) | (values > 38)).fillna(False).sum())
    if outside:
        issues.append(DataQualityIssue(ERROR, table, f"gameweek has {outside} rows outside 1..38."))
    return issues


def _validate_optional_expected_fields(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    optional_fields = {
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    }
    present = optional_fields.intersection(frame.columns)
    if present:
        return []
    return [
        DataQualityIssue(
            WARNING,
            table,
            "No optional expected-goal columns are present.",
        )
    ]


def _validate_provenance(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    issues = _require_columns(table, frame, PROVENANCE_COLUMNS)
    for column in PROVENANCE_COLUMNS:
        if column in frame.columns and int(frame[column].isna().sum()):
            issues.append(DataQualityIssue(ERROR, table, f"Provenance column {column} contains nulls."))
    return issues


def _validate_timestamps(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    for column in ("retrieved_at", "kickoff_time"):
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
        bad_count = int(parsed.isna().sum())
        if bad_count:
            issues.append(DataQualityIssue(ERROR, table, f"{column} has {bad_count} unparsable timestamps."))
    return issues


def _validate_real_source_labels(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    if "source" not in frame.columns:
        return []
    source_values = set(frame["source"].dropna().astype(str).unique())
    invalid = sorted(source_values.difference(REAL_SOURCES))
    if invalid:
        return [
            DataQualityIssue(
                ERROR,
                table,
                f"Invalid real-data source labels: {', '.join(invalid)}",
            )
        ]
    return []


def _validate_synthetic_contamination(table: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    if "source" in frame.columns:
        lowered = frame["source"].dropna().astype(str).str.lower()
        if lowered.apply(lambda value: any(marker in value for marker in SYNTHETIC_SOURCE_MARKERS)).any():
            issues.append(DataQualityIssue(ERROR, table, "Synthetic source marker found."))

    if "raw_snapshot_path" in frame.columns:
        paths = frame["raw_snapshot_path"].dropna().astype(str)
        if paths.str.contains("outputs/synthetic_demo", regex=False).any():
            issues.append(DataQualityIssue(ERROR, table, "Normalized table points at synthetic demo output."))

    if "player_id" in frame.columns:
        ids = set(pd.to_numeric(frame["player_id"], errors="coerce").dropna().astype(int))
        markers = sorted(ids.intersection(SYNTHETIC_PLAYER_IDS))
        if markers:
            issues.append(
                DataQualityIssue(
                    ERROR,
                    table,
                    f"Synthetic demo player IDs found: {', '.join(map(str, markers))}",
                )
            )

    text_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_string_dtype(frame[column]) or frame[column].dtype == object
    ]
    for column in text_columns:
        series = frame[column].dropna().astype(str)
        for marker in SYNTHETIC_TEXT_MARKERS:
            if series.str.contains(marker, regex=False).any():
                issues.append(
                    DataQualityIssue(ERROR, table, f"Synthetic demo text marker found: {marker}")
                )
                return issues
    return issues


def _validate_fixture_team_references(tables: dict[Path, pd.DataFrame]) -> list[DataQualityIssue]:
    current_teams = _first_table(tables, "current_teams")
    fixtures = _first_table(tables, "current_fixtures")
    if current_teams is None or fixtures is None:
        return []
    if "team_id" not in current_teams.columns:
        return []
    required_fixture_columns = {"home_team_id", "away_team_id"}
    if not required_fixture_columns.issubset(fixtures.columns):
        return []

    team_ids = set(pd.to_numeric(current_teams["team_id"], errors="coerce").dropna().astype(int))
    fixture_team_ids = set(
        pd.concat([fixtures["home_team_id"], fixtures["away_team_id"]])
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
        .astype(int)
    )
    missing = sorted(fixture_team_ids.difference(team_ids))
    if missing:
        return [
            DataQualityIssue(
                ERROR,
                "current_fixtures",
                f"Fixtures reference team IDs absent from current_teams: {', '.join(map(str, missing))}",
            )
        ]
    return []


def _validate_historical_xp(
    raw_vaastav_dir: Path,
    tables: dict[Path, pd.DataFrame],
) -> list[DataQualityIssue]:
    historical = _first_table(tables, "historical_player_fixtures")
    if historical is None:
        historical = _first_table(tables, "historical_player_gameweeks")
    if historical is None or "season" not in historical.columns:
        return []
    issues: list[DataQualityIssue] = []
    for season in sorted(historical["season"].dropna().astype(str).unique()):
        raw_path = latest_snapshot_path(
            Path(raw_vaastav_dir),
            season=season,
            endpoint_name=MERGED_GW,
            content_type="csv",
        )
        if raw_path is None:
            continue
        columns = pd.read_csv(raw_path, nrows=0).columns
        if "xP" in columns:
            issues.append(
                DataQualityIssue(
                    WARNING,
                    "historical_player_fixtures",
                    "Raw Vaastav merged_gw.csv contains xP; normalized table should not use it.",
                )
            )
    return issues


def _first_table(tables: dict[Path, pd.DataFrame], stem: str) -> pd.DataFrame | None:
    for path, frame in tables.items():
        if path.stem == stem:
            return frame
    return None
