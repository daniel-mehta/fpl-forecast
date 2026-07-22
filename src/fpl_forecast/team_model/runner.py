from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.features.leakage import audit_leakage
from fpl_forecast.panel.common import parse_seasons
from fpl_forecast.team_model.config import TEAM_REPORTS_DIR, TeamModelConfig, load_team_model_config
from fpl_forecast.team_model.data import (
    assert_frozen_predictions_target_free,
    file_hash,
    frozen_prediction_columns,
    load_current_fixture_frame,
    load_historical_team_fixtures,
)
from fpl_forecast.team_model.metrics import score_fixture_predictions, team_metric_tables
from fpl_forecast.team_model.models import MODEL_NAMES, fit_predict_models
from fpl_forecast.team_model.probabilities import add_probability_columns


@dataclass(frozen=True)
class TeamFold:
    fold_id: str
    season: str
    gameweek: int
    information_cutoff: pd.Timestamp
    train_fixtures: int
    test_fixtures: int


@dataclass(frozen=True)
class TeamRunResult:
    run_id: str
    run_dir: Path
    folds: list[TeamFold]
    manifest_path: Path
    frozen_predictions_path: Path
    scored_predictions_path: Path
    metrics_paths: dict[str, Path]


def run_team_backtest(
    *,
    seasons: str | list[str],
    test_seasons: str | list[str],
    mode: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
    reports_dir: Path | str = TEAM_REPORTS_DIR,
    run_id: str | None = None,
) -> TeamRunResult:
    season_list = parse_seasons(seasons)
    test_season_list = parse_seasons(test_seasons)
    config = load_team_model_config()
    leakage = audit_leakage(seasons=season_list, normalized_dir=normalized_dir)
    if leakage.errors:
        messages = "; ".join(issue.message for issue in leakage.errors)
        raise ValueError(f"Phase 2 leakage audit failed: {messages}")
    fixtures = load_historical_team_fixtures(seasons=season_list, normalized_dir=normalized_dir)
    folds = build_team_folds(fixtures, test_seasons=test_season_list, mode=mode)

    prediction_frames = []
    rating_frames = []
    diagnostics: list[dict[str, object]] = []
    for fold in folds:
        train, test = split_team_fold(fixtures, fold, mode=mode)
        predicted, ratings, model_diagnostics = fit_predict_models(
            train,
            test.drop(columns=["home_goals", "away_goals"], errors="ignore"),
            cutoff=fold.information_cutoff,
            config=config,
        )
        prediction_frames.append(predicted.assign(fold_id=fold.fold_id))
        rating_frames.append(ratings.assign(fold_id=fold.fold_id))
        diagnostics.extend([{**item, "fold_id": fold.fold_id} for item in model_diagnostics])
    frozen = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    frozen = add_probability_columns(frozen, config) if not frozen.empty else frozen
    assert_frozen_predictions_target_free(frozen)
    outcomes = fixtures.loc[fixtures["result_valid"].astype(bool)]
    scored = score_fixture_predictions(frozen, outcomes, config) if not frozen.empty else pd.DataFrame()
    tables = team_metric_tables(scored, config) if not scored.empty else None

    run_id = run_id or f"phase4_team_{mode}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = _write_frame(frozen[frozen_prediction_columns(frozen)], run_dir / "frozen_fixture_predictions.parquet")
    scored_path = _write_frame(scored, run_dir / "scored_fixture_predictions.parquet")
    _write_frame(pd.concat(rating_frames, ignore_index=True), run_dir / "team_ratings.parquet")
    _write_frame(pd.DataFrame(diagnostics), run_dir / "fit_diagnostics.csv")
    metrics_paths = {}
    if tables is not None:
        metrics_paths = {
            "expected_goals": _write_frame(tables.expected_goals, run_dir / "metrics_expected_goals.csv"),
            "expected_goals_by_season": _write_frame(
                tables.expected_goals_by_season,
                run_dir / "metrics_expected_goals_by_season.csv",
            ),
            "expected_goals_by_gameweek": _write_frame(
                tables.expected_goals_by_gameweek,
                run_dir / "metrics_expected_goals_by_gameweek.csv",
            ),
            "expected_goals_by_side": _write_frame(
                tables.expected_goals_by_side,
                run_dir / "metrics_expected_goals_by_side.csv",
            ),
            "expected_goals_by_fallback": _write_frame(
                tables.expected_goals_by_fallback,
                run_dir / "metrics_expected_goals_by_fallback.csv",
            ),
            "clean_sheet": _write_frame(tables.clean_sheet, run_dir / "metrics_clean_sheet.csv"),
            "clean_sheet_calibration": _write_frame(
                tables.clean_sheet_calibration,
                run_dir / "clean_sheet_calibration.csv",
            ),
            "outcome": _write_frame(tables.outcome, run_dir / "metrics_match_outcome.csv"),
            "joint_score": _write_frame(tables.joint_score, run_dir / "metrics_joint_score.csv"),
            "joint_score_by_season": _write_frame(
                tables.joint_score_by_season,
                run_dir / "metrics_joint_score_by_season.csv",
            ),
            "low_score": _write_frame(tables.low_score, run_dir / "metrics_low_score.csv"),
            "low_score_by_season": _write_frame(
                tables.low_score_by_season,
                run_dir / "metrics_low_score_by_season.csv",
            ),
            "bootstrap": _write_frame(tables.bootstrap, run_dir / "bootstrap_goal_mae_differences.csv"),
            "bootstrap_t3_vs_t2": _write_frame(tables.bootstrap_t3_vs_t2, run_dir / "bootstrap_t3_vs_t2.csv"),
        }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                run_id=run_id,
                mode=mode,
                seasons=season_list,
                test_seasons=test_season_list,
                folds=folds,
                config=config,
                normalized_dir=Path(normalized_dir),
                diagnostics=diagnostics,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TeamRunResult(run_id, run_dir, folds, manifest_path, frozen_path, scored_path, metrics_paths)


def forecast_team_fixtures(
    *,
    season: str,
    gameweek: int | None,
    as_of: str,
    seasons: str | list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
    reports_dir: Path | str = TEAM_REPORTS_DIR,
    run_id: str | None = None,
) -> Path:
    config = load_team_model_config()
    historical = load_historical_team_fixtures(seasons=seasons, normalized_dir=normalized_dir)
    as_of_ts = pd.to_datetime(as_of, utc=True)
    train = historical.loc[
        historical["result_valid"].astype(bool)
        & (pd.to_datetime(historical["source_available_time"], utc=True) < as_of_ts)
    ].copy()
    current = load_current_fixture_frame(season=season, as_of=as_of_ts.isoformat(), normalized_dir=normalized_dir)
    if gameweek is not None:
        current = current.loc[pd.to_numeric(current["gameweek"], errors="coerce") == gameweek].copy()
    if current.empty:
        raise ValueError("No forecastable future fixtures match the requested filters.")
    predicted, ratings, diagnostics = fit_predict_models(
        train,
        current,
        cutoff=as_of_ts,
        config=config,
        models=["T2_REGULARIZED_ATTACK_DEFENCE"],
    )
    frozen = add_probability_columns(predicted, config)
    assert_frozen_predictions_target_free(frozen)
    run_id = run_id or f"phase4_team_forecast_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = _write_frame(frozen[frozen_prediction_columns(frozen)], run_dir / "future_fixture_predictions.parquet")
    _write_frame(ratings, run_dir / "team_ratings.parquet")
    _write_frame(pd.DataFrame(diagnostics), run_dir / "fit_diagnostics.csv")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "mode": "future",
                "season": season,
                "gameweek": gameweek,
                "as_of": as_of_ts.isoformat(),
                "historical_seasons": parse_seasons(seasons),
                "training_fixtures": int(len(train)),
                "forecast_fixtures": int(len(current)),
                "baseline_parameters": asdict(config),
                "git": _git_metadata(),
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def build_team_folds(fixtures: pd.DataFrame, *, test_seasons: list[str], mode: str) -> list[TeamFold]:
    if mode not in {"rolling", "gw1"}:
        raise ValueError("mode must be 'rolling' or 'gw1'.")
    valid = fixtures.loc[fixtures["result_valid"].astype(bool)].copy()
    valid["information_cutoff"] = pd.to_datetime(valid["information_cutoff"], utc=True)
    groups = (
        valid.loc[valid["season"].isin(test_seasons)]
        .groupby(["season", "gameweek"], as_index=False)
        .agg(information_cutoff=("information_cutoff", "min"), test_fixtures=("stable_fixture_uid", "nunique"))
        .sort_values(["season", "gameweek"])
    )
    if mode == "gw1":
        groups = groups.loc[pd.to_numeric(groups["gameweek"], errors="coerce") == 1]
    folds = []
    for row in groups.itertuples(index=False):
        cutoff = pd.Timestamp(row.information_cutoff)
        train = valid.loc[pd.to_datetime(valid["source_available_time"], utc=True) < cutoff]
        if mode == "gw1":
            train = train.loc[train["season"].map(_season_start) < _season_start(row.season)]
        train = train.loc[~((train["season"] == row.season) & (train["gameweek"] == row.gameweek))]
        if train.empty:
            continue
        folds.append(
            TeamFold(
                fold_id=f"{row.season}_GW{int(row.gameweek):02d}",
                season=str(row.season),
                gameweek=int(row.gameweek),
                information_cutoff=cutoff,
                train_fixtures=int(len(train)),
                test_fixtures=int(row.test_fixtures),
            )
        )
    return folds


def split_team_fold(fixtures: pd.DataFrame, fold: TeamFold, *, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = fixtures.loc[fixtures["result_valid"].astype(bool)].copy()
    valid["source_available_time"] = pd.to_datetime(valid["source_available_time"], utc=True)
    test = valid.loc[(valid["season"] == fold.season) & (valid["gameweek"] == fold.gameweek)].copy()
    train = valid.loc[
        (valid["source_available_time"] < fold.information_cutoff)
        & ~((valid["season"] == fold.season) & (valid["gameweek"] == fold.gameweek))
    ].copy()
    if mode == "gw1":
        train = train.loc[train["season"].map(_season_start) < _season_start(fold.season)].copy()
    if not train["source_available_time"].lt(fold.information_cutoff).all():
        raise ValueError(f"Fold {fold.fold_id} contains unavailable training results.")
    return train, test


def compare_team_models(*, run_id: str, reports_dir: Path | str = TEAM_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    goals = pd.read_csv(run_dir / "metrics_expected_goals.csv")
    clean = pd.read_csv(run_dir / "metrics_clean_sheet.csv")
    outcome = pd.read_csv(run_dir / "metrics_match_outcome.csv")
    joint = pd.read_csv(run_dir / "metrics_joint_score.csv")
    lines = [f"run_id={run_id}", "expected_goal_metrics:"]
    lines.extend(
        f"{row.model_name}: fixtures={int(row.fixtures)} goal_mae={row.goal_mae:.4f} "
        f"goal_rmse={row.goal_rmse:.4f} bias={row.goal_bias:.4f} poisson_nll={row.poisson_nll:.4f}"
        for row in goals.sort_values("goal_mae").itertuples(index=False)
    )
    lines.append("clean_sheet_metrics:")
    lines.extend(
        f"{row.model_name}: brier={row.brier:.4f} log_loss={row.log_loss:.4f} "
        f"predicted_rate={row.predicted_rate:.4f} observed_rate={row.observed_rate:.4f}"
        for row in clean.sort_values("brier").itertuples(index=False)
    )
    lines.append("match_outcome_metrics:")
    lines.extend(
        f"{row.model_name}: log_loss={row.multiclass_log_loss:.4f} "
        f"brier={row.multiclass_brier:.4f} draw_brier={row.draw_brier:.4f} "
        f"accuracy={row.accuracy:.4f}"
        for row in outcome.sort_values("multiclass_log_loss").itertuples(index=False)
    )
    lines.append("joint_score_metrics:")
    lines.extend(
        f"{row.model_name}: joint_score_nll={row.joint_score_nll:.4f} "
        f"exact_score_accuracy={row.exact_score_accuracy:.4f}"
        for row in joint.sort_values("joint_score_nll").itertuples(index=False)
    )
    return lines


def inspect_team_run(*, run_id: str, reports_dir: Path | str = TEAM_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen = pd.read_parquet(run_dir / "frozen_fixture_predictions.parquet")
    scored = pd.read_parquet(run_dir / "scored_fixture_predictions.parquet")
    return [
        f"run_id={run_id}",
        f"mode={manifest['mode']}",
        f"folds={len(manifest['folds'])}",
        f"frozen_rows={len(frozen)}",
        f"scored_rows={len(scored)}",
        f"models={','.join(MODEL_NAMES)}",
        f"cutoff_min={manifest['cutoff_range']['min']}",
        f"cutoff_max={manifest['cutoff_range']['max']}",
        f"dirty_worktree={manifest['git']['dirty']}",
    ]


def _manifest(
    *,
    run_id: str,
    mode: str,
    seasons: list[str],
    test_seasons: list[str],
    folds: list[TeamFold],
    config: TeamModelConfig,
    normalized_dir: Path,
    diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    fixture_path = normalized_dir / "phase2" / "dim_fixture.parquet"
    cutoff_values = [fold.information_cutoff.isoformat() for fold in folds]
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "seasons": seasons,
        "test_seasons": test_seasons,
        "folds": [
            {
                **asdict(fold),
                "information_cutoff": fold.information_cutoff.isoformat(),
            }
            for fold in folds
        ],
        "cutoff_range": {
            "min": min(cutoff_values) if cutoff_values else None,
            "max": max(cutoff_values) if cutoff_values else None,
        },
        "baseline_parameters": asdict(config),
        "source_data": {"dim_fixture": str(fixture_path), "dim_fixture_hash": file_hash(fixture_path)},
        "model_config": str(PROJECT_ROOT / "src" / "fpl_forecast" / "team_model" / "config.json"),
        "leakage_audit": "passed",
        "git": _git_metadata(),
        "package_version": "0.1.0",
        "fit_diagnostics": diagnostics,
        "command_family": "team_model",
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def _git_metadata() -> dict[str, object]:
    return {
        "commit": _git(["rev-parse", "--short", "HEAD"]),
        "dirty": bool(_git(["status", "--short"])),
    }


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _season_start(season: str) -> int:
    return int(str(season).split("-", maxsplit=1)[0])
