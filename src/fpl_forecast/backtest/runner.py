from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fpl_forecast.backtest.baselines import BASELINES, predict_baselines
from fpl_forecast.backtest.config import REPORTS_DIR, BacktestConfig, load_backtest_config
from fpl_forecast.backtest.folds import BacktestFold, chronological_gameweek_folds, split_fold
from fpl_forecast.backtest.metrics import aggregate_player_gameweek, metric_tables, score_predictions
from fpl_forecast.backtest.populations import assign_pre_deadline_populations, population_coverage
from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.features.leakage import audit_leakage
from fpl_forecast.panel.common import parse_seasons, phase2_dir


@dataclass(frozen=True)
class BacktestRunResult:
    run_id: str
    run_dir: Path
    folds: list[BacktestFold]
    manifest_path: Path
    frozen_predictions_path: Path
    scored_fixture_path: Path
    player_gameweek_path: Path
    metrics_paths: dict[str, Path]


def run_baseline_backtest(
    *,
    seasons: str | list[str],
    test_seasons: str | list[str],
    mode: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
    reports_dir: Path | str = REPORTS_DIR,
    run_id: str | None = None,
    bootstrap_samples: int | None = None,
    seed: int | None = None,
) -> BacktestRunResult:
    season_list = parse_seasons(seasons)
    test_season_list = parse_seasons(test_seasons)
    config = load_backtest_config()
    if bootstrap_samples is not None:
        config = dataclass_replace(config, bootstrap_samples=bootstrap_samples)
    if seed is not None:
        config = dataclass_replace(config, bootstrap_seed=seed)
    leakage = audit_leakage(seasons=season_list, normalized_dir=normalized_dir)
    if leakage.errors:
        messages = "; ".join(issue.message for issue in leakage.errors)
        raise ValueError(f"Phase 2 leakage audit failed: {messages}")

    frame = _load_backtest_frame(normalized_dir=normalized_dir, seasons=season_list)
    _reject_forbidden_model_fields(frame, config)
    folds = chronological_gameweek_folds(
        frame,
        test_seasons=test_season_list,
        mode=mode,
        minimum_training_rows=config.minimum_training_rows,
    )
    predictions = []
    for fold in folds:
        train, test = split_fold(frame, fold, mode=mode)
        predictions.append(predict_baselines(train, test, config=config).assign(fold_id=fold.fold_id))
    frozen = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    scored_fixture = score_predictions(frozen) if not frozen.empty else pd.DataFrame()
    scored_player_gameweek = (
        aggregate_player_gameweek(scored_fixture) if not scored_fixture.empty else pd.DataFrame()
    )
    tables = metric_tables(
        scored_fixture,
        scored_player_gameweek,
        reference_baseline=config.reference_baseline,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )

    run_id = run_id or _run_id(mode)
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = _write_frame(frozen, run_dir / "frozen_fixture_predictions.parquet")
    scored_path = _write_frame(scored_fixture, run_dir / "scored_fixture_predictions.parquet")
    player_gw_path = _write_frame(
        scored_player_gameweek,
        run_dir / "scored_player_gameweek_predictions.parquet",
    )
    metrics_paths = {
        "overall": _write_frame(tables.overall, run_dir / "metrics_overall.csv"),
        "by_season": _write_frame(tables.by_season, run_dir / "metrics_by_season.csv"),
        "by_gameweek": _write_frame(tables.by_gameweek, run_dir / "metrics_by_gameweek.csv"),
        "by_position": _write_frame(tables.by_position, run_dir / "metrics_by_position.csv"),
        "by_population": _write_frame(tables.by_population, run_dir / "metrics_by_population.csv"),
        "calibration": _write_frame(tables.calibration, run_dir / "calibration.csv"),
        "ranking": _write_frame(tables.ranking, run_dir / "ranking_topk.csv"),
        "ranking_by_population": _write_frame(
            tables.ranking_by_population,
            run_dir / "ranking_by_population_topk.csv",
        ),
        "population_coverage": _write_frame(
            population_coverage(scored_player_gameweek),
            run_dir / "population_coverage.csv",
        ),
        "bootstrap": _write_frame(tables.bootstrap, run_dir / "bootstrap_mae_differences.csv"),
    }
    manifest = _manifest(
        run_id=run_id,
        mode=mode,
        seasons=season_list,
        test_seasons=test_season_list,
        config=config,
        folds=folds,
        normalized_dir=Path(normalized_dir),
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return BacktestRunResult(
        run_id=run_id,
        run_dir=run_dir,
        folds=folds,
        manifest_path=manifest_path,
        frozen_predictions_path=frozen_path,
        scored_fixture_path=scored_path,
        player_gameweek_path=player_gw_path,
        metrics_paths=metrics_paths,
    )


def compare_baselines(*, run_id: str, reports_dir: Path | str = REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    overall = pd.read_csv(run_dir / "metrics_overall.csv")
    bootstrap = pd.read_csv(run_dir / "bootstrap_mae_differences.csv")
    lines = [f"run_id={run_id}", "overall_mae:"]
    lines.extend(
        f"{row.baseline}: rows={int(row.rows)} mae={row.mae:.4f} rmse={row.rmse:.4f} bias={row.bias:.4f}"
        for row in overall.sort_values("mae").itertuples(index=False)
    )
    if not bootstrap.empty:
        lines.append("mae_difference_vs_reference:")
        lines.extend(
            f"{row.baseline} - {row.reference_baseline}: mean={row.mean_mae_difference:.4f} "
            f"ci95=[{row.ci_lower:.4f}, {row.ci_upper:.4f}] "
            f"blocks={int(row.evaluated_gameweeks)}"
            for row in bootstrap.itertuples(index=False)
        )
    return lines


def inspect_backtest_run(*, run_id: str, reports_dir: Path | str = REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    scored = pd.read_parquet(run_dir / "scored_player_gameweek_predictions.parquet")
    lines = [
        f"run_id={run_id}",
        f"mode={manifest['mode']}",
        f"folds={len(manifest['folds'])}",
        f"seasons={','.join(manifest['seasons'])}",
        f"test_seasons={','.join(manifest['test_seasons'])}",
        f"rows={len(scored)}",
        f"baselines={','.join(BASELINES)}",
        f"cutoff_min={manifest['cutoff_range']['min']}",
        f"cutoff_max={manifest['cutoff_range']['max']}",
        f"dirty_worktree={manifest['git']['dirty']}",
    ]
    return lines


def _load_backtest_frame(*, normalized_dir: Path | str, seasons: list[str]) -> pd.DataFrame:
    output_dir = phase2_dir(normalized_dir)
    features = pd.read_parquet(output_dir / "features_player_fixture.parquet")
    fact = pd.read_parquet(output_dir / "fact_player_fixture.parquet")
    fact_columns = [
        "season",
        "player_uid",
        "fixture_key",
        "player_name",
        "fpl_position",
        "minutes",
    ]
    frame = features.merge(
        fact[fact_columns],
        on=["season", "player_uid", "fixture_key"],
        how="left",
    )
    frame = frame.loc[frame["season"].isin(seasons)].copy()
    frame["target_total_points"] = pd.to_numeric(frame["target_total_points"], errors="coerce")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0)
    frame = assign_pre_deadline_populations(frame)
    return frame


def _reject_forbidden_model_fields(frame: pd.DataFrame, config: BacktestConfig) -> None:
    lower_columns = {column.lower() for column in frame.columns}
    forbidden = {value.lower() for value in config.forbidden_fields}
    present = sorted(lower_columns & forbidden)
    if present:
        raise ValueError(f"Forbidden Phase 3 model fields are present: {', '.join(present)}")


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def _run_id(mode: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"phase3_{mode}_{timestamp}"


def _manifest(
    *,
    run_id: str,
    mode: str,
    seasons: list[str],
    test_seasons: list[str],
    config: BacktestConfig,
    folds: list[BacktestFold],
    normalized_dir: Path,
) -> dict[str, object]:
    feature_path = phase2_dir(normalized_dir) / "features_player_fixture.parquet"
    fact_path = phase2_dir(normalized_dir) / "fact_player_fixture.parquet"
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
        "source_data": {
            "fact_player_fixture": str(fact_path),
            "fact_hash": _file_hash(fact_path),
            "features_player_fixture": str(feature_path),
            "features_hash": _file_hash(feature_path),
        },
        "feature_registry": {
            "path": str(PROJECT_ROOT / "src" / "fpl_forecast" / "features" / "registry.json"),
            "backtest_config": str(PROJECT_ROOT / "src" / "fpl_forecast" / "backtest" / "config.json"),
        },
        "git": {
            "commit": _git(["rev-parse", "--short", "HEAD"]),
            "dirty": bool(_git(["status", "--short"])),
        },
        "leakage_audit": "passed",
        "package_version": "0.1.0",
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def dataclass_replace(config: BacktestConfig, **updates: object) -> BacktestConfig:
    values = asdict(config)
    values.update(updates)
    if isinstance(values["rolling_windows"], list):
        values["rolling_windows"] = tuple(values["rolling_windows"])
    if isinstance(values["forbidden_fields"], list):
        values["forbidden_fields"] = tuple(values["forbidden_fields"])
    return BacktestConfig(**values)
