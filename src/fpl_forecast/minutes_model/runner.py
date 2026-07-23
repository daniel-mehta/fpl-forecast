from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fpl_forecast.backtest.folds import BacktestFold, chronological_gameweek_folds, split_fold
from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.features.leakage import audit_leakage
from fpl_forecast.minutes_model.coherence import add_lineup_coherence
from fpl_forecast.minutes_model.config import (
    MINUTES_REPORTS_DIR,
    MinutesModelConfig,
    load_minutes_config,
)
from fpl_forecast.minutes_model.data import (
    assert_frozen_predictions_target_free,
    file_hash,
    frozen_prediction_columns,
    load_minutes_frame,
)
from fpl_forecast.minutes_model.inference import validate_current_minutes_inputs
from fpl_forecast.minutes_model.metrics import (
    minutes_metric_tables,
    population_coverage,
    score_minutes_predictions,
)
from fpl_forecast.minutes_model.models import MODEL_NAMES, fit_predict_minutes_models
from fpl_forecast.panel.common import parse_seasons, phase2_dir


@dataclass(frozen=True)
class MinutesRunResult:
    run_id: str
    run_dir: Path
    folds: list[BacktestFold]
    manifest_path: Path
    frozen_predictions_path: Path
    scored_predictions_path: Path
    metrics_paths: dict[str, Path]


def run_minutes_backtest(
    *,
    seasons: str | list[str],
    test_seasons: str | list[str],
    mode: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
    reports_dir: Path | str = MINUTES_REPORTS_DIR,
    run_id: str | None = None,
    bootstrap_samples: int | None = None,
    seed: int | None = None,
) -> MinutesRunResult:
    season_list = parse_seasons(seasons)
    test_season_list = parse_seasons(test_seasons)
    config = _config_with_overrides(load_minutes_config(), bootstrap_samples, seed)
    leakage = audit_leakage(seasons=season_list, normalized_dir=normalized_dir)
    if leakage.errors:
        messages = "; ".join(issue.message for issue in leakage.errors)
        raise ValueError(f"Phase 2 leakage audit failed: {messages}")
    frame = load_minutes_frame(seasons=season_list, normalized_dir=normalized_dir)
    folds = chronological_gameweek_folds(frame, test_seasons=test_season_list, mode=mode)
    prediction_frames = []
    diagnostic_rows: list[dict[str, object]] = []
    for fold in folds:
        train, test = split_fold(frame, fold, mode=mode)
        if not train["source_available_time"].lt(fold.information_cutoff).all():
            raise ValueError(f"Fold {fold.fold_id} contains unavailable training labels.")
        if not test["max_feature_source_available_time"].lt(fold.information_cutoff).all():
            raise ValueError(f"Fold {fold.fold_id} contains features unavailable at cutoff.")
        predicted, diagnostics = fit_predict_minutes_models(train, test, config=config)
        prediction_frames.append(predicted.assign(fold_id=fold.fold_id))
        diagnostic_rows.extend([{**asdict(item), "fold_id": fold.fold_id} for item in diagnostics])
    frozen = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    lineup = pd.DataFrame()
    if not frozen.empty:
        frozen, lineup = add_lineup_coherence(frozen, config)
        assert_frozen_predictions_target_free(frozen, config.forbidden_frozen_columns)
    scored = score_minutes_predictions(frozen, frame) if not frozen.empty else pd.DataFrame()
    tables = minutes_metric_tables(scored, config) if not scored.empty else None

    run_id = run_id or f"phase5_minutes_{mode}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = _write_frame(frozen[frozen_prediction_columns(frozen)], run_dir / "frozen_predictions.parquet")
    scored_path = _write_frame(scored, run_dir / "scored_predictions.parquet")
    metrics_paths: dict[str, Path] = {
        "population_coverage": _write_frame(population_coverage(scored), run_dir / "population_coverage.csv"),
        "lineup_coherence": _write_frame(lineup, run_dir / "lineup_coherence.csv"),
        "fit_diagnostics": _write_frame(pd.DataFrame(diagnostic_rows), run_dir / "fit_diagnostics.csv"),
    }
    if tables is not None:
        metrics_paths.update(
            {
                "overall": _write_frame(tables.overall, run_dir / "metrics_overall.csv"),
                "by_population": _write_frame(tables.by_population, run_dir / "metrics_by_population.csv"),
                "by_season": _write_frame(tables.by_season, run_dir / "metrics_by_season.csv"),
                "binary": _write_frame(tables.binary, run_dir / "metrics_binary.csv"),
                "state": _write_frame(tables.state, run_dir / "metrics_state.csv"),
                "calibration": _write_frame(tables.calibration, run_dir / "calibration_minutes.csv"),
                "ranking": _write_frame(tables.ranking, run_dir / "ranking_topk.csv"),
                "bootstrap": _write_frame(tables.bootstrap, run_dir / "bootstrap_mae_differences.csv"),
            }
        )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(run_id, mode, season_list, test_season_list, folds, config, Path(normalized_dir)),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return MinutesRunResult(run_id, run_dir, folds, manifest_path, frozen_path, scored_path, metrics_paths)


def compare_minutes_models(*, run_id: str, reports_dir: Path | str = MINUTES_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    overall = pd.read_csv(run_dir / "metrics_overall.csv")
    binary = pd.read_csv(run_dir / "metrics_binary.csv")
    lines = [f"run_id={run_id}", "minutes_metrics:"]
    lines.extend(
        f"{row.model_name}: rows={int(row.rows)} mae={row.mae:.4f} rmse={row.rmse:.4f} "
        f"bias={row.bias:.4f} spearman={row.spearman:.4f}"
        for row in overall.sort_values("mae").itertuples(index=False)
    )
    lines.append("binary_metrics:")
    for row in binary.sort_values(["target", "brier"]).itertuples(index=False):
        lines.append(
            f"{row.model_name} {row.target}: rows={int(row.rows)} brier={row.brier:.4f} "
            f"log_loss={row.log_loss:.4f} pred={row.predicted_rate:.4f} actual={row.observed_rate:.4f}"
        )
    return lines


def inspect_minutes_run(*, run_id: str, reports_dir: Path | str = MINUTES_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen = pd.read_parquet(run_dir / "frozen_predictions.parquet")
    scored = pd.read_parquet(run_dir / "scored_predictions.parquet")
    coverage = pd.read_csv(run_dir / "population_coverage.csv")
    return [
        f"run_id={run_id}",
        f"mode={manifest['mode']}",
        f"folds={len(manifest['folds'])}",
        f"frozen_rows={len(frozen)}",
        f"scored_rows={len(scored)}",
        f"models={','.join(MODEL_NAMES)}",
        f"starts_exact_coverage={manifest['starts_exact_coverage']:.4f}",
        f"cold_start_rows={int(coverage.loc[coverage['evaluation_population'].eq('cold_start_no_history'), 'rows'].sum())}",
        f"cutoff_min={manifest['cutoff_range']['min']}",
        f"cutoff_max={manifest['cutoff_range']['max']}",
        f"dirty_worktree={manifest['git']['dirty']}",
    ]


def forecast_minutes(
    *,
    season: str,
    gameweek: int | None,
    as_of: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> Path:
    validate_current_minutes_inputs(season=season, gameweek=gameweek, as_of=as_of, normalized_dir=normalized_dir)
    raise ValueError(
        "Phase 5 current player-minute inference requires official future-season launch data; "
        "historical backtests are available, but no current player forecasts were written."
    )


def _config_with_overrides(
    config: MinutesModelConfig,
    bootstrap_samples: int | None,
    seed: int | None,
) -> MinutesModelConfig:
    data = asdict(config)
    if bootstrap_samples is not None:
        data["bootstrap_samples"] = bootstrap_samples
    if seed is not None:
        data["bootstrap_seed"] = seed
    data["model_names"] = tuple(data["model_names"])
    data["state_order"] = tuple(data["state_order"])
    data["calibration_bins"] = tuple(data["calibration_bins"])
    data["forbidden_frozen_columns"] = tuple(data["forbidden_frozen_columns"])
    return MinutesModelConfig(**data)


def _manifest(
    run_id: str,
    mode: str,
    seasons: list[str],
    test_seasons: list[str],
    folds: list[BacktestFold],
    config: MinutesModelConfig,
    normalized_dir: Path,
) -> dict[str, object]:
    fact_path = phase2_dir(normalized_dir) / "fact_player_fixture.parquet"
    cutoff_values = [fold.information_cutoff.isoformat() for fold in folds]
    frame = pd.read_parquet(fact_path, columns=["starts", "entity_type"])
    players = frame.loc[frame["entity_type"].eq("player")]
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "seasons": seasons,
        "test_seasons": test_seasons,
        "folds": [
            {**asdict(fold), "information_cutoff": fold.information_cutoff.isoformat()} for fold in folds
        ],
        "cutoff_range": {
            "min": min(cutoff_values) if cutoff_values else None,
            "max": max(cutoff_values) if cutoff_values else None,
        },
        "model_parameters": asdict(config),
        "source_data": {"fact_player_fixture": str(fact_path), "fact_hash": file_hash(fact_path)},
        "model_config": str(PROJECT_ROOT / "src" / "fpl_forecast" / "minutes_model" / "config.json"),
        "candidate_rule": config.candidate_rule,
        "starts_exact_coverage": float(pd.to_numeric(players["starts"], errors="coerce").notna().mean()),
        "leakage_audit": "passed",
        "git": _git_metadata(),
        "package_version": "0.1.0",
        "command_family": "minutes_model",
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
