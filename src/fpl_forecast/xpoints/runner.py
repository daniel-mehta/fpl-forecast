from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_forecast.backtest.folds import BacktestFold, chronological_gameweek_folds, split_fold
from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.features.leakage import audit_leakage
from fpl_forecast.panel.common import parse_seasons, phase2_dir
from fpl_forecast.team_model.data import load_current_fixture_frame
from fpl_forecast.xpoints.config import XPOINTS_REPORTS_DIR, XPointsConfig, load_xpoints_config
from fpl_forecast.xpoints.data import (
    assert_frozen_target_free,
    file_hash,
    frozen_prediction_columns,
    load_minutes_predictions,
    load_phase3_reference,
    load_team_predictions,
    load_xpoints_frame,
)
from fpl_forecast.xpoints.metrics import metric_tables, score_predictions
from fpl_forecast.xpoints.models import predict_xpoints_models
from fpl_forecast.xpoints.scoring import reconstruction_audit
from fpl_forecast.xpoints.simulation import aggregate_gameweek_draws


@dataclass(frozen=True)
class XPointsRunResult:
    run_id: str
    run_dir: Path
    folds: list[BacktestFold]
    manifest_path: Path
    frozen_predictions_path: Path
    scored_predictions_path: Path
    player_gameweek_path: Path
    metrics_paths: dict[str, Path]


def validate_scoring(
    *,
    seasons: str | list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
    reports_dir: Path | str = XPOINTS_REPORTS_DIR,
    run_id: str = "scoring_reconstruction",
) -> dict[str, Path]:
    frame = load_xpoints_frame(seasons=seasons, normalized_dir=normalized_dir)
    audit = reconstruction_audit(frame)
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        name: _write_frame(table, run_dir / f"{name}.csv")
        for name, table in audit.items()
    }


def run_xpoints_backtest(
    *,
    seasons: str | list[str],
    test_seasons: str | list[str],
    mode: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
    reports_dir: Path | str = XPOINTS_REPORTS_DIR,
    run_id: str | None = None,
) -> XPointsRunResult:
    if mode not in {"rolling", "gw1"}:
        raise ValueError("mode must be 'rolling' or 'gw1'.")
    season_list = parse_seasons(seasons)
    test_season_list = parse_seasons(test_seasons)
    config = load_xpoints_config()
    leakage = audit_leakage(seasons=season_list, normalized_dir=normalized_dir)
    if leakage.errors:
        messages = "; ".join(issue.message for issue in leakage.errors)
        raise ValueError(f"Phase 2 leakage audit failed: {messages}")
    frame = load_xpoints_frame(seasons=season_list, normalized_dir=normalized_dir)
    minutes = load_minutes_predictions(mode=mode, config=config)
    team = load_team_predictions(mode=mode, config=config)
    phase3 = load_phase3_reference(mode=mode, config=config)
    folds = chronological_gameweek_folds(frame, test_seasons=test_season_list, mode=mode)
    prediction_frames = []
    draw_frames = []
    conservation_frames = []
    for fold_index, fold in enumerate(folds):
        train, test = split_fold(frame, fold, mode=mode)
        test = test.assign(fold_id=fold.fold_id)
        train = train.assign(fold_id="train")
        fold_minutes = minutes.loc[minutes["fold_id"].eq(fold.fold_id)].copy()
        fold_team = team.loc[team["fold_id"].eq(fold.fold_id)].copy()
        fold_phase3 = phase3.loc[phase3["fold_id"].eq(fold.fold_id)].copy()
        predictions, draws, conservation = predict_xpoints_models(
            train,
            test,
            minutes_predictions=fold_minutes,
            team_predictions=fold_team,
            phase3_reference=fold_phase3,
            config=config,
            fold_index=fold_index,
        )
        predictions = predictions.assign(fold_id=fold.fold_id)
        prediction_frames.append(predictions)
        draw_frames.append(pd.DataFrame(draws))
        conservation_frames.append(conservation.assign(fold_id=fold.fold_id))
    frozen_internal = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    draws_all = pd.concat(draw_frames, ignore_index=True).to_numpy(dtype=np.int16) if draw_frames else np.empty((0, 0))
    assert_frozen_target_free(frozen_internal, config.forbidden_frozen_columns)
    scored = score_predictions(frozen_internal, frame) if not frozen_internal.empty else pd.DataFrame()
    gameweek = (
        aggregate_gameweek_draws(
            frozen_internal,
            draws_all,
            key_columns=["season", "gameweek", "player_uid", "model_name", "pre_deadline_population"],
        )
        if len(frozen_internal)
        else pd.DataFrame()
    )
    tables = metric_tables(scored, config) if not scored.empty else None
    scoring_audit = reconstruction_audit(frame)

    run_id = run_id or f"phase6_xpoints_{mode}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(reports_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = frozen_internal[frozen_prediction_columns(frozen_internal)].copy()
    frozen_path = _write_frame(frozen, run_dir / "frozen_player_fixture_predictions.parquet")
    scored_path = _write_frame(scored, run_dir / "scored_player_fixture_predictions.parquet")
    gameweek_path = _write_frame(gameweek, run_dir / "player_gameweek_predictions.parquet")
    metrics_paths: dict[str, Path] = {
        "conservation": _write_frame(
            pd.concat(conservation_frames, ignore_index=True), run_dir / "conservation_diagnostics.csv"
        ),
        "scoring_by_season_position": _write_frame(
            scoring_audit["by_season_position"], run_dir / "scoring_reconstruction_by_season_position.csv"
        ),
        "scoring_difference_counts": _write_frame(
            scoring_audit["difference_counts"], run_dir / "scoring_reconstruction_difference_counts.csv"
        ),
        "scoring_component_nulls": _write_frame(
            scoring_audit["component_nulls"], run_dir / "scoring_component_nulls.csv"
        ),
        "scoring_mismatches": _write_frame(
            scoring_audit["mismatches"], run_dir / "scoring_reconstruction_mismatches.csv"
        ),
    }
    if tables is not None:
        metrics_paths.update(
            {
                "overall": _write_frame(tables.overall, run_dir / "metrics_overall.csv"),
                "by_population": _write_frame(tables.by_population, run_dir / "metrics_by_population.csv"),
                "by_position": _write_frame(tables.by_position, run_dir / "metrics_by_position.csv"),
                "by_season": _write_frame(tables.by_season, run_dir / "metrics_by_season.csv"),
                "calibration": _write_frame(tables.calibration, run_dir / "calibration_points.csv"),
                "ranking": _write_frame(tables.ranking, run_dir / "ranking_topk.csv"),
                "distribution": _write_frame(tables.distribution, run_dir / "metrics_distribution.csv"),
                "components": _write_frame(tables.components, run_dir / "metrics_components.csv"),
                "bootstrap": _write_frame(tables.bootstrap, run_dir / "bootstrap_mae_difference.csv"),
            }
        )
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
                frozen_rows=len(frozen),
                scored_rows=len(scored),
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return XPointsRunResult(run_id, run_dir, folds, manifest_path, frozen_path, scored_path, gameweek_path, metrics_paths)


def compare_xpoints(*, run_id: str, reports_dir: Path | str = XPOINTS_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    overall = pd.read_csv(run_dir / "metrics_overall.csv")
    distribution = pd.read_csv(run_dir / "metrics_distribution.csv")
    bootstrap = pd.read_csv(run_dir / "bootstrap_mae_difference.csv")
    lines = [f"run_id={run_id}", "point_metrics:"]
    lines.extend(
        f"{row.model_name}: rows={int(row.rows)} mae={row.mae:.4f} rmse={row.rmse:.4f} "
        f"bias={row.bias:.4f} spearman={row.spearman:.4f}"
        for row in overall.sort_values("mae").itertuples(index=False)
    )
    lines.append("distribution_metrics:")
    lines.extend(
        f"{row.model_name}: ge5_brier={row.prob_ge_5_brier:.4f} "
        f"central80={row.central_80_coverage:.4f} zero_pred={row.zero_rate_predicted:.4f} "
        f"zero_actual={row.zero_rate_actual:.4f}"
        for row in distribution.itertuples(index=False)
    )
    if not bootstrap.empty:
        row = bootstrap.iloc[0]
        lines.append(
            f"bootstrap {row['model_name']} - {row['reference_model']}: "
            f"mean_mae_difference={row['mean_mae_difference']:.4f} "
            f"ci95=[{row['ci_lower']:.4f},{row['ci_upper']:.4f}] "
            f"blocks={int(row['evaluated_gameweeks'])}"
        )
    return lines


def inspect_xpoints(*, run_id: str, reports_dir: Path | str = XPOINTS_REPORTS_DIR) -> list[str]:
    run_dir = Path(reports_dir) / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen = pd.read_parquet(run_dir / "frozen_player_fixture_predictions.parquet")
    scored = pd.read_parquet(run_dir / "scored_player_fixture_predictions.parquet")
    gameweek = pd.read_parquet(run_dir / "player_gameweek_predictions.parquet")
    conservation = pd.read_csv(run_dir / "conservation_diagnostics.csv")
    return [
        f"run_id={run_id}",
        f"mode={manifest['mode']}",
        f"folds={len(manifest['folds'])}",
        f"frozen_rows={len(frozen)}",
        f"scored_rows={len(scored)}",
        f"player_gameweek_rows={len(gameweek)}",
        f"models={','.join(manifest['model_names'])}",
        f"max_goal_conservation_abs_error={conservation['goal_conservation_error'].abs().max():.8f}",
        f"cutoff_min={manifest['cutoff_range']['min']}",
        f"cutoff_max={manifest['cutoff_range']['max']}",
        f"dirty_worktree={manifest['git']['dirty']}",
    ]


def forecast_xpoints(
    *,
    season: str,
    gameweek: int | None,
    as_of: str,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> Path:
    current = load_current_fixture_frame(season=season, as_of=as_of, normalized_dir=normalized_dir)
    if gameweek is not None:
        current = current.loc[pd.to_numeric(current["gameweek"], errors="coerce").eq(gameweek)].copy()
    if current.empty:
        raise ValueError("No forecastable future fixtures match the requested filters.")
    raise ValueError(
        "Phase 6 current xPoints inference requires genuine target-season launch player/minutes "
        "forecasts; historical backtests are available, but no current xPoints were written."
    )


def _manifest(
    *,
    run_id: str,
    mode: str,
    seasons: list[str],
    test_seasons: list[str],
    folds: list[BacktestFold],
    config: XPointsConfig,
    normalized_dir: Path,
    frozen_rows: int,
    scored_rows: int,
) -> dict[str, object]:
    fact_path = phase2_dir(normalized_dir) / "fact_player_fixture.parquet"
    cutoffs = [fold.information_cutoff.isoformat() for fold in folds]
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "seasons": seasons,
        "test_seasons": test_seasons,
        "folds": [{**asdict(fold), "information_cutoff": fold.information_cutoff.isoformat()} for fold in folds],
        "cutoff_range": {"min": min(cutoffs) if cutoffs else None, "max": max(cutoffs) if cutoffs else None},
        "model_names": list(config.model_names),
        "default_model": config.default_model,
        "rules_version": config.rules_version,
        "simulation_version": config.simulation_version,
        "draw_count": config.draw_count,
        "random_seed": config.random_seed,
        "phase4_team_runs": config.team_runs,
        "phase5_minutes_runs": config.minutes_runs,
        "phase3_reference_runs": config.phase3_runs,
        "source_data": {"fact_player_fixture": str(fact_path), "fact_hash": file_hash(fact_path)},
        "config_hash": file_hash(PROJECT_ROOT / "src" / "fpl_forecast" / "xpoints" / "config.json"),
        "frozen_rows": frozen_rows,
        "scored_rows": scored_rows,
        "git": _git_metadata(),
        "leakage_audit": "passed",
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def _git_metadata() -> dict[str, object]:
    return {"commit": _git(["rev-parse", "--short", "HEAD"]), "dirty": bool(_git(["status", "--short"]))}


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip()
