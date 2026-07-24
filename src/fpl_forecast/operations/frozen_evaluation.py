from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR, PROJECT_ROOT
from fpl_forecast.minutes_model.config import CONFIG_PATH as MINUTES_CONFIG_PATH
from fpl_forecast.minutes_model.config import load_minutes_config
from fpl_forecast.minutes_model.data import load_minutes_frame
from fpl_forecast.minutes_model.metrics import minutes_metric_tables, population_coverage, score_minutes_predictions
from fpl_forecast.minutes_model.models import fit_predict_minutes_models
from fpl_forecast.team_model.config import CONFIG_PATH as TEAM_CONFIG_PATH
from fpl_forecast.team_model.config import load_team_model_config
from fpl_forecast.team_model.data import frozen_prediction_columns, load_historical_team_fixtures
from fpl_forecast.team_model.metrics import score_fixture_predictions, team_metric_tables
from fpl_forecast.team_model.models import fit_predict_models
from fpl_forecast.team_model.probabilities import add_probability_columns
from fpl_forecast.xpoints.config import CONFIG_PATH as XPOINTS_CONFIG_PATH
from fpl_forecast.xpoints.config import load_xpoints_config
from fpl_forecast.xpoints.data import load_xpoints_frame
from fpl_forecast.xpoints.metrics import metric_tables as xpoints_metric_tables
from fpl_forecast.xpoints.metrics import score_predictions as score_xpoints_predictions
from fpl_forecast.xpoints.models import predict_xpoints_models


FROZEN_EVALUATION_DIR = PROJECT_ROOT / "reports" / "operational" / "frozen_evaluation"
DEFAULT_TRAINING_SEASONS = ("2022-23", "2023-24", "2024-25")


@dataclass(frozen=True)
class FrozenEvaluationResult:
    run_id: str
    run_dir: Path
    summary_path: Path
    metadata_path: Path
    metrics: dict[str, Any]


def run_frozen_out_of_time_evaluation(
    *,
    run_id: str,
    evaluation_season: str = "2025-26",
    training_seasons: tuple[str, ...] = DEFAULT_TRAINING_SEASONS,
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> FrozenEvaluationResult:
    seasons = [*training_seasons, evaluation_season]
    run_dir = FROZEN_EVALUATION_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    team_config = load_team_model_config()
    team_frame = load_historical_team_fixtures(seasons=seasons, normalized_dir=normalized_dir)
    team_frame = team_frame.loc[team_frame["result_valid"].astype(bool)].copy()
    team_train = team_frame.loc[team_frame["season"].isin(training_seasons)].copy()
    team_test = team_frame.loc[team_frame["season"].eq(evaluation_season)].copy()
    team_target = team_test[frozen_prediction_columns(team_test)].copy()
    team_predictions, team_ratings, team_diagnostics = fit_predict_models(
        team_train,
        team_target,
        cutoff=pd.to_datetime(team_test["information_cutoff"], utc=True).min(),
        config=team_config,
        models=["T2_REGULARIZED_ATTACK_DEFENCE"],
    )
    team_predictions = add_probability_columns(team_predictions, team_config)
    team_scored = score_fixture_predictions(team_predictions, team_test, team_config)
    team_tables = team_metric_tables(team_scored, team_config)

    minutes_config = load_minutes_config()
    minutes_frame = load_minutes_frame(seasons=seasons, normalized_dir=normalized_dir)
    minutes_train = minutes_frame.loc[minutes_frame["season"].isin(training_seasons)].copy()
    minutes_test = minutes_frame.loc[minutes_frame["season"].eq(evaluation_season)].copy()
    minutes_predictions, minutes_diagnostics = fit_predict_minutes_models(
        minutes_train,
        minutes_test,
        config=minutes_config,
        models=["M3_EWMA_MINUTES"],
    )
    minutes_predictions["minutes_variant"] = "M3"
    minutes_scored = score_minutes_predictions(minutes_predictions, minutes_test)
    minutes_tables = minutes_metric_tables(minutes_scored, minutes_config)

    xpoints_config = load_xpoints_config()
    xpoints_frame = load_xpoints_frame(seasons=seasons, normalized_dir=normalized_dir)
    xpoints_train = xpoints_frame.loc[xpoints_frame["season"].isin(training_seasons)].copy()
    xpoints_test = xpoints_frame.loc[xpoints_frame["season"].eq(evaluation_season)].copy()
    phase3_reference = pd.DataFrame(columns=["season", "stable_fixture_uid", "player_uid", "expected_points"])
    xpoints_predictions, xpoints_draws, xpoints_conservation = predict_xpoints_models(
        xpoints_train,
        xpoints_test,
        minutes_predictions=minutes_predictions,
        team_predictions=team_predictions,
        phase3_reference=phase3_reference,
        config=xpoints_config,
        fold_index=0,
    )
    xpoints_scored = score_xpoints_predictions(xpoints_predictions, xpoints_test)
    xpoints_scored_m3 = xpoints_scored.loc[xpoints_scored["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M3")].copy()
    xpoints_tables = xpoints_metric_tables(xpoints_scored_m3, xpoints_config)

    _write_frame(team_predictions, run_dir / "team_frozen_fixture_predictions.parquet")
    _write_frame(team_scored, run_dir / "team_scored_fixture_predictions.parquet")
    _write_frame(team_tables.expected_goals, run_dir / "team_metrics_expected_goals.csv")
    _write_frame(team_tables.clean_sheet, run_dir / "team_metrics_clean_sheet.csv")
    _write_frame(team_tables.outcome, run_dir / "team_metrics_outcome.csv")
    _write_frame(minutes_predictions, run_dir / "minutes_frozen_predictions.parquet")
    _write_frame(minutes_scored, run_dir / "minutes_scored_predictions.parquet")
    _write_frame(minutes_tables.overall, run_dir / "minutes_metrics_overall.csv")
    _write_frame(minutes_tables.binary, run_dir / "minutes_metrics_binary.csv")
    _write_frame(population_coverage(minutes_scored), run_dir / "minutes_population_coverage.csv")
    _write_frame(xpoints_predictions, run_dir / "xpoints_frozen_player_fixture_predictions.parquet")
    _write_frame(xpoints_scored_m3, run_dir / "xpoints_scored_player_fixture_predictions.parquet")
    _write_frame(xpoints_tables.overall, run_dir / "xpoints_metrics_overall.csv")
    _write_frame(xpoints_tables.by_population, run_dir / "xpoints_metrics_by_population.csv")
    _write_frame(xpoints_tables.distribution, run_dir / "xpoints_metrics_distribution.csv")
    _write_frame(xpoints_conservation, run_dir / "xpoints_conservation.csv")
    _write_frame(team_ratings, run_dir / "team_ratings.csv")

    summary = _summary_metrics(
        team_tables=team_tables,
        minutes_tables=minutes_tables,
        minutes_scored=minutes_scored,
        xpoints_tables=xpoints_tables,
        xpoints_scored=xpoints_scored_m3,
    )
    summary_path = run_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "evaluation_label": "frozen_out_of_time",
        "evaluation_season": evaluation_season,
        "training_seasons": list(training_seasons),
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": _git_revision(),
        "model_names": {
            "team": "T2_REGULARIZED_ATTACK_DEFENCE",
            "minutes": "M3_EWMA_MINUTES",
            "xpoints": "X2_TEAM_CONSTRAINED_SIM_M3",
        },
        "config_hashes": {
            "team": _file_sha256(TEAM_CONFIG_PATH),
            "minutes": _file_sha256(MINUTES_CONFIG_PATH),
            "xpoints": _file_sha256(XPOINTS_CONFIG_PATH),
        },
        "row_counts": {
            "team_train": int(len(team_train)),
            "team_test": int(len(team_test)),
            "minutes_train": int(len(minutes_train)),
            "minutes_test": int(len(minutes_test)),
            "xpoints_train": int(len(xpoints_train)),
            "xpoints_test": int(len(xpoints_test)),
        },
        "diagnostics": {
            "team": team_diagnostics,
            "minutes": [diagnostic.__dict__ for diagnostic in minutes_diagnostics],
        },
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return FrozenEvaluationResult(run_id, run_dir, summary_path, metadata_path, summary)


def frozen_evaluation_lines(result: FrozenEvaluationResult) -> list[str]:
    metrics = result.metrics
    return [
        f"run_id={result.run_id}",
        f"run_dir={result.run_dir}",
        f"summary={result.summary_path}",
        f"metadata={result.metadata_path}",
        f"team_goal_mae={metrics['team']['goal_mae']:.4f}",
        f"team_poisson_nll={metrics['team']['poisson_nll']:.4f}",
        f"minutes_mae={metrics['minutes']['mae']:.4f}",
        f"minutes_appearance_brier={metrics['minutes']['appearance_brier']:.4f}",
        f"xpoints_mae={metrics['xpoints']['mae']:.4f}",
        f"xpoints_prob_ge_5_brier={metrics['xpoints']['prob_ge_5_brier']:.4f}",
    ]


def _summary_metrics(
    *,
    team_tables,
    minutes_tables,
    minutes_scored: pd.DataFrame,
    xpoints_tables,
    xpoints_scored: pd.DataFrame,
) -> dict[str, Any]:
    team_goals = team_tables.expected_goals.iloc[0].to_dict()
    team_clean = team_tables.clean_sheet.iloc[0].to_dict()
    team_outcome = team_tables.outcome.iloc[0].to_dict()
    minutes = minutes_tables.overall.iloc[0].to_dict()
    minutes_binary = minutes_tables.binary.set_index("target")
    xpoints = xpoints_tables.overall.iloc[0].to_dict()
    xpoints_distribution = xpoints_tables.distribution.iloc[0].to_dict()
    cold = xpoints_tables.by_population.loc[xpoints_tables.by_population["pre_deadline_population"].eq("cold_start_no_history")]
    return {
        "team": {
            "fixtures": int(team_goals["fixtures"]),
            "goal_mae": float(team_goals["goal_mae"]),
            "goal_rmse": float(team_goals["goal_rmse"]),
            "goal_bias": float(team_goals["goal_bias"]),
            "poisson_nll": float(team_goals["poisson_nll"]),
            "clean_sheet_brier": float(team_clean["brier"]),
            "match_outcome_log_loss": float(team_outcome["multiclass_log_loss"]),
        },
        "minutes": {
            "rows": int(minutes["rows"]),
            "mae": float(minutes["mae"]),
            "rmse": float(minutes["rmse"]),
            "bias": float(minutes["bias"]),
            "appearance_brier": float(minutes_binary.loc["appearance", "brier"]),
            "start_brier": float(minutes_binary.loc["start", "brier"]),
            "population_coverage": int(
                minutes_scored.drop_duplicates(["season", "stable_fixture_uid", "player_uid"]).shape[0]
            ),
        },
        "xpoints": {
            "rows": int(xpoints["rows"]),
            "mae": float(xpoints["mae"]),
            "rmse": float(xpoints["rmse"]),
            "bias": float(xpoints["bias"]),
            "spearman": float(xpoints["spearman"]),
            "prob_ge_5_brier": float(xpoints_distribution["prob_ge_5_brier"]),
            "central_80_coverage": float(xpoints_distribution["central_80_coverage"]),
            "cold_start": cold.to_dict(orient="records"),
            "negative_expected_points": int(xpoints_scored["expected_points"].lt(0).sum()),
        },
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None
