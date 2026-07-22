from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fpl_forecast.backtest.baselines import predict_baselines
from fpl_forecast.backtest.config import load_backtest_config
from fpl_forecast.backtest.folds import chronological_gameweek_folds, split_fold
from fpl_forecast.backtest.metrics import (
    aggregate_player_gameweek,
    block_bootstrap_differences,
    metric_tables,
    score_predictions,
)
from fpl_forecast.backtest.populations import (
    COLD_START_NO_HISTORY,
    PRE_DEADLINE_HISTORY_ACTIVE,
    assign_pre_deadline_populations,
    population_coverage,
)
from fpl_forecast.backtest.runner import run_baseline_backtest
from fpl_forecast.cli import app


def test_chronological_folds_are_whole_gameweeks_and_not_random():
    frame = _backtest_frame()

    folds = chronological_gameweek_folds(
        frame,
        test_seasons=["2023-24"],
        mode="rolling",
    )

    assert [fold.fold_id for fold in folds] == ["2023-24_GW01", "2023-24_GW02"]
    assert folds[0].test_rows == 4
    assert folds[1].test_rows == 3
    assert folds[0].information_cutoff < folds[1].information_cutoff


def test_split_fold_uses_only_available_training_labels_and_excludes_same_gameweek():
    frame = _backtest_frame()
    fold = chronological_gameweek_folds(frame, test_seasons=["2023-24"], mode="rolling")[1]

    train, test = split_fold(frame, fold, mode="rolling")

    assert train["source_available_time"].lt(fold.information_cutoff).all()
    assert not ((train["season"] == "2023-24") & (train["gameweek"] == 2)).any()
    assert set(test["fixture_key"]) == {"2023-24:3", "2023-24:4"}


def test_gw1_fold_isolates_season_boundary_and_keeps_expected_cases():
    frame = _backtest_frame()

    folds = chronological_gameweek_folds(frame, test_seasons=["2023-24"], mode="gw1")
    train, test = split_fold(frame, folds[0], mode="gw1")

    assert folds[0].fold_id == "2023-24_GW01"
    assert set(train["season"]) == {"2022-23"}
    assert {"returning", "transferred", "position_change", "new_player"} <= set(test["case"])
    assert (test.loc[test["case"] == "position_change", "fpl_position"] == "MID").all()


def test_baselines_have_hand_checked_values_and_no_zero_history_failure():
    config = load_backtest_config()
    train = pd.DataFrame(
        [
            _row("2022-23", 1, "p1", "Known", "MID", "team_a", 1, "2022-08-01T12:00:00Z", 2, 90),
            _row("2022-23", 2, "p2", "Known", "DEF", "team_b", 1, "2022-08-01T12:00:00Z", 6, 90),
        ]
    )
    test = pd.DataFrame(
        [
            _row(
                "2023-24",
                1,
                "p3",
                "No History",
                "MID",
                "team_c",
                1,
                "2023-08-01T12:00:00Z",
                0,
                0,
                prev3_points=9,
                prev3_minutes=180,
                prior_appearances=2,
            ),
            _row("2023-24", 1, "p4", "Cold", "FWD", "team_c", 1, "2023-08-01T12:00:00Z", 0, 0),
        ]
    )

    pred = predict_baselines(train, test, config=config)
    p3 = pred.loc[(pred["player_uid"] == "p3") & (pred["baseline"] == "B3_RECENT_POINTS_P3")]
    cold = pred.loc[(pred["player_uid"] == "p4") & (pred["baseline"] == "B3_RECENT_POINTS_P3")]

    assert p3.iloc[0]["prediction"] == pytest.approx(4.5)
    assert cold.iloc[0]["prediction"] == pytest.approx(4.0)
    assert pred["prediction"].notna().all()


def test_b6_previous_season_shrunk_rate_formula_for_gw1_cases():
    config = load_backtest_config()
    train = pd.DataFrame(
        [
            _row("2022-23", 1, "returning", "Return", "DEF", "team_a", 1, "2022-08-01T12:00:00Z", 6, 90),
            _row("2022-23", 2, "returning", "Return", "DEF", "team_a", 2, "2022-08-08T12:00:00Z", 4, 90),
            _row("2022-23", 1, "transfer", "Mover", "MID", "team_b", 1, "2022-08-01T12:00:00Z", 3, 90),
            _row("2022-23", 1, "position", "Switch", "DEF", "team_c", 1, "2022-08-01T12:00:00Z", 2, 45),
        ]
    )
    test = pd.DataFrame(
        [
            _row("2023-24", 1, "returning", "Return", "DEF", "team_a", 1, "2023-08-01T12:00:00Z", 0, 0),
            _row("2023-24", 1, "transfer", "Mover", "MID", "team_new", 1, "2023-08-01T12:00:00Z", 0, 0),
            _row("2023-24", 1, "position", "Switch", "MID", "team_c", 1, "2023-08-01T12:00:00Z", 0, 0),
            _row("2023-24", 1, "new", "Cold", "FWD", "team_d", 1, "2023-08-01T12:00:00Z", 0, 0),
        ]
    )

    pred = predict_baselines(train, test, config=config)
    b6 = pred.loc[pred["baseline"] == "B6_PREVIOUS_SEASON_GW1"].set_index("player_uid")
    global_mean = 15 / 4
    global_per90 = 15 * 90 / 315
    prior_minutes = 5 * 90

    assert b6.loc["returning", "prediction"] == pytest.approx(
        ((10 * 90) + (global_per90 * prior_minutes)) / (180 + prior_minutes)
    )
    assert b6.loc["transfer", "prediction"] == pytest.approx(
        ((3 * 90) + (global_per90 * prior_minutes)) / (90 + prior_minutes)
    )
    assert b6.loc["position", "fpl_position"] == "MID"
    assert b6.loc["position", "prediction"] == pytest.approx(
        (((2 * 90) + (global_per90 * prior_minutes)) / (45 + prior_minutes)) * 0.5
    )
    assert b6.loc["new", "prediction"] == pytest.approx(global_mean)


def test_pre_deadline_candidate_populations_use_only_lagged_history():
    frame = assign_pre_deadline_populations(_backtest_frame())

    gw1 = frame.loc[(frame["season"] == "2023-24") & (frame["gameweek"] == 1)]
    gw2 = frame.loc[(frame["season"] == "2023-24") & (frame["gameweek"] == 2)]

    assert set(gw1.loc[gw1["player_uid"].isin(["p1", "p2"]), "pre_deadline_population"]) == {
        PRE_DEADLINE_HISTORY_ACTIVE
    }
    assert set(gw1.loc[gw1["player_uid"].isin(["p3", "p4"]), "pre_deadline_population"]) == {
        COLD_START_NO_HISTORY
    }
    assert set(gw2.loc[gw2["player_uid"].isin(["p1", "p2"]), "pre_deadline_population"]) == {
        PRE_DEADLINE_HISTORY_ACTIVE
    }


def test_double_gameweek_aggregation_metrics_and_constant_spearman():
    predictions = pd.DataFrame(
        [
            _prediction("2023-24:3", "p1", 2, 1.0),
            _prediction("2023-24:4", "p1", 3, 1.0),
            _prediction("2023-24:3", "p2", 1, 1.0),
            _prediction("2023-24:4", "p2", 0, 1.0),
        ]
    )
    scored = score_predictions(predictions)
    player_gw = aggregate_player_gameweek(scored)
    tables = metric_tables(
        scored,
        player_gw,
        reference_baseline="B1_GLOBAL_MEAN",
        bootstrap_samples=5,
        bootstrap_seed=7,
    )

    assert player_gw.loc[player_gw["player_uid"] == "p1", "target_total_points"].iloc[0] == 5
    assert pd.isna(tables.overall.loc[0, "spearman"])
    assert tables.overall.loc[0, "mae"] == pytest.approx(2.0)


def test_population_metrics_and_topk_include_pre_deadline_candidates():
    predictions = pd.DataFrame(
        [
            _prediction(
                "2023-24:1",
                "p1",
                8,
                5.0,
                baseline="B5_EB_POINTS_PER90",
                pre_deadline_population=PRE_DEADLINE_HISTORY_ACTIVE,
            ),
            _prediction(
                "2023-24:1",
                "p2",
                1,
                2.0,
                baseline="B5_EB_POINTS_PER90",
                pre_deadline_population=COLD_START_NO_HISTORY,
            ),
            _prediction(
                "2023-24:1",
                "p3",
                3,
                4.0,
                baseline="B5_EB_POINTS_PER90",
                pre_deadline_population=PRE_DEADLINE_HISTORY_ACTIVE,
            ),
        ]
    )
    scored = aggregate_player_gameweek(score_predictions(predictions))
    tables = metric_tables(
        scored,
        scored,
        reference_baseline="B5_EB_POINTS_PER90",
        bootstrap_samples=5,
        bootstrap_seed=1,
    )
    coverage = population_coverage(scored)

    active_metrics = tables.by_population.loc[
        tables.by_population["population"] == PRE_DEADLINE_HISTORY_ACTIVE
    ]
    active_ranking = tables.ranking_by_population.loc[
        tables.ranking_by_population["population"] == PRE_DEADLINE_HISTORY_ACTIVE
    ]
    assert active_metrics.iloc[0]["rows"] == 2
    assert active_metrics.iloc[0]["mae"] == pytest.approx(2.0)
    assert set(tables.by_population["population"]) >= {
        PRE_DEADLINE_HISTORY_ACTIVE,
        COLD_START_NO_HISTORY,
    }
    assert set(active_ranking["k"]) == {15, 30}
    assert coverage.iloc[0]["pre_deadline_history_active_rows"] == 2
    assert coverage.iloc[0]["cold_start_no_history_rows"] == 1


def test_gameweek_block_bootstrap_is_deterministic():
    frame = pd.DataFrame(
        [
            _prediction("2023-24:1", "p1", 1, 1.0, baseline="B1_GLOBAL_MEAN", gameweek=1),
            _prediction("2023-24:1", "p1", 1, 2.0, baseline="B2_POSITION_MEAN", gameweek=1),
            _prediction("2023-24:2", "p1", 5, 1.0, baseline="B1_GLOBAL_MEAN", gameweek=2),
            _prediction("2023-24:2", "p1", 5, 4.0, baseline="B2_POSITION_MEAN", gameweek=2),
        ]
    )
    scored = score_predictions(frame)

    first = block_bootstrap_differences(
        scored,
        reference_baseline="B1_GLOBAL_MEAN",
        samples=20,
        seed=11,
    )
    second = block_bootstrap_differences(
        scored,
        reference_baseline="B1_GLOBAL_MEAN",
        samples=20,
        seed=11,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["evaluated_gameweeks"] == 2


def test_runner_writes_frozen_outputs_excludes_assistant_managers_and_repeats(tmp_path):
    normalized_dir = _write_phase2_tables(tmp_path, _backtest_frame())
    reports_dir = tmp_path / "reports"

    first = run_baseline_backtest(
        seasons=["2022-23", "2023-24"],
        test_seasons=["2023-24"],
        mode="rolling",
        normalized_dir=normalized_dir,
        reports_dir=reports_dir,
        run_id="repeat_a",
        bootstrap_samples=5,
        seed=3,
    )
    second = run_baseline_backtest(
        seasons=["2022-23", "2023-24"],
        test_seasons=["2023-24"],
        mode="rolling",
        normalized_dir=normalized_dir,
        reports_dir=reports_dir,
        run_id="repeat_b",
        bootstrap_samples=5,
        seed=3,
    )

    first_predictions = pd.read_parquet(first.frozen_predictions_path)
    second_predictions = pd.read_parquet(second.frozen_predictions_path)
    first_metrics = pd.read_csv(first.metrics_paths["overall"])
    second_metrics = pd.read_csv(second.metrics_paths["overall"])
    assert "am1" not in set(first_predictions["player_uid"])
    pd.testing.assert_frame_equal(
        first_predictions.drop(columns=["fold_id"]).reset_index(drop=True),
        second_predictions.drop(columns=["fold_id"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(first_metrics, second_metrics)


def test_cli_fails_when_phase2_leakage_audit_fails(tmp_path):
    normalized_dir = tmp_path / "normalized"
    (normalized_dir / "phase2").mkdir(parents=True)
    pd.DataFrame([{"season": "2022-23", "player_uid": "p1", "fixture_key": "f1"}]).to_parquet(
        normalized_dir / "phase2" / "fact_player_fixture.parquet",
        index=False,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "backtest-baselines",
            "--seasons",
            "2022-23",
            "--test-seasons",
            "2022-23",
            "--mode",
            "rolling",
            "--normalized-dir",
            str(normalized_dir),
        ],
    )

    assert result.exit_code == 1
    assert "Phase 2 leakage audit failed" in result.output or "Missing feature table" in result.output


def test_forbidden_price_form_and_xp_fields_are_rejected(tmp_path):
    frame = _backtest_frame()
    frame["form"] = 1.2
    normalized_dir = _write_phase2_tables(tmp_path, frame)

    with pytest.raises(ValueError, match="Forbidden|Phase 2 leakage audit failed"):
        run_baseline_backtest(
            seasons=["2022-23", "2023-24"],
            test_seasons=["2023-24"],
            mode="rolling",
            normalized_dir=normalized_dir,
            reports_dir=tmp_path / "reports",
            bootstrap_samples=2,
        )


def _write_phase2_tables(tmp_path: Path, frame: pd.DataFrame) -> Path:
    normalized_dir = tmp_path / "normalized"
    phase2 = normalized_dir / "phase2"
    phase2.mkdir(parents=True)
    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "player_name",
            "fpl_position",
            "minutes",
            "case",
        }
    ]
    frame[feature_columns].to_parquet(phase2 / "features_player_fixture.parquet", index=False)
    fact_columns = [
        "season",
        "player_uid",
        "fixture_key",
        "player_name",
        "fpl_position",
        "minutes",
    ]
    frame[[*fact_columns, "entity_type"]].to_parquet(phase2 / "fact_player_fixture.parquet", index=False)
    return normalized_dir


def _backtest_frame() -> pd.DataFrame:
    rows = [
        _row("2022-23", 1, "p1", "Return", "DEF", "team_a", 1, "2022-08-01T12:00:00Z", 2, 90),
        _row("2022-23", 1, "p2", "Transfer", "MID", "team_b", 2, "2022-08-08T12:00:00Z", 6, 90),
        _row("2022-23", 2, "p1", "Return", "DEF", "team_a", 3, "2022-08-15T12:00:00Z", 4, 90),
        _row("2023-24", 1, "p1", "Return", "MID", "team_a", 1, "2023-08-01T12:00:00Z", 3, 90, case="position_change"),
        _row("2023-24", 1, "p2", "Transfer", "MID", "team_c", 2, "2023-08-01T12:00:00Z", 5, 90, case="transferred"),
        _row("2023-24", 1, "p3", "New", "FWD", "team_d", 2, "2023-08-01T12:00:00Z", 0, 0, case="new_player"),
        _row("2023-24", 1, "p4", "Return Two", "GKP", "team_e", 2, "2023-08-01T12:00:00Z", 1, 0, case="returning"),
        _row("2023-24", 2, "p1", "Return", "MID", "team_a", 3, "2023-08-08T12:00:00Z", 2, 90, prev3_points=3, prev3_minutes=90, prior_appearances=1, case="returning"),
        _row("2023-24", 2, "p1", "Return", "MID", "team_a", 4, "2023-08-09T12:00:00Z", 1, 30, prev3_points=3, prev3_minutes=90, prior_appearances=1, case="returning"),
        _row("2023-24", 2, "p2", "Transfer", "MID", "team_c", 3, "2023-08-08T12:00:00Z", 7, 90, prev3_points=5, prev3_minutes=90, prior_appearances=1, case="transferred"),
        _row("2023-24", 2, "am1", "Assistant", "AM", "team_a", 3, "2023-08-08T12:00:00Z", 1, 0, entity_type="assistant_manager"),
    ]
    return pd.DataFrame(rows)


def _row(
    season: str,
    gameweek: int,
    player_uid: str,
    name: str,
    position: str,
    team_uid: str,
    fixture_id: int,
    kickoff: str,
    points: int,
    minutes: int,
    *,
    prev3_points: int = 0,
    prev3_minutes: int = 0,
    prior_appearances: int = 0,
    entity_type: str = "player",
    case: str = "returning",
) -> dict[str, object]:
    kickoff_time = pd.Timestamp(kickoff, tz="UTC")
    return {
        "season": season,
        "gameweek": gameweek,
        "fixture_key": f"{season}:{fixture_id}",
        "player_uid": player_uid,
        "player_name": name,
        "fpl_position": position,
        "kickoff_time": kickoff_time,
        "source_available_time": kickoff_time + pd.Timedelta(hours=3),
        "information_cutoff": kickoff_time,
        "player_team_uid": team_uid,
        "opponent_team_uid": "team_x",
        "was_home": True,
        "entity_type": entity_type,
        "target_total_points": points,
        "home_flag": 1,
        "player_max_source_kickoff": pd.NaT,
        "player_max_source_available_time": pd.NaT,
        "prev_fixture_minutes": pd.NA,
        "prev3_minutes_sum": prev3_minutes,
        "prev5_minutes_sum": prev3_minutes,
        "prev3_starts_sum": int(prev3_minutes > 0),
        "prev5_starts_sum": int(prev3_minutes > 0),
        "prev3_total_points_sum": prev3_points,
        "prev5_total_points_sum": prev3_points,
        "prev3_goals_scored_sum": 0,
        "prev5_goals_scored_sum": 0,
        "prev3_assists_sum": 0,
        "prev5_assists_sum": 0,
        "season_to_date_minutes": prev3_minutes,
        "season_to_date_appearances": prior_appearances,
        "prior_source_season": "2022-23" if season == "2023-24" and player_uid in {"p1", "p2"} else pd.NA,
        "prior_season_minutes": 180 if season == "2023-24" and player_uid in {"p1", "p2"} else pd.NA,
        "prior_season_appearances": 2 if season == "2023-24" and player_uid in {"p1", "p2"} else pd.NA,
        "team_max_source_kickoff": pd.NaT,
        "team_max_source_available_time": pd.NaT,
        "team_prev3_goals_for_sum": 0,
        "opponent_max_source_kickoff": pd.NaT,
        "opponent_max_source_available_time": pd.NaT,
        "opponent_prev3_goals_against_sum": 0,
        "feature_registry_version": "registry.json",
        "minutes": minutes,
        "case": case,
    }


def _prediction(
    fixture_key: str,
    player_uid: str,
    points: int,
    prediction: float,
    *,
    baseline: str = "B1_GLOBAL_MEAN",
    gameweek: int = 2,
    pre_deadline_population: str = PRE_DEADLINE_HISTORY_ACTIVE,
) -> dict[str, object]:
    return {
        "season": "2023-24",
        "gameweek": gameweek,
        "fixture_key": fixture_key,
        "player_uid": player_uid,
        "player_name": player_uid,
        "fpl_position": "MID",
        "player_team_uid": "team_a",
        "opponent_team_uid": "team_b",
        "information_cutoff": "2023-08-08T12:00:00Z",
        "source_available_time": "2023-08-08T15:00:00Z",
        "minutes": 90,
        "target_total_points": points,
        "pre_deadline_population": pre_deadline_population,
        "baseline": baseline,
        "prediction": prediction,
    }


def test_manifest_records_provenance(tmp_path):
    normalized_dir = _write_phase2_tables(tmp_path, _backtest_frame())

    result = run_baseline_backtest(
        seasons=["2022-23", "2023-24"],
        test_seasons=["2023-24"],
        mode="gw1",
        normalized_dir=normalized_dir,
        reports_dir=tmp_path / "reports",
        run_id="manifest_check",
        bootstrap_samples=5,
        seed=3,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["leakage_audit"] == "passed"
    assert manifest["git"]["commit"]
    assert manifest["source_data"]["fact_hash"]
    assert manifest["baseline_parameters"]["points_per_90_prior_matches"] == 5
