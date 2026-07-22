from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from typer.testing import CliRunner

from fpl_forecast.cli import app
from fpl_forecast.team_model.config import load_team_model_config
from fpl_forecast.team_model.data import (
    assert_frozen_predictions_target_free,
    build_team_fixture_table,
    load_current_fixture_frame,
    validate_team_fixture_table,
)
from fpl_forecast.team_model.metrics import team_metric_tables
from fpl_forecast.team_model.models import ModelFit, fit_t0, fit_t1, fit_t2, predict_with_fit
from fpl_forecast.team_model.probabilities import (
    add_probability_columns,
    outcome_probabilities,
    poisson_goal_distribution,
    validate_probability_frame,
)
from fpl_forecast.team_model.runner import build_team_folds, forecast_team_fixtures, run_team_backtest, split_team_fold


def test_team_fixture_table_validation_and_keys():
    table = build_team_fixture_table(_fixtures(), _team_map())

    validate_team_fixture_table(table)

    assert table["stable_fixture_uid"].is_unique
    assert set(table["home_goals"]) == {1, 2, 3, 0}


def test_team_fixture_validation_rejects_duplicate_same_team_and_bad_goals():
    table = build_team_fixture_table(_fixtures(), _team_map())
    duplicate = pd.concat([table, table.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_team_fixture_table(duplicate)

    same_team = table.copy()
    same_team.loc[0, "away_team_uid"] = same_team.loc[0, "home_team_uid"]
    with pytest.raises(ValueError, match="home team equal"):
        validate_team_fixture_table(same_team)

    bad_goal = table.copy()
    bad_goal.loc[0, "home_goals"] = -1
    with pytest.raises(ValueError, match="goals"):
        validate_team_fixture_table(bad_goal)


def test_fold_training_requires_strict_availability_and_same_block_exclusion():
    table = build_team_fixture_table(_fixtures(include_equal_cutoff=True), _team_map())
    folds = build_team_folds(table, test_seasons=["2023-24"], mode="rolling")
    fold = next(item for item in folds if item.fold_id == "2023-24_GW02")

    train, test = split_team_fold(table, fold, mode="rolling")

    assert train["source_available_time"].lt(fold.information_cutoff).all()
    assert "2023-24:4" not in set(train["stable_fixture_uid"])
    assert set(test["stable_fixture_uid"]) == {"2023-24:4", "2023-24:5"}


def test_gw1_uses_no_target_season_results_and_double_gameweek_separate():
    table = build_team_fixture_table(_fixtures(), _team_map())
    fold = build_team_folds(table, test_seasons=["2023-24"], mode="gw1")[0]

    train, test = split_team_fold(table, fold, mode="gw1")

    assert set(train["season"]) == {"2022-23"}
    assert len(test) == 1
    assert table.loc[
        table["season"].eq("2023-24") & table["gameweek"].eq(2),
        "stable_fixture_uid",
    ].nunique() == 2


def test_reentered_team_history_survives_source_id_change():
    fixtures = pd.DataFrame(
        [
            _fixture("2022-23", 1, 1, "team_reentry", "team_stayer", "2022-08-01T12:00:00Z", 1, 0),
            _fixture("2022-23", 2, 2, "team_stayer", "team_reentry", "2022-08-08T12:00:00Z", 2, 1),
            _fixture("2024-25", 1, 3, "team_reentry", "team_stayer", "2024-08-17T12:00:00Z", 0, 1),
            _fixture("2024-25", 2, 4, "team_stayer", "team_reentry", "2024-08-24T12:00:00Z", 1, 1),
        ]
    )
    team_map = pd.DataFrame(
        [
            {
                "season": "2022-23",
                "team_uid": "team_reentry",
                "source_team_id": 10,
                "source_team_name": "Reentry",
            },
            {
                "season": "2024-25",
                "team_uid": "team_reentry",
                "source_team_id": 11,
                "source_team_name": "Reentry",
            },
            {
                "season": "2022-23",
                "team_uid": "team_stayer",
                "source_team_id": 1,
                "source_team_name": "Stayer",
            },
            {
                "season": "2024-25",
                "team_uid": "team_stayer",
                "source_team_id": 1,
                "source_team_name": "Stayer",
            },
        ]
    )
    table = build_team_fixture_table(fixtures, team_map)
    fold = build_team_folds(table, test_seasons=["2024-25"], mode="rolling")[1]

    train, _ = split_team_fold(table, fold, mode="rolling")

    assert fold.fold_id == "2024-25_GW02"
    assert set(table.loc[table["home_team_uid"].eq("team_reentry"), "source_home_team_id"]) == {10, 11}
    assert len(train.loc[train["home_team_uid"].eq("team_reentry") | train["away_team_uid"].eq("team_reentry")]) == 3


def test_t0_and_t1_hand_calculated_rates():
    config = load_team_model_config()
    table = build_team_fixture_table(_fixtures(), _team_map())
    train = table.loc[table["season"].eq("2022-23")]
    t0 = fit_t0(train, config=config)
    t1 = fit_t1(train, cutoff=pd.Timestamp("2023-08-10T12:00:00Z"), config=config)

    assert t0.parameters["home_mean"] == pytest.approx((1 + 2) / 2)
    assert t0.parameters["away_mean"] == pytest.approx((0 + 1) / 2)
    team_a = t1.parameters["team_rates"]["team_a"]
    expected = (3 + config.t1_shrink_matches * 1.5) / (2 + config.t1_shrink_matches)
    assert team_a["home_attack"] == pytest.approx(expected)


def test_t2_sign_convention_and_unseen_fallback_are_finite():
    config = load_team_model_config()
    base = pd.DataFrame(
        [
            _prediction_row("team_a", "team_b", 1.0, 1.0),
            _prediction_row("team_a", "team_new", 1.0, 1.0),
        ]
    )
    fit = ModelFit(
        "T2_REGULARIZED_ATTACK_DEFENCE",
        {
            "intercept": 0.0,
            "home_advantage": 0.1,
            "attack": {"team_a": 0.0, "team_b": 0.4},
            "defence": {"team_a": 0.0, "team_b": 0.5},
        },
        pd.DataFrame(
            [
                {"team_uid": "team_a", "training_fixture_count": 10},
                {"team_uid": "team_b", "training_fixture_count": 10},
            ]
        ),
    )

    pred = predict_with_fit(fit, base, config=config)

    assert pred.loc[0, "expected_home_goals"] > pred.loc[0, "expected_away_goals"]
    assert pred.loc[1, "away_unseen_or_promoted_flag"]
    assert pred[["expected_home_goals", "expected_away_goals"]].ge(0).all().all()


def test_t2_penalized_poisson_fit_has_expected_behavior_and_is_deterministic():
    config = load_team_model_config()
    train = build_team_fixture_table(_attack_defence_fixtures(), _attack_defence_team_map())

    first = fit_t2(train, cutoff=pd.Timestamp("2024-01-01T00:00:00Z"), config=config)
    second = fit_t2(train, cutoff=pd.Timestamp("2024-01-01T00:00:00Z"), config=config)

    assert first.converged
    assert first.parameters["solver"] == "deterministic_newton_line_search"
    assert first.parameters["max_abs_gradient"] < config.t2_convergence_tolerance
    assert first.parameters["iterations"] > 0
    assert sum(first.parameters["attack"].values()) == pytest.approx(0.0)
    assert sum(first.parameters["defence"].values()) == pytest.approx(0.0)
    assert first.parameters["home_advantage"] > 0
    assert first.parameters["attack"]["team_strong"] > first.parameters["attack"]["team_weak"]
    assert first.parameters["defence"]["team_strong"] < first.parameters["defence"]["team_weak"]
    assert first.parameters["intercept"] == pytest.approx(second.parameters["intercept"])
    assert first.parameters["home_advantage"] == pytest.approx(second.parameters["home_advantage"])
    assert first.parameters["attack"] == pytest.approx(second.parameters["attack"])
    assert first.parameters["defence"] == pytest.approx(second.parameters["defence"])
    pd.testing.assert_frame_equal(first.ratings, second.ratings)


def test_t2_regularization_shrinks_team_effects_and_nonconvergence_fails():
    config = load_team_model_config()
    train = build_team_fixture_table(_attack_defence_fixtures(), _attack_defence_team_map())
    low_ridge = replace(config, t2_ridge_penalty=0.1)
    high_ridge = replace(config, t2_ridge_penalty=100.0)

    loose = fit_t2(train, cutoff=pd.Timestamp("2024-01-01T00:00:00Z"), config=low_ridge)
    tight = fit_t2(train, cutoff=pd.Timestamp("2024-01-01T00:00:00Z"), config=high_ridge)

    assert abs(tight.parameters["attack"]["team_strong"]) < abs(loose.parameters["attack"]["team_strong"])
    assert abs(tight.parameters["defence"]["team_weak"]) < abs(loose.parameters["defence"]["team_weak"])
    with pytest.raises(RuntimeError, match="failed to converge"):
        fit_t2(
            train,
            cutoff=pd.Timestamp("2024-01-01T00:00:00Z"),
            config=replace(config, t2_max_iterations=0, t2_convergence_tolerance=0.0),
        )


def test_probability_outputs_are_valid_and_clean_sheet_matches_zero_goal_probability():
    config = load_team_model_config()
    pred = add_probability_columns(pd.DataFrame([_prediction_row("team_a", "team_b", 1.4, 0.8)]), config)

    validate_probability_frame(pred, config)

    assert pred.iloc[0]["home_clean_sheet_probability"] == pytest.approx(
        pred.iloc[0]["away_goals_prob_0"]
    )
    assert pred.iloc[0]["away_clean_sheet_probability"] == pytest.approx(
        pred.iloc[0]["home_goals_prob_0"]
    )
    assert sum(poisson_goal_distribution(1.4, max_goals=8)) == pytest.approx(1.0)
    assert sum(outcome_probabilities(1.4, 0.8, max_goals=20)) == pytest.approx(1.0)


def test_target_outcomes_do_not_appear_in_frozen_predictions_and_metrics_work(tmp_path):
    normalized_dir = _write_phase2_team_tables(tmp_path)
    result = run_team_backtest(
        seasons=["2022-23", "2023-24"],
        test_seasons=["2023-24"],
        mode="rolling",
        normalized_dir=normalized_dir,
        reports_dir=tmp_path / "reports",
        run_id="team_repeat_a",
    )
    again = run_team_backtest(
        seasons=["2022-23", "2023-24"],
        test_seasons=["2023-24"],
        mode="rolling",
        normalized_dir=normalized_dir,
        reports_dir=tmp_path / "reports",
        run_id="team_repeat_b",
    )

    frozen = pd.read_parquet(result.frozen_predictions_path)
    frozen_again = pd.read_parquet(again.frozen_predictions_path)
    assert_frozen_predictions_target_free(frozen)
    assert "home_goals" not in frozen.columns
    pd.testing.assert_frame_equal(
        frozen.drop(columns=["fold_id"]).reset_index(drop=True),
        frozen_again.drop(columns=["fold_id"]).reset_index(drop=True),
    )
    scored = pd.read_parquet(result.scored_predictions_path)
    tables = team_metric_tables(scored, load_team_model_config())
    assert set(tables.expected_goals["model_name"]) == {
        "T0_LEAGUE_HOME_AWAY",
        "T1_SHRUNK_ROLLING_TEAM_RATE",
        "T2_REGULARIZED_ATTACK_DEFENCE",
        "T3_DIXON_COLES",
    }


def test_cli_fails_when_team_model_leakage_precondition_fails(tmp_path):
    normalized_dir = tmp_path / "normalized"
    (normalized_dir / "phase2").mkdir(parents=True)
    pd.DataFrame([{"season": "2022-23"}]).to_parquet(
        normalized_dir / "phase2" / "fact_player_fixture.parquet",
        index=False,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "backtest-team-model",
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
    assert "Team-model backtest failed" in result.output


def test_current_fixture_inference_blocks_missing_and_mismatched_season(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_current_fixture_frame(
            season="2026-27",
            as_of="2026-07-22T00:00:00Z",
            normalized_dir=tmp_path / "normalized",
        )

    normalized_dir = _write_current_tables(tmp_path, season="2026-27", kickoff_year=2025)
    with pytest.raises(ValueError, match="season mismatch"):
        load_current_fixture_frame(
            season="2026-27",
            as_of="2026-07-22T00:00:00Z",
            normalized_dir=normalized_dir,
        )


def test_promoted_current_team_can_map_to_neutral_forecast_fallback(tmp_path):
    normalized_dir = _write_current_tables(tmp_path, season="2025-26", kickoff_year=2025)
    phase2 = normalized_dir / "phase2"
    _fixtures().to_parquet(phase2 / "dim_fixture.parquet", index=False)
    _team_map().to_parquet(phase2 / "team_season_map.parquet", index=False)
    pd.DataFrame(
        [
            {"team_uid": "team_a", "canonical_name": "Team A", "normalized_short_name": "team a"},
            {"team_uid": "team_promoted", "canonical_name": "Promoted FC", "normalized_short_name": "promoted fc"},
        ]
    ).to_parquet(phase2 / "dim_team.parquet", index=False)
    current_teams = pd.read_parquet(normalized_dir / "2025-26" / "current_teams.parquet")
    current_teams.loc[current_teams["team_id"].eq(2), ["team_name", "short_name"]] = ["Promoted FC", "PFC"]
    current_teams.to_parquet(normalized_dir / "2025-26" / "current_teams.parquet", index=False)

    path = forecast_team_fixtures(
        season="2025-26",
        gameweek=1,
        as_of="2025-08-01T00:00:00Z",
        seasons=["2022-23", "2023-24"],
        normalized_dir=normalized_dir,
        reports_dir=tmp_path / "reports",
        run_id="promoted_fallback",
    )
    forecast = pd.read_parquet(path)

    assert set(forecast["away_team_uid"]) == {"team_promoted"}
    assert forecast["away_unseen_or_promoted_flag"].all()
    assert forecast.loc[forecast["model_name"].eq("T2_REGULARIZED_ATTACK_DEFENCE"), "expected_away_goals"].notna().all()


def test_current_price_tenths_preserved_with_provenance_and_unused_by_team_model(tmp_path):
    normalized_dir = _write_current_tables(tmp_path, season="2025-26", kickoff_year=2025)
    players = pd.read_parquet(normalized_dir / "2025-26" / "current_players.parquet")

    assert players.loc[0, "price_tenths"] == 75
    assert players.loc[0, "source_version"] == "current-sha"
    team_columns = set(build_team_fixture_table(_fixtures(), _team_map()).columns)
    assert "price_tenths" not in team_columns


def test_final_retrospective_scores_do_not_enter_future_fixture_forecast(tmp_path):
    normalized_dir = _write_current_tables(tmp_path, season="2025-26", kickoff_year=2025, with_scores=True)
    future = load_current_fixture_frame(
        season="2025-26",
        as_of="2025-08-01T00:00:00Z",
        normalized_dir=normalized_dir,
    )

    assert "home_goals" not in future.columns
    assert "away_goals" not in future.columns
    assert future["kickoff_time"].gt(pd.Timestamp("2025-08-01T00:00:00Z")).all()


def _fixtures(include_equal_cutoff: bool = False) -> pd.DataFrame:
    rows = [
        _fixture("2022-23", 1, 1, "team_a", "team_b", "2022-08-01T12:00:00Z", 1, 0),
        _fixture("2022-23", 2, 2, "team_a", "team_c", "2022-08-08T12:00:00Z", 2, 1),
        _fixture("2023-24", 1, 3, "team_b", "team_a", "2023-08-01T12:00:00Z", 3, 2),
        _fixture("2023-24", 2, 4, "team_c", "team_a", "2023-08-08T12:00:00Z", 0, 0),
        _fixture("2023-24", 2, 5, "team_a", "team_b", "2023-08-09T12:00:00Z", 1, 1),
    ]
    if include_equal_cutoff:
        rows[2]["source_available_time"] = "2023-08-08T12:00:00Z"
    return pd.DataFrame(rows)


def _fixture(
    season: str,
    gameweek: int,
    fixture_id: int,
    home: str,
    away: str,
    kickoff: str,
    home_goals: int,
    away_goals: int,
) -> dict[str, object]:
    return {
        "season": season,
        "source_fixture_id": fixture_id,
        "fixture_key": f"{season}:{fixture_id}",
        "gameweek": gameweek,
        "home_team_uid": home,
        "away_team_uid": away,
        "home_team_name": home.replace("team_", "Team ").title(),
        "away_team_name": away.replace("team_", "Team ").title(),
        "kickoff_time": kickoff,
        "source_available_time": (pd.Timestamp(kickoff) + pd.Timedelta(hours=3)).isoformat(),
        "source_available_method": "kickoff_plus_3h",
        "finished": True,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "source_version": "abc123",
        "raw_snapshot_path": "/tmp/source.csv",
    }


def _team_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": season, "team_uid": team, "source_team_id": idx, "source_team_name": team}
            for season in ("2022-23", "2023-24")
            for idx, team in enumerate(["team_a", "team_b", "team_c"], start=1)
        ]
    )


def _prediction_row(home: str, away: str, home_lambda: float, away_lambda: float) -> dict[str, object]:
    return {
        "season": "2023-24",
        "gameweek": 1,
        "stable_fixture_uid": "2023-24:1",
        "source_fixture_id": 1,
        "home_team_uid": home,
        "away_team_uid": away,
        "source_home_team_id": 1,
        "source_away_team_id": 2,
        "home_team_name": home,
        "away_team_name": away,
        "kickoff_time": "2023-08-01T12:00:00Z",
        "information_cutoff": "2023-08-01T12:00:00Z",
        "source_available_time": "2023-08-01T15:00:00Z",
        "source_available_method": "kickoff_plus_3h",
        "finished": True,
        "result_valid": True,
        "source_version": "abc123",
        "raw_snapshot_path": "/tmp/source.csv",
        "model_name": "T0_LEAGUE_HOME_AWAY",
        "expected_home_goals": home_lambda,
        "expected_away_goals": away_lambda,
        "home_unseen_or_promoted_flag": False,
        "away_unseen_or_promoted_flag": False,
        "home_low_history_flag": False,
        "away_low_history_flag": False,
        "fit_status": "ok",
        "fit_converged": True,
    }


def _write_phase2_team_tables(tmp_path) -> str:
    normalized_dir = tmp_path / "normalized"
    phase2 = normalized_dir / "phase2"
    phase2.mkdir(parents=True)
    _fixtures().to_parquet(phase2 / "dim_fixture.parquet", index=False)
    _team_map().to_parquet(phase2 / "team_season_map.parquet", index=False)
    feature = pd.DataFrame(
        {
            "season": ["2022-23"],
            "player_uid": ["p1"],
            "fixture_key": ["2022-23:1"],
            "gameweek": [1],
            "kickoff_time": ["2022-08-01T12:00:00Z"],
            "information_cutoff": ["2022-08-01T12:00:00Z"],
            "player_max_source_available_time": [pd.NaT],
            "team_max_source_available_time": [pd.NaT],
            "opponent_max_source_available_time": [pd.NaT],
            "feature_registry_version": ["registry.json"],
        }
    )
    fact = feature[["season", "player_uid", "fixture_key"]].copy()
    feature.to_parquet(phase2 / "features_player_fixture.parquet", index=False)
    fact.to_parquet(phase2 / "fact_player_fixture.parquet", index=False)
    return normalized_dir


def _write_current_tables(tmp_path, *, season: str, kickoff_year: int, with_scores: bool = False) -> str:
    normalized_dir = tmp_path / "normalized"
    season_dir = normalized_dir / season
    phase2 = normalized_dir / "phase2"
    season_dir.mkdir(parents=True)
    phase2.mkdir(parents=True)
    pd.DataFrame(
        [
            {"team_uid": "team_a", "canonical_name": "Team A", "normalized_short_name": "team a"},
            {"team_uid": "team_b", "canonical_name": "Team B", "normalized_short_name": "team b"},
        ]
    ).to_parquet(phase2 / "dim_team.parquet", index=False)
    pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "fixture_code": 100,
                "gameweek": 1,
                "home_team_id": 1,
                "away_team_id": 2,
                "kickoff_time": f"{kickoff_year}-08-15T19:00:00Z",
                "finished": with_scores,
                "started": with_scores,
                "team_h_score": 2 if with_scores else pd.NA,
                "team_a_score": 1 if with_scores else pd.NA,
                "source": "fpl_api",
                "source_version": "fixture-sha",
                "retrieved_at": "2026-07-22T00:00:00Z",
                "season": season,
                "raw_snapshot_path": "/tmp/fixtures.json",
            }
        ]
    ).to_parquet(season_dir / "current_fixtures.parquet", index=False)
    pd.DataFrame(
        [
            {
                "team_id": 1,
                "team_code": 1,
                "team_name": "Team A",
                "short_name": "TA",
                "strength": 3,
                "source": "fpl_api",
                "source_version": "current-sha",
                "retrieved_at": "2026-07-22T00:00:00Z",
                "season": season,
                "raw_snapshot_path": "/tmp/bootstrap.json",
            },
            {
                "team_id": 2,
                "team_code": 2,
                "team_name": "Team B",
                "short_name": "TB",
                "strength": 3,
                "source": "fpl_api",
                "source_version": "current-sha",
                "retrieved_at": "2026-07-22T00:00:00Z",
                "season": season,
                "raw_snapshot_path": "/tmp/bootstrap.json",
            },
        ]
    ).to_parquet(season_dir / "current_teams.parquet", index=False)
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_code": 11,
                "first_name": "Price",
                "second_name": "Check",
                "web_name": "Check",
                "team_id": 1,
                "position_id": 3,
                "price_tenths": 75,
                "status": "a",
                "minutes": 0,
                "total_points": 0,
                "position": "MID",
                "source": "fpl_api",
                "source_version": "current-sha",
                "retrieved_at": "2026-07-22T00:00:00Z",
                "season": season,
                "raw_snapshot_path": "/tmp/bootstrap.json",
            }
        ]
    ).to_parquet(season_dir / "current_players.parquet", index=False)
    return normalized_dir


def _attack_defence_fixtures() -> pd.DataFrame:
    rows = []
    fixture_id = 1
    fixtures = [
        ("team_strong", "team_weak", 4, 0),
        ("team_average", "team_strong", 0, 2),
        ("team_weak", "team_average", 1, 2),
        ("team_strong", "team_average", 3, 1),
        ("team_weak", "team_strong", 0, 2),
        ("team_average", "team_weak", 2, 1),
        ("team_strong", "team_weak", 3, 1),
        ("team_average", "team_strong", 1, 2),
        ("team_weak", "team_average", 0, 2),
    ]
    for idx, (home, away, home_goals, away_goals) in enumerate(fixtures, start=1):
        rows.append(
            _fixture(
                "2023-24",
                idx,
                fixture_id,
                home,
                away,
                f"2023-08-{idx:02d}T12:00:00Z",
                home_goals,
                away_goals,
            )
        )
        fixture_id += 1
    return pd.DataFrame(rows)


def _attack_defence_team_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": "2023-24", "team_uid": team, "source_team_id": idx, "source_team_name": team}
            for idx, team in enumerate(["team_average", "team_strong", "team_weak"], start=1)
        ]
    )
