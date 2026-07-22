from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fpl_forecast.team_model.config import load_team_model_config
from fpl_forecast.team_model.data import (
    assert_frozen_predictions_target_free,
    build_team_fixture_table,
)
from fpl_forecast.team_model.models import fit_predict_models, fit_t3
from fpl_forecast.team_model.probabilities import (
    add_probability_columns,
    dixon_coles_joint_distribution,
    dixon_coles_outcome_probabilities,
    dixon_coles_tau,
    outcome_probabilities,
    poisson_goal_distribution,
)
from fpl_forecast.team_model.runner import TeamFold, split_team_fold


def test_tau_cases_and_invalid_boundaries():
    assert dixon_coles_tau(0, 0, 1.2, 0.8, -0.1) == pytest.approx(1.096)
    assert dixon_coles_tau(0, 1, 1.2, 0.8, -0.1) == pytest.approx(0.88)
    assert dixon_coles_tau(1, 0, 1.2, 0.8, -0.1) == pytest.approx(0.92)
    assert dixon_coles_tau(1, 1, 1.2, 0.8, -0.1) == pytest.approx(1.1)
    assert dixon_coles_tau(2, 1, 1.2, 0.8, -0.1) == pytest.approx(1.0)
    assert dixon_coles_tau(0, 0, 1.2, 0.8, 0.0) == pytest.approx(1.0)
    assert dixon_coles_tau(1, 0, 1.2, 0.8, 0.1) == pytest.approx(1.08)
    with pytest.raises(ValueError, match="Invalid Dixon-Coles tau"):
        dixon_coles_tau(0, 0, 3.0, 3.0, 0.2)
    with pytest.raises(ValueError, match="Invalid Dixon-Coles tau"):
        dixon_coles_tau(0, 1, 3.0, 1.0, -0.5)


def test_joint_distribution_rho_zero_matches_independent_and_preserves_marginals():
    home_lambda = 1.4
    away_lambda = 0.9
    independent = np.outer(
        poisson_goal_distribution(home_lambda, max_goals=8),
        poisson_goal_distribution(away_lambda, max_goals=8),
    )
    dc_zero = dixon_coles_joint_distribution(home_lambda, away_lambda, 0.0, max_goals=8)

    assert dc_zero == pytest.approx(independent)
    assert dc_zero.sum() == pytest.approx(1.0)
    assert dc_zero.sum(axis=1) == pytest.approx(poisson_goal_distribution(home_lambda, max_goals=8))
    assert dc_zero.sum(axis=0) == pytest.approx(poisson_goal_distribution(away_lambda, max_goals=8))
    assert dc_zero[:, 0].sum() == pytest.approx(poisson_goal_distribution(away_lambda, max_goals=8)[0])
    assert dc_zero[-1, :].sum() + dc_zero[:-1, -1].sum() > 0


def test_joint_distribution_only_low_score_cells_change_and_outcomes_sum():
    home_lambda = 1.3
    away_lambda = 1.1
    rho = -0.08
    independent = np.outer(
        poisson_goal_distribution(home_lambda, max_goals=8),
        poisson_goal_distribution(away_lambda, max_goals=8),
    )
    dc = dixon_coles_joint_distribution(home_lambda, away_lambda, rho, max_goals=8)
    changed = {
        tuple(idx)
        for idx in np.argwhere(~np.isclose(dc, independent, atol=1e-12))
    }

    assert changed == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert dc.min() >= 0
    assert dc.max() <= 1
    assert dc.sum() == pytest.approx(1.0)
    assert dc.sum(axis=1) == pytest.approx(poisson_goal_distribution(home_lambda, max_goals=8))
    assert dc.sum(axis=0) == pytest.approx(poisson_goal_distribution(away_lambda, max_goals=8))
    assert dixon_coles_outcome_probabilities(home_lambda, away_lambda, rho, max_goals=20) == pytest.approx(
        tuple(
            value
            for value in dixon_coles_outcome_probabilities(home_lambda, away_lambda, rho, max_goals=20)
        )
    )
    assert sum(dixon_coles_outcome_probabilities(home_lambda, away_lambda, rho, max_goals=20)) == pytest.approx(1.0)
    assert dixon_coles_outcome_probabilities(home_lambda, away_lambda, 0.0, max_goals=20) == pytest.approx(
        outcome_probabilities(home_lambda, away_lambda, max_goals=20)
    )


def test_t3_fit_negative_rho_for_excess_low_draws_and_zero_without_low_scores():
    config = load_team_model_config()
    excess = build_team_fixture_table(_low_draw_fixture_rows(), _team_map())
    independent = build_team_fixture_table(_no_low_score_fixture_rows(), _team_map())

    excess_fit = fit_t3(excess, cutoff=pd.Timestamp("2024-01-01T00:00:00Z"), config=config)
    independent_fit = fit_t3(independent, cutoff=pd.Timestamp("2024-01-01T00:00:00Z"), config=config)

    assert excess_fit.parameters["rho"] < 0
    assert abs(independent_fit.parameters["rho"]) < 1e-6
    assert excess_fit.parameters["minimum_tau"] > 0
    assert excess_fit.parameters["max_abs_gradient"] < config.t3_convergence_tolerance
    assert sum(excess_fit.parameters["attack"].values()) == pytest.approx(0.0)
    assert sum(excess_fit.parameters["defence"].values()) == pytest.approx(0.0)


def test_t3_deterministic_nonconvergence_and_invalid_config_fail():
    config = load_team_model_config()
    table = build_team_fixture_table(_low_draw_fixture_rows(), _team_map())

    first = fit_t3(table, cutoff=pd.Timestamp("2024-01-01T00:00:00Z"), config=config)
    second = fit_t3(table, cutoff=pd.Timestamp("2024-01-01T00:00:00Z"), config=config)

    assert first.parameters["rho"] == pytest.approx(second.parameters["rho"])
    assert first.parameters["attack"] == pytest.approx(second.parameters["attack"])
    with pytest.raises(RuntimeError, match="failed to converge"):
        fit_t3(
            table,
            cutoff=pd.Timestamp("2024-01-01T00:00:00Z"),
            config=replace(config, t3_max_iterations=0, t3_convergence_tolerance=0.0),
        )
    with pytest.raises(ValueError, match="lower rho bound"):
        fit_t3(
            table,
            cutoff=pd.Timestamp("2024-01-01T00:00:00Z"),
            config=replace(config, t3_rho_lower_bound=-0.3),
        )
    with pytest.raises(ValueError, match="upper rho bound"):
        fit_t3(
            table,
            cutoff=pd.Timestamp("2024-01-01T00:00:00Z"),
            config=replace(config, t3_rho_upper_bound=0.05),
        )


def test_t3_backtest_rows_are_target_free_and_do_not_change_t2_outputs():
    config = load_team_model_config()
    table = build_team_fixture_table(_mixed_fixture_rows(), _team_map())
    train = table.loc[table["gameweek"].le(6)]
    test = table.loc[table["gameweek"].eq(7)].drop(columns=["home_goals", "away_goals"])

    all_predictions, _, diagnostics = fit_predict_models(
        train,
        test,
        cutoff=pd.Timestamp("2023-08-07T12:00:00Z"),
        config=config,
    )
    t2_predictions, _, _ = fit_predict_models(
        train,
        test,
        cutoff=pd.Timestamp("2023-08-07T12:00:00Z"),
        config=config,
        models=["T2_REGULARIZED_ATTACK_DEFENCE"],
    )
    frozen = add_probability_columns(all_predictions, config)

    assert set(all_predictions["model_name"]) == {
        "T0_LEAGUE_HOME_AWAY",
        "T1_SHRUNK_ROLLING_TEAM_RATE",
        "T2_REGULARIZED_ATTACK_DEFENCE",
        "T3_DIXON_COLES",
    }
    assert len(all_predictions) == len(test) * 4
    assert_frozen_predictions_target_free(frozen)
    pd.testing.assert_series_equal(
        all_predictions.loc[
            all_predictions["model_name"].eq("T2_REGULARIZED_ATTACK_DEFENCE"),
            "expected_home_goals",
        ].reset_index(drop=True),
        t2_predictions["expected_home_goals"].reset_index(drop=True),
        check_names=False,
    )
    t3_diagnostic = next(item for item in diagnostics if item["model_name"] == "T3_DIXON_COLES")
    assert t3_diagnostic["rho"] is not None
    assert t3_diagnostic["minimum_tau"] > 0


def test_same_deadline_results_cannot_change_t3_rho():
    config = load_team_model_config()
    base = build_team_fixture_table(_mixed_fixture_rows(), _team_map())
    mutated = base.copy()
    mutated.loc[mutated["gameweek"].eq(7), ["home_goals", "away_goals"]] = [0, 0]
    fold = TeamFold(
        fold_id="2023-24_GW07",
        season="2023-24",
        gameweek=7,
        information_cutoff=pd.Timestamp("2023-08-07T12:00:00Z"),
        train_fixtures=6,
        test_fixtures=1,
    )

    base_train, _ = split_team_fold(base, fold, mode="rolling")
    mutated_train, _ = split_team_fold(mutated, fold, mode="rolling")
    base_fit = fit_t3(base_train, cutoff=fold.information_cutoff, config=config)
    mutated_fit = fit_t3(mutated_train, cutoff=fold.information_cutoff, config=config)

    assert base_train["source_available_time"].lt(fold.information_cutoff).all()
    assert not set(base_train["stable_fixture_uid"]) & set(base.loc[base["gameweek"].eq(7), "stable_fixture_uid"])
    assert base_fit.parameters["rho"] == pytest.approx(mutated_fit.parameters["rho"])
    assert base_fit.parameters["attack"] == pytest.approx(mutated_fit.parameters["attack"])


def test_gw1_training_excludes_target_season_for_t3():
    config = load_team_model_config()
    table = build_team_fixture_table(_two_season_fixture_rows(), _two_season_team_map())
    fold = TeamFold(
        fold_id="2024-25_GW01",
        season="2024-25",
        gameweek=1,
        information_cutoff=pd.Timestamp("2024-08-10T12:00:00Z"),
        train_fixtures=3,
        test_fixtures=1,
    )

    train, test = split_team_fold(table, fold, mode="gw1")
    predictions, _, _ = fit_predict_models(
        train,
        test.drop(columns=["home_goals", "away_goals"]),
        cutoff=fold.information_cutoff,
        config=config,
    )

    assert set(train["season"]) == {"2023-24"}
    assert len(predictions.loc[predictions["model_name"].eq("T3_DIXON_COLES")]) == len(test)


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
        "home_team_name": home,
        "away_team_name": away,
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
            {"season": "2023-24", "team_uid": team, "source_team_id": idx, "source_team_name": team}
            for idx, team in enumerate(["team_a", "team_b", "team_c"], start=1)
        ]
    )


def _two_season_team_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": season, "team_uid": team, "source_team_id": idx, "source_team_name": team}
            for season in ("2023-24", "2024-25")
            for idx, team in enumerate(["team_a", "team_b", "team_c"], start=1)
        ]
    )


def _low_draw_fixture_rows() -> pd.DataFrame:
    rows = []
    fixtures = [
        ("team_a", "team_b", 0, 0),
        ("team_b", "team_c", 1, 1),
        ("team_c", "team_a", 0, 0),
        ("team_a", "team_c", 1, 1),
        ("team_b", "team_a", 0, 0),
        ("team_c", "team_b", 1, 1),
    ]
    for idx, (home, away, hg, ag) in enumerate(fixtures, start=1):
        rows.append(_fixture("2023-24", idx, idx, home, away, f"2023-08-{idx:02d}T12:00:00Z", hg, ag))
    return pd.DataFrame(rows)


def _no_low_score_fixture_rows() -> pd.DataFrame:
    rows = []
    fixtures = [
        ("team_a", "team_b", 2, 2),
        ("team_b", "team_c", 3, 2),
        ("team_c", "team_a", 2, 3),
        ("team_a", "team_c", 4, 2),
        ("team_b", "team_a", 2, 4),
        ("team_c", "team_b", 3, 3),
    ]
    for idx, (home, away, hg, ag) in enumerate(fixtures, start=1):
        rows.append(_fixture("2023-24", idx, idx, home, away, f"2023-08-{idx:02d}T12:00:00Z", hg, ag))
    return pd.DataFrame(rows)


def _mixed_fixture_rows() -> pd.DataFrame:
    rows = []
    fixtures = [
        ("team_a", "team_b", 0, 0),
        ("team_b", "team_c", 2, 1),
        ("team_c", "team_a", 1, 0),
        ("team_a", "team_c", 1, 1),
        ("team_b", "team_a", 0, 1),
        ("team_c", "team_b", 2, 2),
        ("team_a", "team_b", 3, 0),
    ]
    for idx, (home, away, hg, ag) in enumerate(fixtures, start=1):
        rows.append(_fixture("2023-24", idx, idx, home, away, f"2023-08-{idx:02d}T12:00:00Z", hg, ag))
    return pd.DataFrame(rows)


def _two_season_fixture_rows() -> pd.DataFrame:
    rows = []
    for idx, (home, away, hg, ag) in enumerate(
        [
            ("team_a", "team_b", 0, 0),
            ("team_b", "team_c", 2, 1),
            ("team_c", "team_a", 1, 1),
        ],
        start=1,
    ):
        rows.append(_fixture("2023-24", idx, idx, home, away, f"2023-08-{idx:02d}T12:00:00Z", hg, ag))
    rows.append(_fixture("2024-25", 1, 10, "team_a", "team_b", "2024-08-10T12:00:00Z", 4, 4))
    return pd.DataFrame(rows)
