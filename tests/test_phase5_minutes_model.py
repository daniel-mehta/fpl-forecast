from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fpl_forecast.minutes_model.coherence import add_lineup_coherence
from fpl_forecast.minutes_model.config import load_minutes_config
from fpl_forecast.minutes_model.data import (
    assert_frozen_predictions_target_free,
    build_minutes_frame,
    role_duration_state,
    validate_minutes_frame,
)
from fpl_forecast.minutes_model.inference import validate_current_minutes_inputs
from fpl_forecast.minutes_model.metrics import minutes_metric_tables, score_minutes_predictions
from fpl_forecast.minutes_model.models import fit_predict_minutes_models


def test_role_duration_state_labels_use_exact_starts():
    assert role_duration_state(0, 0) == "DNP"
    assert role_duration_state(22, 0) == "SUB_UNDER_60"
    assert role_duration_state(65, 0) == "SUB_60_PLUS"
    assert role_duration_state(45, 1) == "START_UNDER_60"
    assert role_duration_state(75, 1) == "START_60_TO_89"
    assert role_duration_state(90, 1) == "START_90"


def test_minutes_frame_populations_and_leakage_validation():
    frame = build_minutes_frame(_fact_rows(), _feature_rows(), _player_map())

    validate_minutes_frame(frame)

    p1_gw1 = frame.loc[frame["season"].eq("2023-24") & frame["player_uid"].eq("player_1")].iloc[0]
    p2_gw1 = frame.loc[frame["season"].eq("2023-24") & frame["player_uid"].eq("player_2")].iloc[0]
    assert p1_gw1["evaluation_population"] == "pre_deadline_history_active"
    assert p2_gw1["evaluation_population"] == "cold_start_no_history"
    assert p1_gw1["transferred_player"]
    assert p1_gw1["position_change"]
    assert p1_gw1["actual_state"] == "START_90"

    leaked = frame.copy()
    leaked.loc[0, "max_feature_source_available_time"] = leaked.loc[0, "information_cutoff"]
    with pytest.raises(ValueError, match="at or after the cutoff"):
        validate_minutes_frame(leaked)


def test_minutes_models_emit_target_free_coherent_predictions_and_metrics():
    config = replace(load_minutes_config(), softmax_iterations=20, bootstrap_samples=4)
    frame = build_minutes_frame(_fact_rows(), _feature_rows(), _player_map())
    train = frame.loc[frame["season"].eq("2022-23")]
    test = frame.loc[frame["season"].eq("2023-24")]

    predictions, diagnostics = fit_predict_minutes_models(train, test, config=config)

    assert {item.model_name for item in diagnostics} == set(config.model_names)
    assert_frozen_predictions_target_free(predictions, config.forbidden_frozen_columns)
    prob_cols = [column for column in predictions.columns if column.startswith("prob_state_")]
    assert np.allclose(predictions[prob_cols].sum(axis=1), 1.0)
    assert predictions["predicted_minutes"].between(0, 90).all()
    assert (predictions["p_start"] <= predictions["p_appearance"] + 1e-9).all()

    scored = score_minutes_predictions(predictions, frame)
    tables = minutes_metric_tables(scored, config)
    assert set(tables.overall["model_name"]) == set(config.model_names)
    assert {"appearance", "start", "reached_60", "played_90"}.issubset(set(tables.binary["target"]))


def test_lineup_adjustment_preserves_ranking_and_sums_to_eleven_when_complete():
    config = replace(load_minutes_config(), lineup_adjustment_min_candidates=15)
    base = pd.DataFrame(
        {
            "model_name": ["M"] * 16,
            "season": ["2024-25"] * 16,
            "stable_fixture_uid": ["f1"] * 16,
            "player_team_uid": ["team_a"] * 16,
            "p_start": np.linspace(0.1, 0.9, 16),
            "prob_state_dnp": [0.1] * 16,
            "prob_state_sub_under_60": [0.1] * 16,
            "prob_state_sub_60_plus": [0.1] * 16,
            "prob_state_start_under_60": [0.2] * 16,
            "prob_state_start_60_to_89": [0.2] * 16,
            "prob_state_start_90": [0.3] * 16,
        }
    )

    adjusted, diagnostics = add_lineup_coherence(base, config)

    assert diagnostics.iloc[0]["lineup_adjustment_applied"]
    assert adjusted["p_start"].sum() == pytest.approx(11.0)
    assert adjusted["p_start"].rank().equals(base["p_start"].rank())


def test_current_minutes_inputs_reject_mismatched_current_season(tmp_path):
    season_dir = tmp_path / "2026-27"
    phase2_dir = tmp_path / "phase2"
    season_dir.mkdir()
    phase2_dir.mkdir()
    pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "fixture_code": 1,
                "gameweek": 1,
                "home_team_id": 1,
                "away_team_id": 2,
                "kickoff_time": "2025-08-15T19:00:00Z",
                "finished": False,
                "started": False,
                "team_h_score": None,
                "team_a_score": None,
                "source": "fpl_api",
                "source_version": "snapshot",
                "retrieved_at": "2026-07-22T00:00:00Z",
                "season": "2026-27",
                "raw_snapshot_path": "raw.json",
            }
        ]
    ).to_parquet(season_dir / "current_fixtures.parquet", index=False)
    pd.DataFrame([{"team_id": 1, "team_name": "A"}, {"team_id": 2, "team_name": "B"}]).to_parquet(
        season_dir / "current_teams.parquet",
        index=False,
    )
    pd.DataFrame([{"player_code": 1, "season": "2026-27"}]).to_parquet(
        season_dir / "current_players.parquet",
        index=False,
    )
    pd.DataFrame({"team_uid": ["team_a", "team_b"], "normalized_short_name": ["a", "b"]}).to_parquet(
        phase2_dir / "dim_team.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="season mismatch"):
        validate_current_minutes_inputs(
            season="2026-27",
            gameweek=1,
            as_of="2026-07-22T00:00:00Z",
            normalized_dir=tmp_path,
        )


def _fact_rows() -> pd.DataFrame:
    rows = [
        _fact("2022-23", 1, "player_1", "f1", 90, 1, "MID", "team_old", "2022-08-01T15:00:00Z"),
        _fact("2022-23", 2, "player_1", "f2", 25, 0, "MID", "team_old", "2022-08-08T15:00:00Z"),
        _fact("2022-23", 2, "player_2", "f2", 0, 0, "FWD", "team_b", "2022-08-08T15:00:00Z"),
        _fact("2023-24", 1, "player_1", "f3", 90, 1, "FWD", "team_new", "2023-08-12T15:00:00Z"),
        _fact("2023-24", 1, "player_2", "f3", 0, 0, "FWD", "team_b", "2023-08-12T15:00:00Z"),
    ]
    return pd.DataFrame(rows)


def _fact(
    season: str,
    gameweek: int,
    player_uid: str,
    fixture_key: str,
    minutes: int,
    starts: int,
    position: str,
    team_uid: str,
    source_available_time: str,
) -> dict[str, object]:
    kickoff = pd.Timestamp(source_available_time) - pd.Timedelta(hours=3)
    return {
        "season": season,
        "gameweek": gameweek,
        "player_uid": player_uid,
        "fixture_key": fixture_key,
        "player_name": player_uid,
        "player_team_uid": team_uid,
        "opponent_team_uid": "opponent",
        "fpl_position": position,
        "entity_type": "player",
        "was_home": True,
        "kickoff_time": kickoff.isoformat(),
        "information_cutoff": kickoff.isoformat(),
        "source_available_time": source_available_time,
        "source_available_method": "kickoff_plus_3h",
        "minutes": minutes,
        "starts": starts,
        "source_version": "test",
        "raw_snapshot_path": "raw.csv",
    }


def _feature_rows() -> pd.DataFrame:
    records = []
    for row in _fact_rows().itertuples(index=False):
        prior_minutes = 115 if row.season == "2023-24" and row.player_uid == "player_1" else 0
        prior_apps = 2 if row.season == "2023-24" and row.player_uid == "player_1" else 0
        records.append(
            {
                "season": row.season,
                "player_uid": row.player_uid,
                "fixture_key": row.fixture_key,
                "prev_fixture_minutes": 0,
                "prev3_minutes_sum": prior_minutes,
                "prev5_minutes_sum": prior_minutes,
                "prev3_starts_sum": prior_apps,
                "prev5_starts_sum": prior_apps,
                "season_to_date_minutes": 0,
                "season_to_date_appearances": 0,
                "prior_source_season": "2022-23" if prior_apps else None,
                "prior_season_minutes": prior_minutes,
                "prior_season_appearances": prior_apps,
                "player_max_source_available_time": "1900-01-01T00:00:00Z",
            }
        )
    return pd.DataFrame(records)


def _player_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": "2022-23", "player_uid": "player_1", "source_team": "Old", "fpl_position": "MID"},
            {"season": "2023-24", "player_uid": "player_1", "source_team": "New", "fpl_position": "FWD"},
            {"season": "2022-23", "player_uid": "player_2", "source_team": "B", "fpl_position": "FWD"},
            {"season": "2023-24", "player_uid": "player_2", "source_team": "B", "fpl_position": "FWD"},
        ]
    )
