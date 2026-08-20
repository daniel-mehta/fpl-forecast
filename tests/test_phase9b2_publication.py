from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fpl_forecast.operations.publication_pipeline import (
    PublicationError,
    PublicationPreparation,
    TargetGameweekResolution,
    prepare_publication_data,
    resolve_target_gameweek,
    run_official_publication_forecast,
    validate_publication_candidate,
)
from fpl_forecast.operations.current_panel import CurrentSeasonReconstruction
from fpl_forecast.operations.publication import publish_failure, publish_success
from fpl_forecast.operations.orchestrator import require_clean_source_state


NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)


def test_target_gameweek_resolves_earliest_unfinished_future_deadline() -> None:
    resolution = resolve_target_gameweek(
        season="2026-27",
        events=_events(),
        fixtures=_fixtures(),
        requested_gameweek=None,
        now=NOW,
    )

    assert resolution.gameweek == 1
    assert resolution.method == "earliest_unfinished_future_deadline"
    assert resolution.prior_events_verified == 0


def test_prepare_publication_reconstructs_from_empty_directories(monkeypatch, tmp_path) -> None:
    raw_fpl = tmp_path / "raw_fpl"
    raw_vaastav = tmp_path / "raw_vaastav"
    normalized = tmp_path / "normalized"
    recorded: list[str] = []

    class FakeVaastav:
        def __init__(self, *, raw_dir):
            assert raw_dir == raw_vaastav

        def ingest_season(self, *, season, revision, refresh):
            recorded.append(f"history:{season}:{revision}:{refresh}")

    class FakeFPL:
        def __init__(self, *, raw_dir):
            assert raw_dir == raw_fpl

        def snapshot_current(self, *, season, refresh, offline, extra_metadata=None):
            recorded.append(f"current:{season}:{refresh}:{offline}")

    def fake_normalize_history(*, season, raw_dir, normalized_dir):
        output = normalized_dir / season
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "season": season,
                    "player_id": 1,
                    "player_code": 101,
                    "fixture_id": 10,
                    "gameweek": 1,
                    "kickoff_time": "2025-08-15T19:00:00Z",
                    "retrieved_at": "2026-07-24T10:00:00Z",
                }
            ]
        ).to_parquet(output / "historical_player_fixtures.parquet", index=False)

    def fake_normalize_current(*, season, raw_dir, normalized_dir):
        output = normalized_dir / season
        output.mkdir(parents=True, exist_ok=True)
        _events().to_parquet(output / "current_events.parquet", index=False)
        _fixtures().to_parquet(output / "current_fixtures.parquet", index=False)

    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.VaastavIngestor", FakeVaastav)
    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.FPLApiClient", FakeFPL)
    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.normalize_historical", fake_normalize_history)
    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.normalize_current", fake_normalize_current)
    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.build_identities", lambda **kwargs: None)
    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.build_panel", lambda **kwargs: None)
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline._validate_phase2_publication_artifacts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.validate_all",
        lambda **kwargs: SimpleNamespace(errors=[]),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.audit_leakage",
        lambda **kwargs: SimpleNamespace(errors=[]),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.check_season_launch",
        lambda **kwargs: SimpleNamespace(
            status=SimpleNamespace(state=SimpleNamespace(value="READY_TO_REFRESH"), reason="ready")
        ),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline._snapshot_source_hashes",
        lambda *args, **kwargs: {"bootstrap_static": "a" * 64, "fixtures": "b" * 64},
    )

    prepared = prepare_publication_data(
        season="2026-27",
        requested_gameweek=1,
        historical_seasons=("2024-25", "2025-26"),
        revision="f" * 40,
        raw_fpl_dir=raw_fpl,
        raw_vaastav_dir=raw_vaastav,
        normalized_dir=normalized,
        now=NOW,
    )

    assert prepared.target.gameweek == 1
    assert prepared.historical_rows == {"2024-25": 1, "2025-26": 1}
    assert recorded == [
        f"history:2024-25:{'f' * 40}:True",
        f"history:2025-26:{'f' * 40}:True",
        "current:2026-27:True:False",
    ]


def test_prepare_publication_fails_when_historical_source_is_missing(monkeypatch, tmp_path) -> None:
    class MissingVaastav:
        def __init__(self, *, raw_dir):
            pass

        def ingest_season(self, **kwargs):
            raise RuntimeError("historical source missing")

    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.VaastavIngestor", MissingVaastav)

    with pytest.raises(RuntimeError, match="historical source missing"):
        prepare_publication_data(
            season="2026-27",
            historical_seasons=("2025-26",),
            raw_fpl_dir=tmp_path / "fpl",
            raw_vaastav_dir=tmp_path / "history",
            normalized_dir=tmp_path / "normalized",
            now=NOW,
        )


def test_preparation_reconstructs_completed_current_season_beyond_gw1(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.resolve_target_gameweek",
        lambda **kwargs: TargetGameweekResolution(
            season="2026-27",
            gameweek=2,
            deadline_time="2026-08-22T11:00:00+00:00",
            resolved_at=NOW.isoformat(),
            method="validated_workflow_input",
            prior_events_verified=1,
        ),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.VaastavIngestor",
        lambda **kwargs: SimpleNamespace(ingest_season=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.normalize_historical",
        lambda **kwargs: _write_minimal_history(kwargs["normalized_dir"], kwargs["season"]),
    )
    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.build_identities", lambda **kwargs: None)
    monkeypatch.setattr("fpl_forecast.operations.publication_pipeline.build_panel", lambda **kwargs: None)
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline._validate_phase2_publication_artifacts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.validate_all",
        lambda **kwargs: SimpleNamespace(errors=[]),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.audit_leakage",
        lambda **kwargs: SimpleNamespace(errors=[]),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.FPLApiClient",
        lambda **kwargs: SimpleNamespace(snapshot_current=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.normalize_current",
        lambda **kwargs: _write_minimal_current(kwargs["normalized_dir"], kwargs["season"]),
    )
    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.check_season_launch",
        lambda **kwargs: SimpleNamespace(
            status=SimpleNamespace(state=SimpleNamespace(value="READY_TO_REFRESH"), reason="ready")
        ),
    )
    manifest = tmp_path / "normalized" / "2026-27" / "current_season_reconstruction.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")
    reconstructed: dict[str, object] = {}

    def fake_reconstruct(**kwargs):
        reconstructed.update(kwargs)
        return CurrentSeasonReconstruction(
            player_history_path=None,
            team_history_path=None,
            manifest_path=manifest,
            player_rows=0,
            team_rows=0,
            event_count=1,
            blank_events=(),
            source_hashes={
                "bootstrap_static": "a" * 64,
                "fixtures": "b" * 64,
                "event_live_1": "c" * 64,
            },
        )

    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.reconstruct_completed_current_season",
        fake_reconstruct,
    )

    prepared = prepare_publication_data(
        season="2026-27",
        requested_gameweek=2,
        historical_seasons=("2025-26",),
        raw_fpl_dir=tmp_path / "fpl",
        raw_vaastav_dir=tmp_path / "history",
        normalized_dir=tmp_path / "normalized",
        now=NOW,
        run_id="official_gw2_test",
    )

    assert prepared.target.gameweek == 2
    assert prepared.reconstructed_events == 1
    assert prepared.source_hashes["event_live_1"] == "c" * 64
    assert reconstructed["target_gameweek"] == 2
    assert reconstructed["run_id"] == "official_gw2_test"


def test_target_gameweek_rejects_completed_or_past_event() -> None:
    events = _events()
    events.loc[events["gameweek"].eq(1), ["finished", "deadline_time"]] = [
        True,
        "2026-07-20T11:00:00Z",
    ]

    with pytest.raises(PublicationError, match="completed"):
        resolve_target_gameweek(
            season="2026-27",
            events=events,
            fixtures=_fixtures(),
            requested_gameweek=1,
            now=NOW,
        )


def test_target_gameweek_rejects_incomplete_prior_results() -> None:
    events = _events()
    events.loc[events["gameweek"].eq(1), ["finished", "data_checked"]] = [True, False]

    with pytest.raises(PublicationError, match="not finalized"):
        resolve_target_gameweek(
            season="2026-27",
            events=events,
            fixtures=_fixtures(),
            requested_gameweek=2,
            now=NOW,
        )


def test_publication_candidate_accepts_legal_official_contract(tmp_path) -> None:
    run_dir, preparation = _publication_candidate(tmp_path)

    result = validate_publication_candidate(
        run_dir=run_dir,
        preparation=preparation,
        audit_dir=tmp_path / "audit",
        now=NOW,
    )

    assert result.run_id == "official_publication"
    assert all(value == "passed" for value in result.gates.values())
    assert result.audit_path.is_file()
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert "raw_data" not in audit
    assert audit["target_gameweek_resolution"]["gameweek"] == 1


def test_official_publication_requires_authoritative_clean_run_class(
    monkeypatch, tmp_path
) -> None:
    _, preparation = _publication_candidate(tmp_path)
    captured = {}

    def fake_refresh(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status=SimpleNamespace(
                state=SimpleNamespace(value="SUCCEEDED"), reason="published"
            ),
            run_dir=tmp_path / "official_run",
        )

    monkeypatch.setattr(
        "fpl_forecast.operations.publication_pipeline.refresh_operational", fake_refresh
    )

    run_official_publication_forecast(
        preparation=preparation,
        run_id="official_clean_run",
        raw_fpl_dir=tmp_path / "raw",
        normalized_dir=tmp_path / "normalized",
    )

    assert captured["authoritative_publication"] is True
    assert captured["mock_launch"] is False
    assert captured["run_id"] == "official_clean_run"


def test_publication_candidate_rejects_mock_source(tmp_path) -> None:
    run_dir, preparation = _publication_candidate(tmp_path)
    manifest = _read(run_dir / "run_manifest.json")
    manifest["model_lineage"]["source_mode"] = "mock"
    _write_json(run_dir / "run_manifest.json", manifest)

    with pytest.raises(PublicationError, match="official_source"):
        validate_publication_candidate(
            run_dir=run_dir,
            preparation=preparation,
            audit_dir=tmp_path / "audit",
            now=NOW,
        )


def test_publication_candidate_rejects_wrong_season(tmp_path) -> None:
    run_dir, preparation = _publication_candidate(tmp_path)
    manifest = _read(run_dir / "run_manifest.json")
    manifest["inferred_official_season"] = "2025-26"
    _write_json(run_dir / "run_manifest.json", manifest)

    with pytest.raises(PublicationError, match="inferred_season_match"):
        validate_publication_candidate(
            run_dir=run_dir,
            preparation=preparation,
            audit_dir=tmp_path / "audit",
            now=NOW,
        )


def test_publication_candidate_rejects_invalid_identity_or_frontend_schema(tmp_path) -> None:
    run_dir, preparation = _publication_candidate(tmp_path)
    projections = pd.read_csv(run_dir / "player_gameweek_projections.csv")
    projections.loc[0, "stable_player_id"] = pd.NA
    projections.to_csv(run_dir / "player_gameweek_projections.csv", index=False)

    with pytest.raises(PublicationError, match="identity_coverage"):
        validate_publication_candidate(
            run_dir=run_dir,
            preparation=preparation,
            audit_dir=tmp_path / "audit",
            now=NOW,
        )


def test_publication_candidate_requires_direct_conditional_xpoints(tmp_path) -> None:
    run_dir, preparation = _publication_candidate(tmp_path)
    projections = pd.read_csv(run_dir / "player_gameweek_projections.csv").drop(
        columns=["expected_points_given_appearance"]
    )
    projections.to_csv(run_dir / "player_gameweek_projections.csv", index=False)

    with pytest.raises(PublicationError, match="conditional_xpoints_required"):
        validate_publication_candidate(
            run_dir=run_dir,
            preparation=preparation,
            audit_dir=tmp_path / "audit",
            now=NOW,
        )


def test_publication_candidate_rejects_stale_artifact(tmp_path) -> None:
    run_dir, preparation = _publication_candidate(tmp_path)
    freshness = _read(run_dir / "data_freshness.json")
    freshness["generated_at"] = "2026-07-23T00:00:00Z"
    _write_json(run_dir / "data_freshness.json", freshness)

    with pytest.raises(PublicationError, match="freshness_age"):
        validate_publication_candidate(
            run_dir=run_dir,
            preparation=preparation,
            audit_dir=tmp_path / "audit",
            now=NOW,
        )


def test_publication_candidate_rejects_local_paths(tmp_path) -> None:
    run_dir, preparation = _publication_candidate(tmp_path)
    status = _read(run_dir / "operational_status.json")
    status["debug"] = "/Users/example/private.csv"
    _write_json(run_dir / "operational_status.json", status)

    with pytest.raises(PublicationError, match="no_local_paths"):
        validate_publication_candidate(
            run_dir=run_dir,
            preparation=preparation,
            audit_dir=tmp_path / "audit",
            now=NOW,
        )


def test_synchronized_publication_must_match_validated_run(tmp_path) -> None:
    run_dir, preparation = _publication_candidate(tmp_path)
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    for source in run_dir.iterdir():
        if source.is_file():
            (public_dir / source.name).write_bytes(source.read_bytes())
    manifest = _read(public_dir / "run_manifest.json")
    manifest["run_id"] = "different_run"
    _write_json(public_dir / "run_manifest.json", manifest)

    with pytest.raises(PublicationError, match="synced_run_id"):
        validate_publication_candidate(
            run_dir=run_dir,
            preparation=preparation,
            audit_dir=tmp_path / "audit",
            public_dir=public_dir,
            now=NOW,
        )


def test_pages_workflow_is_manual_official_only_and_failure_preserving() -> None:
    workflow = Path(".github/workflows/deploy-pages.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "--mock-launch" not in workflow
    assert "contents: write" in workflow
    assert "needs: build" in workflow
    assert "needs.build.result == 'success'" in workflow
    assert "npm run sync-data" in workflow
    assert "validate-publication" in workflow
    assert "Freeze the sanitized public forecast bundle" in workflow
    assert "official-forecast-data" in workflow


def test_frozen_forecast_seed_handles_new_and_existing_branches_safely() -> None:
    workflow = Path(".github/workflows/seed-frozen-forecast.yml").read_text(encoding="utf-8")

    assert "git switch --orphan official-forecast-data" in workflow
    assert "git rm -rf ." not in workflow
    assert workflow.index("fetch-live-frozen-forecast.mjs") < workflow.index("git switch --orphan")
    assert "frozen-forecast-cli.mjs validate --source" in workflow
    assert workflow.index("frozen-forecast-cli.mjs validate --source") < workflow.index(
        'cmp "$RUNNER_TEMP/data/frozen_forecast_manifest.json"'
    )
    assert "Existing frozen bundle differs; refusing to overwrite it." in workflow


def test_failed_publication_does_not_change_latest_successful_pointer(monkeypatch, tmp_path) -> None:
    runs = tmp_path / "runs"
    failed = tmp_path / "failed"
    pointer = tmp_path / "latest_successful.json"
    monkeypatch.setattr("fpl_forecast.operations.publication.OPERATIONAL_RUNS_DIR", runs)
    monkeypatch.setattr("fpl_forecast.operations.publication.OPERATIONAL_FAILED_DIR", failed)
    monkeypatch.setattr("fpl_forecast.operations.publication.LATEST_SUCCESSFUL_PATH", pointer)

    successful_temp = tmp_path / "successful_temp"
    successful_temp.mkdir()
    publish_success(
        successful_temp,
        run_id="good",
        manifest={
            "completed_at": "2026-07-24T11:00:00Z",
            "frontend_schema_version": "phase9_frontend_v1",
        },
    )
    before = pointer.read_text(encoding="utf-8")
    failed_temp = tmp_path / "failed_temp"
    failed_temp.mkdir()
    publish_failure(failed_temp, run_id="bad", manifest={"error": "validation failed"})

    assert pointer.read_text(encoding="utf-8") == before


def test_operational_publication_refuses_to_replace_existing_run(monkeypatch, tmp_path) -> None:
    runs = tmp_path / "runs"
    pointer = tmp_path / "latest_successful.json"
    existing = runs / "immutable"
    existing.mkdir(parents=True)
    marker = existing / "run_manifest.json"
    marker.write_text('{"preserve": true}\n', encoding="utf-8")
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr("fpl_forecast.operations.publication.OPERATIONAL_RUNS_DIR", runs)
    monkeypatch.setattr(
        "fpl_forecast.operations.publication.LATEST_SUCCESSFUL_PATH", pointer
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_success(
            temp,
            run_id="immutable",
            manifest={
                "completed_at": "2026-07-24T11:00:00Z",
                "frontend_schema_version": "phase9_frontend_v1",
            },
        )

    assert marker.read_text(encoding="utf-8") == '{"preserve": true}\n'
    assert temp.is_dir()
    assert not pointer.exists()


def test_ci_workflows_separate_fast_full_and_frontend_jobs() -> None:
    fast = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    full = Path(".github/workflows/full-python.yml").read_text(encoding="utf-8")
    frontend = Path(".github/workflows/frontend-ci.yml").read_text(encoding="utf-8")

    assert 'pytest -q -m "not slow"' in fast
    assert "npm " not in fast
    assert "workflow_dispatch:" in full and "schedule:" in full
    assert "pytest -q" in full and '-m "not slow"' not in full
    assert "npm run lint" in frontend and "npm run build" in frontend
    assert "pytest" not in frontend
    assert "deploy-pages" not in frontend


def test_publication_generator_and_policy_are_repository_inputs() -> None:
    ignore_policy = Path(".gitignore").read_text(encoding="utf-8")
    artifact_policy = Path("docs/research/publication-artifacts.md").read_text(
        encoding="utf-8"
    )

    assert Path("scripts/build_paper_evidence.py").is_file()
    assert Path("scripts/replay_clean_prospective_evidence.py").is_file()
    assert "!scripts/build_paper_evidence.py" in ignore_policy
    assert "/paper" in ignore_policy
    assert "The completed replay" in artifact_policy
    assert "Raw or normalized third-party FPL data" in artifact_policy
    assert "/tmp" in artifact_policy
    assert Path(
        "reports/goalkeeper_scoring_fix/clean_replay_inventory_034830b041c1.json"
    ).is_file()

    help_run = subprocess.run(
        [
            sys.executable,
            "scripts/replay_clean_prospective_evidence.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-id" in help_run.stdout


def test_clean_prospective_replay_rejects_dirty_source() -> None:
    with pytest.raises(RuntimeError, match="requires a clean Git worktree"):
        require_clean_source_state(
            {"dirty": True}, operation="Clean prospective evidence replay"
        )

    require_clean_source_state(
        {"dirty": False}, operation="Clean prospective evidence replay"
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameweek": 1,
                "deadline_time": "2026-08-15T11:00:00Z",
                "finished": False,
                "data_checked": False,
                "is_current": False,
                "is_next": True,
            },
            {
                "gameweek": 2,
                "deadline_time": "2026-08-22T11:00:00Z",
                "finished": False,
                "data_checked": False,
                "is_current": False,
                "is_next": False,
            },
        ]
    )


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameweek": gameweek,
                "finished": False,
                "finished_provisional": False,
            }
            for gameweek in (1, 2)
        ]
    )


def _publication_candidate(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source_hashes = {"bootstrap_static": "a" * 64, "fixtures": "b" * 64}
    preparation = PublicationPreparation(
        season="2026-27",
        historical_seasons=("2022-23", "2023-24", "2024-25", "2025-26"),
        source_revision="f" * 40,
        target=TargetGameweekResolution(
            season="2026-27",
            gameweek=1,
            deadline_time="2026-08-15T11:00:00+00:00",
            resolved_at=NOW.isoformat(),
            method="validated_workflow_input",
            prior_events_verified=0,
        ),
        historical_rows={"2022-23": 1, "2023-24": 1, "2024-25": 1, "2025-26": 1},
        stage_seconds={"historical_reconstruction": 1.0},
        launch_state="READY_TO_REFRESH",
        source_hashes=source_hashes,
    )
    manifest = {
        "schema_version": "phase8_operational_v1",
        "frontend_schema_version": "phase9_frontend_v1",
        "run_id": "official_publication",
        "run_class": "authoritative_publication",
        "target_season": "2026-27",
        "target_gameweek": 1,
        "inferred_official_season": "2026-27",
        "warnings": [],
        "completed_at": "2026-07-24T11:00:00Z",
        "model_lineage": {
            "source_mode": "official_current_season",
            "team_model": "T2_REGULARIZED_ATTACK_DEFENCE",
            "minutes_models": ["M7_HIERARCHICAL_AVAILABILITY_STATE"],
            "xpoints_models": ["X2_TEAM_CONSTRAINED_SIM_M7"],
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(
        run_dir / "operational_status.json",
        {
            "schema_version": "phase9_frontend_v1",
            "state": "SUCCEEDED",
            "target_season": "2026-27",
            "target_gameweek": 1,
            "run_id": "official_publication",
            "warning": None,
            "disclaimer": "Experimental forecast. Not affiliated with Fantasy Premier League.",
        },
    )
    _write_json(
        run_dir / "data_freshness.json",
        {
            "schema_version": "phase9_frontend_v1",
            "generated_at": "2026-07-24T11:00:00Z",
            "stale": False,
            "official_snapshots": {
                key: {"sha256": value, "retrieved_at": "2026-07-24T10:55:00Z"}
                for key, value in source_hashes.items()
            },
        },
    )
    projections = pd.DataFrame(
        [
            {
                "schema_version": "phase9_frontend_v1",
                "season": "2026-27",
                "gameweek": 1,
                "stable_player_id": "player_code_101",
                "player": "Archer",
                "team": "team_arsenal",
                "position": "MID",
                "price_tenths": 70,
                "fixture_count": 1,
                "opponent_display": "CHE (H)",
                "expected_points": 4.2,
                "expected_points_given_appearance": 4.7,
                "expected_minutes": 75,
                "p_appearance": 0.9,
                "p_start": 0.8,
                "prob_points_ge_5": 0.35,
                "prob_points_ge_10": 0.08,
            }
        ]
    )
    projections.to_csv(run_dir / "player_gameweek_projections.csv", index=False)
    squad = _squad()
    squad.to_csv(run_dir / "optimized_squad.csv", index=False)
    pd.DataFrame(
        [
            {
                "solver_status": "heuristic_feasible",
                "lineup_refinement_status": "single_change_local_optimum",
                "termination_reason": "configured_iteration_bound_reached",
                "optimality_scope": "bounded squad search and fixed-squad lineup refinement",
            }
        ]
    ).to_csv(run_dir / "optimized_lineup.csv", index=False)
    pd.DataFrame([{"model": "X2_TEAM_CONSTRAINED_SIM_M7"}]).to_csv(
        run_dir / "model_comparison.csv",
        index=False,
    )
    return run_dir, preparation


def _squad() -> pd.DataFrame:
    positions = ["GKP", "GKP", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD"]
    roles = [
        "Starter",
        "Squad 1",
        "Starter",
        "Starter",
        "Starter",
        "Squad 2",
        "Squad 3",
        "Captain",
        "Vice Captain",
        "Starter",
        "Starter",
        "Starter",
        "Starter",
        "Starter",
        "Squad 4",
    ]
    return pd.DataFrame(
        [
            {
                "player_uid": f"player_code_{1000 + index}",
                "player_name": f"Footballer{index}",
                "fpl_position": position,
                "player_team_uid": f"team_{index % 5}",
                "price_tenths": 50,
                "selected_role": roles[index],
                "bench_order": index if roles[index].startswith("Squad") else 0,
            }
            for index, position in enumerate(positions)
        ]
    )


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_minimal_history(normalized_dir: Path, season: str) -> None:
    output = normalized_dir / season
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "season": season,
                "player_id": 1,
                "player_code": 101,
                "fixture_id": 10,
                "gameweek": 1,
                "kickoff_time": "2025-08-15T19:00:00Z",
                "retrieved_at": "2026-07-24T10:00:00Z",
            }
        ]
    ).to_parquet(output / "historical_player_fixtures.parquet", index=False)


def _write_minimal_current(normalized_dir: Path, season: str) -> None:
    output = normalized_dir / season
    output.mkdir(parents=True, exist_ok=True)
    _events().to_parquet(output / "current_events.parquet", index=False)
    _fixtures().to_parquet(output / "current_fixtures.parquet", index=False)
