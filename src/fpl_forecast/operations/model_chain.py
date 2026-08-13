from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.decision.milp import optimize_squad_expected_realized
from fpl_forecast.decision.rules import default_rules
from fpl_forecast.minutes_model.config import load_minutes_config
from fpl_forecast.minutes_model.data import load_minutes_frame
from fpl_forecast.minutes_model.models import fit_predict_minutes_models
from fpl_forecast.panel.common import normalize_name, phase2_dir, uid_from_slug
from fpl_forecast.panel.teams import TEAM_ALIAS_TEMPLATE, read_team_aliases
from fpl_forecast.operations.training_seasons import resolve_historical_training_seasons
from fpl_forecast.team_model.config import load_team_model_config
from fpl_forecast.team_model.data import load_historical_team_fixtures
from fpl_forecast.team_model.models import fit_predict_models
from fpl_forecast.team_model.probabilities import add_probability_columns
from fpl_forecast.xpoints.config import load_xpoints_config
from fpl_forecast.xpoints.data import load_xpoints_frame
from fpl_forecast.xpoints.models import predict_xpoints_models
from fpl_forecast.xpoints.simulation import aggregate_gameweek_draws


@dataclass(frozen=True)
class OperationalModelChainResult:
    team_predictions: pd.DataFrame
    minutes_predictions: pd.DataFrame
    xpoints_fixture_predictions: pd.DataFrame
    player_gameweek_predictions: pd.DataFrame
    decision_candidates: pd.DataFrame
    optimized_squad: pd.DataFrame
    optimized_lineup: pd.DataFrame
    model_comparison: pd.DataFrame
    lineage: dict[str, Any]


def run_operational_model_chain(
    *,
    season: str,
    run_id: str,
    output_dir: Path,
    normalized_dir: Path | str = NORMALIZED_DIR,
    fixture_variant: str = "base",
    price_variant: str = "base",
    target_gameweek: int = 1,
    as_of: pd.Timestamp | None = None,
    completed_player_fixtures: pd.DataFrame | None = None,
    completed_team_fixtures: pd.DataFrame | None = None,
    source_mode: str = "mock",
) -> OperationalModelChainResult:
    if source_mode not in {"mock", "official_current_season"}:
        raise ValueError(f"Unknown operational source mode: {source_mode}")
    if source_mode == "official_current_season":
        official_context = _official_current_context(season, normalized_dir=normalized_dir, target_gameweek=target_gameweek)
        as_of = as_of or official_context["deadline"]
    else:
        official_context = {}
        as_of = as_of or _default_information_cutoff(season, target_gameweek=target_gameweek)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    _validate_completed_inputs(
        completed_player_fixtures=completed_player_fixtures,
        completed_team_fixtures=completed_team_fixtures,
        information_cutoff=as_of,
    )
    training_seasons = resolve_historical_training_seasons(target_season=season, normalized_dir=normalized_dir)
    if source_mode == "official_current_season":
        target_fixtures = _official_target_fixtures(official_context, as_of=as_of)
        target_players = _official_target_players(
            season,
            normalized_dir=normalized_dir,
            price_variant=price_variant,
            team_identity=official_context["team_identity"],
            completed_player_fixtures=completed_player_fixtures,
        )
    else:
        target_fixtures = _target_fixtures(
            season,
            as_of=as_of,
            variant=fixture_variant,
            normalized_dir=normalized_dir,
            target_gameweek=target_gameweek,
        )
        target_players = _target_players(season, variant=price_variant, normalized_dir=normalized_dir)
    target_rows = _target_player_fixture_rows(target_fixtures, target_players, as_of=as_of)

    team_config = load_team_model_config()
    team_train = load_historical_team_fixtures(seasons=training_seasons, normalized_dir=normalized_dir)
    team_train = team_train.loc[team_train["result_valid"].astype(bool)].copy()
    if completed_team_fixtures is not None and not completed_team_fixtures.empty:
        team_train = pd.concat([team_train, _completed_team_training_frame(completed_team_fixtures)], ignore_index=True)
    team_predictions, team_ratings, team_diagnostics = fit_predict_models(
        team_train,
        target_fixtures,
        cutoff=as_of,
        config=team_config,
        models=["T2_REGULARIZED_ATTACK_DEFENCE"],
    )
    team_predictions = add_probability_columns(team_predictions, team_config)

    minutes_config = load_minutes_config()
    minutes_train = load_minutes_frame(seasons=training_seasons, normalized_dir=normalized_dir)
    if completed_player_fixtures is not None and not completed_player_fixtures.empty:
        completed_minutes = _completed_minutes_training_frame(
            completed_player_fixtures,
            historical=minutes_train,
        )
        minutes_train = pd.concat(
            [minutes_train, completed_minutes],
            ignore_index=True,
        )
    target_minutes_frame = _target_minutes_frame(target_rows, minutes_train)
    minutes_predictions, minutes_diagnostics = fit_predict_minutes_models(
        minutes_train,
        target_minutes_frame,
        config=minutes_config,
        models=["M3_EWMA_MINUTES", "M5_REGULARIZED_STATE_SOFTMAX", "M7_HIERARCHICAL_AVAILABILITY_STATE"],
    )
    minutes_predictions["minutes_variant"] = minutes_predictions["model_name"].map(
        {
            "M3_EWMA_MINUTES": "M3",
            "M5_REGULARIZED_STATE_SOFTMAX": "M5",
            "M7_HIERARCHICAL_AVAILABILITY_STATE": "M7",
        }
    )

    xpoints_config = load_xpoints_config()
    xpoints_train = load_xpoints_frame(seasons=training_seasons, normalized_dir=normalized_dir)
    if completed_player_fixtures is not None and not completed_player_fixtures.empty:
        xpoints_train = pd.concat([xpoints_train, _completed_xpoints_training_frame(completed_player_fixtures)], ignore_index=True)
    xpoints_test = _target_xpoints_frame(target_rows)
    phase3_reference = pd.DataFrame(columns=["season", "stable_fixture_uid", "player_uid", "expected_points"])
    xpoints_predictions, draws, conservation = predict_xpoints_models(
        xpoints_train,
        xpoints_test,
        minutes_predictions=minutes_predictions,
        team_predictions=team_predictions,
        phase3_reference=phase3_reference,
        config=xpoints_config,
        fold_index=0,
    )
    gameweek_predictions = aggregate_gameweek_draws(
        xpoints_predictions,
        draws,
        key_columns=["season", "gameweek", "player_uid", "model_name", "pre_deadline_population"],
    )
    gameweek_predictions, minutes_predictions = _add_blank_target_predictions(
        gameweek_predictions,
        minutes_predictions,
        target_players=target_players,
        season=season,
        target_gameweek=target_gameweek,
        draw_count=xpoints_config.draw_count,
    )
    gameweek_predictions = _add_gameweek_fixture_metadata(gameweek_predictions, target_rows)
    decision_candidates = _decision_candidates(gameweek_predictions, target_players, minutes_predictions)
    selected_candidates = decision_candidates.loc[
        decision_candidates["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M7")
    ].copy()
    solution = optimize_squad_expected_realized(selected_candidates, default_rules())
    optimized_squad = _squad_table(solution, selected_candidates)
    optimized_lineup = pd.DataFrame(
        [
            {
                "season": season,
                "gameweek": target_gameweek,
                "model_name": "X2_TEAM_CONSTRAINED_SIM_M7",
                "lineup": ",".join(solution.lineup_decision.lineup),
                "captain": solution.lineup_decision.captain,
                "vice_captain": solution.lineup_decision.vice_captain,
                "bench": ",".join(solution.lineup_decision.bench),
                "formation": solution.lineup_decision.formation,
                "cost_tenths": solution.cost_tenths,
                "bank_tenths": solution.bank_tenths,
                "expected_team_points": solution.objective,
                "optimizer_variant": "D2_EXPECTED_REALIZED_POINTS",
                "nominal_starting_xi_xpoints": (solution.diagnostics or {}).get("nominal_starting_xi_xpoints"),
                "expected_nominal_starting_xi_points": (solution.diagnostics or {}).get(
                    "expected_nominal_starting_xi_points"
                ),
                "expected_active_starter_points": (solution.diagnostics or {}).get("expected_active_starter_points"),
                "expected_autosub_contribution": (solution.diagnostics or {}).get("expected_autosub_contribution"),
                "expected_captain_bonus": (solution.diagnostics or {}).get("expected_captain_bonus"),
                "expected_vice_captain_contingency": (solution.diagnostics or {}).get(
                    "expected_vice_captain_contingency"
                ),
                "expected_realized_total": (solution.diagnostics or {}).get("expected_realized_total", solution.objective),
                "probability_all_starters_appear": (solution.diagnostics or {}).get("probability_all_starters_appear"),
                "expected_automatic_substitutions": (solution.diagnostics or {}).get(
                    "expected_automatic_substitutions"
                ),
                "probability_unreplaced_starter": (solution.diagnostics or {}).get("probability_unreplaced_starter"),
                "expected_bench_points_used": (solution.diagnostics or {}).get("expected_bench_points_used"),
                "scenario_count": (solution.diagnostics or {}).get("scenario_count"),
                "probability_mass": (solution.diagnostics or {}).get("probability_mass"),
                "analytic_method": (solution.diagnostics or {}).get("analytic_method"),
                "evaluated_squads": solution.evaluated_squads,
                "eligible_players_full_pool": (solution.diagnostics or {}).get("eligible_players_full_pool"),
                "raw_swap_proposals_generated": (solution.diagnostics or {}).get("raw_swap_proposals_generated"),
                "proposals_rejected_position_mismatch": (solution.diagnostics or {}).get(
                    "proposals_rejected_position_mismatch"
                ),
                "proposals_rejected_budget": (solution.diagnostics or {}).get("proposals_rejected_budget"),
                "proposals_rejected_club_limit": (solution.diagnostics or {}).get(
                    "proposals_rejected_club_limit"
                ),
                "other_illegal_proposals_rejected": (solution.diagnostics or {}).get(
                    "other_illegal_proposals_rejected"
                ),
                "feasible_unique_squad_proposals": (solution.diagnostics or {}).get(
                    "feasible_unique_squad_proposals"
                ),
                "duplicate_feasible_proposals_reevaluated": (solution.diagnostics or {}).get(
                    "duplicate_feasible_proposals_reevaluated"
                ),
                "squad_evaluation_calls": (solution.diagnostics or {}).get("squad_evaluation_calls"),
                "accepted_improving_moves": (solution.diagnostics or {}).get("accepted_improving_moves"),
                "local_search_iterations": (solution.diagnostics or {}).get("local_search_iterations"),
                "termination_reason": (solution.diagnostics or {}).get("termination_reason"),
                "lineup_refinement_status": (solution.diagnostics or {}).get("lineup_refinement_status"),
                "lineup_refinement_iterations": (solution.diagnostics or {}).get(
                    "lineup_refinement_iterations"
                ),
                "lineup_refinement_exact_evaluations": (solution.diagnostics or {}).get(
                    "lineup_refinement_exact_evaluations"
                ),
                "lineup_refinement_gain": (solution.diagnostics or {}).get("lineup_refinement_gain"),
                "lineup_refinement_returned_goalkeeper_order_value": (solution.diagnostics or {}).get(
                    "lineup_refinement_returned_goalkeeper_order_value"
                ),
                "lineup_refinement_reversed_goalkeeper_order_value": (solution.diagnostics or {}).get(
                    "lineup_refinement_reversed_goalkeeper_order_value"
                ),
                "d1_nominal_starting_xi_xpoints": (solution.diagnostics or {}).get(
                    "d1_nominal_starting_xi_xpoints"
                ),
                "d1_expected_active_starter_points": (solution.diagnostics or {}).get(
                    "d1_expected_active_starter_points"
                ),
                "d1_expected_autosub_contribution": (solution.diagnostics or {}).get(
                    "d1_expected_autosub_contribution"
                ),
                "d1_expected_captain_bonus": (solution.diagnostics or {}).get("d1_expected_captain_bonus"),
                "d1_expected_vice_captain_contingency": (solution.diagnostics or {}).get(
                    "d1_expected_vice_captain_contingency"
                ),
                "d1_expected_realized_total": (solution.diagnostics or {}).get("d1_expected_realized_total"),
                "d1_expected_automatic_substitutions": (solution.diagnostics or {}).get(
                    "d1_expected_automatic_substitutions"
                ),
                "d1_probability_all_starters_appear": (solution.diagnostics or {}).get(
                    "d1_probability_all_starters_appear"
                ),
                "d1_probability_unreplaced_starter": (solution.diagnostics or {}).get(
                    "d1_probability_unreplaced_starter"
                ),
                "d1_cost_tenths": (solution.diagnostics or {}).get("d1_cost_tenths"),
                "d1_bank_tenths": (solution.diagnostics or {}).get("d1_bank_tenths"),
                "d1_formation": (solution.diagnostics or {}).get("d1_formation"),
                "d1_squad": (solution.diagnostics or {}).get("d1_squad"),
                "d1_lineup": (solution.diagnostics or {}).get("d1_lineup"),
                "d1_captain": (solution.diagnostics or {}).get("d1_captain"),
                "d1_vice_captain": (solution.diagnostics or {}).get("d1_vice_captain"),
                "d1_bench_order": (solution.diagnostics or {}).get("d1_bench_order"),
                "optimality_scope": solution.optimality_scope,
                "solver_name": solution.solver_name,
                "solver_message": solution.solver_message,
                "solver_status": solution.solver_status,
                "objective_bound": solution.objective_bound,
                "objective_gap": solution.objective_gap,
                "runtime_seconds": solution.runtime_seconds,
            }
        ]
    )
    comparison = _model_comparison(decision_candidates)
    lineage = {
        "team_model_run_id": f"{run_id}_team_current",
        "minutes_model_run_id": f"{run_id}_minutes_current",
        "xpoints_model_run_id": f"{run_id}_xpoints_current",
        "decision_run_id": f"{run_id}_decision_current",
        "team_model": "T2_REGULARIZED_ATTACK_DEFENCE",
        "decision_optimizer": "D2_EXPECTED_REALIZED_POINTS",
        "decision_optimizer_scope": solution.optimality_scope,
        "minutes_models": [
            "M3_EWMA_MINUTES",
            "M5_REGULARIZED_STATE_SOFTMAX",
            "M7_HIERARCHICAL_AVAILABILITY_STATE",
        ],
        "xpoints_models": [
            "X2_TEAM_CONSTRAINED_SIM_M3",
            "X2_TEAM_CONSTRAINED_SIM_M5",
            "X2_TEAM_CONSTRAINED_SIM_M7",
        ],
        "xpoints_simulator": {
            "version": xpoints_config.simulation_version,
            "architecture_type": xpoints_config.architecture_type,
            "model_contract_version": xpoints_config.model_contract_version,
            "production_draw_count": xpoints_config.draw_count,
            "master_seed": xpoints_config.random_seed,
            "model_seed_offsets": {
                "X1_INDEPENDENT_COMPONENT_RATES_M3": 1,
                "X2_TEAM_CONSTRAINED_SIM_M3": 2,
                "X2_TEAM_CONSTRAINED_SIM_M5": 3,
                "X2_TEAM_CONSTRAINED_SIM_M7": 4,
            },
            "seed_derivation_policy": xpoints_config.seed_derivation_policy,
            "analytic_components": [
                "appearance",
                "goals",
                "assists",
                "clean_sheets",
                "saves",
                "penalties",
                "goals_conceded",
                "cards",
                "own_goals",
                "defensive_contribution",
                "bonus",
            ],
            "simulated_quantities": [
                "joint_scoreline",
                "scorer_allocation",
                "assist_allocation",
                "shared_clean_sheet",
                "shared_goals_conceded",
                "point_distribution",
                "tail_probabilities",
                "prediction_intervals",
            ],
            "component_reconciliation_tolerance": xpoints_config.component_reconciliation_tolerance,
        },
        "source_mode": source_mode,
        "training_seasons": training_seasons,
        "target_gameweek": target_gameweek,
        "target_deadline": as_of.isoformat(),
        "fixture_variant": fixture_variant,
        "price_variant": price_variant,
        "completed_player_fixture_rows_used": int(0 if completed_player_fixtures is None else len(completed_player_fixtures)),
        "completed_team_fixture_rows_used": int(0 if completed_team_fixtures is None else len(completed_team_fixtures)),
        "max_completed_source_available_time": _max_source_available_time(completed_player_fixtures),
        "team_diagnostics": team_diagnostics,
        "minutes_diagnostics": [diagnostic.__dict__ for diagnostic in minutes_diagnostics],
        "team_ratings_rows": int(len(team_ratings)),
        "conservation_rows": int(len(conservation)),
        **_official_lineage(
            source_mode=source_mode,
            official_context=official_context,
            target_players=target_players,
            target_fixtures=target_fixtures,
            decision_candidates=decision_candidates,
            normalized_dir=normalized_dir,
        ),
    }
    _write_chain_outputs(
        output_dir,
        team_predictions=team_predictions,
        minutes_predictions=minutes_predictions,
        xpoints_predictions=xpoints_predictions,
        gameweek_predictions=gameweek_predictions,
        decision_candidates=decision_candidates,
        optimized_squad=optimized_squad,
        optimized_lineup=optimized_lineup,
        comparison=comparison,
        lineage=lineage,
    )
    return OperationalModelChainResult(
        team_predictions=team_predictions,
        minutes_predictions=minutes_predictions,
        xpoints_fixture_predictions=xpoints_predictions,
        player_gameweek_predictions=gameweek_predictions,
        decision_candidates=decision_candidates,
        optimized_squad=optimized_squad,
        optimized_lineup=optimized_lineup,
        model_comparison=comparison,
        lineage=lineage,
    )


def _target_fixtures(
    season: str,
    *,
    as_of: pd.Timestamp,
    variant: str,
    normalized_dir: Path | str,
    target_gameweek: int,
) -> pd.DataFrame:
    teams = pd.read_parquet(phase2_dir(normalized_dir) / "dim_team.parquet")
    known = sorted(teams["team_uid"].dropna().astype(str).unique())[:20]
    if len(known) < 20:
        raise ValueError("Need at least 20 stable teams for mocked target fixtures.")
    known[-1] = "team_promoted_neutral_fallback"
    if variant == "opponent_swap":
        known[1], known[2] = known[2], known[1]
    rows = []
    for index in range(10):
        home = known[index * 2]
        away = known[index * 2 + 1]
        rows.append(
            {
                "season": season,
                "gameweek": target_gameweek,
                "stable_fixture_uid": f"{season}:mock_gw{target_gameweek}_{index + 1}",
                "fixture_key": f"{season}:mock_gw{target_gameweek}_{index + 1}",
                "source_fixture_id": (target_gameweek - 1) * 10 + index + 1,
                "home_team_uid": home,
                "away_team_uid": away,
                "source_home_team_id": index * 2 + 1,
                "source_away_team_id": index * 2 + 2,
                "home_team_name": home,
                "away_team_name": away,
                "home_team_short_name": str(home).removeprefix("team_").upper()[:3],
                "away_team_short_name": str(away).removeprefix("team_").upper()[:3],
                "kickoff_time": pd.Timestamp(
                    f"{season[:4]}-08-{15 + (target_gameweek - 1) * 7 + index // 5:02d}T15:00:00Z"
                ),
                "information_cutoff": as_of,
                "source_available_time": pd.NaT,
                "source_available_method": "future_fixture_no_result",
                "finished": False,
                "result_valid": False,
                "source_version": "mock_official_target_fixture",
                "raw_snapshot_path": "mock://phase8/fixtures",
            }
        )
    return pd.DataFrame(rows)


def _official_current_context(
    season: str,
    *,
    normalized_dir: Path | str,
    target_gameweek: int,
) -> dict[str, Any]:
    season_dir = Path(normalized_dir) / season
    players_path = season_dir / "current_players.parquet"
    teams_path = season_dir / "current_teams.parquet"
    fixtures_path = season_dir / "current_fixtures.parquet"
    events_path = season_dir / "current_events.parquet"
    missing = [path for path in (players_path, teams_path, fixtures_path, events_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing official normalized current-season inputs: {', '.join(str(path) for path in missing)}")
    events = pd.read_parquet(events_path)
    fixtures = pd.read_parquet(fixtures_path)
    teams = pd.read_parquet(teams_path)
    event = events.loc[pd.to_numeric(events["gameweek"], errors="coerce").eq(target_gameweek)]
    if event.empty:
        raise ValueError(f"Official current events do not contain gameweek {target_gameweek}.")
    deadline = pd.to_datetime(event.iloc[0]["deadline_time"], utc=True, errors="coerce")
    if pd.isna(deadline):
        raise ValueError(f"Official current gameweek {target_gameweek} has no valid deadline.")
    target = fixtures.loc[pd.to_numeric(fixtures["gameweek"], errors="coerce").eq(target_gameweek)].copy()
    if target.empty:
        raise ValueError(f"Official current fixtures do not contain gameweek {target_gameweek}.")
    if target["team_h_score"].notna().any() or target["team_a_score"].notna().any():
        raise ValueError("Official target fixtures contain outcomes and cannot be used for a pre-deadline forecast.")
    team_identity = _current_team_identity(teams, normalized_dir=normalized_dir, season=season)
    return {
        "deadline": deadline,
        "events": events,
        "event": event.iloc[0].to_dict(),
        "fixtures": target,
        "teams": teams,
        "team_identity": team_identity,
        "snapshot_metadata": _official_snapshot_metadata(season_dir),
    }


def _current_team_identity(teams: pd.DataFrame, *, normalized_dir: Path | str, season: str) -> pd.DataFrame:
    dim_path = phase2_dir(normalized_dir) / "dim_team.parquet"
    historical = pd.read_parquet(dim_path) if dim_path.exists() else pd.DataFrame(columns=["team_uid", "normalized_short_name"])
    if "normalized_short_name" not in historical.columns:
        historical["normalized_short_name"] = historical["canonical_name"].map(normalize_name)
    records = teams.copy()
    records["normalized_short_name"] = records["team_name"].map(normalize_name)
    aliases = read_team_aliases(TEAM_ALIAS_TEMPLATE)
    records["identity_lookup_name"] = records["normalized_short_name"].map(
        lambda value: normalize_name(aliases.get(value, _trim_team_suffix(value)))
    )
    duplicates = records["normalized_short_name"].duplicated(keep=False)
    if duplicates.any():
        names = ", ".join(records.loc[duplicates, "team_name"].astype(str))
        raise ValueError(f"Ambiguous current team identities after normalization: {names}")
    bridged = records.merge(
        historical[["team_uid", "normalized_short_name"]].drop_duplicates("normalized_short_name"),
        left_on="identity_lookup_name",
        right_on="normalized_short_name",
        how="left",
        suffixes=("", "_historical"),
    )
    bridged["identity_status"] = "historical"
    bridged.loc[bridged["team_uid"].isna(), "identity_status"] = "genuinely_unseen"
    bridged["team_uid"] = bridged.apply(
        lambda row: row.team_uid if pd.notna(row.team_uid) else uid_from_slug("team", row.team_name),
        axis=1,
    )
    bridged["neutral_team_strength_fallback"] = bridged["identity_status"].eq("genuinely_unseen")
    bridged["match_method"] = bridged["identity_status"].map(
        {"historical": "current_name_to_historical_identity", "genuinely_unseen": "official_name_deterministic_uid"}
    )
    output = bridged[
        [
            "season",
            "team_id",
            "team_code",
            "team_name",
            "short_name",
            "team_uid",
            "identity_status",
            "match_method",
            "neutral_team_strength_fallback",
            "source",
            "source_version",
            "retrieved_at",
            "raw_snapshot_path",
        ]
    ].sort_values("team_id")
    output.to_parquet(Path(normalized_dir) / season / "current_team_identities.parquet", index=False)
    return output


def _trim_team_suffix(normalized_name: str) -> str:
    for suffix in (" town", " city"):
        if normalized_name.endswith(suffix):
            return normalized_name[: -len(suffix)]
    return normalized_name


def _official_target_fixtures(context: dict[str, Any], *, as_of: pd.Timestamp) -> pd.DataFrame:
    fixtures = context["fixtures"].copy()
    team_identity = context["team_identity"]
    home = team_identity[["team_id", "team_uid", "team_name"]].rename(
        columns={"team_id": "home_team_id", "team_uid": "home_team_uid", "team_name": "home_team_name"}
    )
    away = team_identity[["team_id", "team_uid", "team_name"]].rename(
        columns={"team_id": "away_team_id", "team_uid": "away_team_uid", "team_name": "away_team_name"}
    )
    home_short = team_identity[["team_id", "short_name"]].rename(
        columns={"team_id": "home_team_id", "short_name": "home_team_short_name"}
    )
    away_short = team_identity[["team_id", "short_name"]].rename(
        columns={"team_id": "away_team_id", "short_name": "away_team_short_name"}
    )
    frame = (
        fixtures.merge(home, on="home_team_id", how="left")
        .merge(away, on="away_team_id", how="left")
        .merge(home_short, on="home_team_id", how="left")
        .merge(away_short, on="away_team_id", how="left")
    )
    if frame[["home_team_uid", "away_team_uid"]].isna().any().any():
        raise ValueError("Official target fixtures reference teams without stable current identities.")
    frame["stable_fixture_uid"] = frame["season"].astype(str) + ":official_fixture_" + frame["fixture_id"].astype(str)
    frame["fixture_key"] = frame["stable_fixture_uid"]
    frame["source_fixture_id"] = pd.to_numeric(frame["fixture_id"], errors="raise").astype(int)
    frame["source_home_team_id"] = pd.to_numeric(frame["home_team_id"], errors="raise").astype(int)
    frame["source_away_team_id"] = pd.to_numeric(frame["away_team_id"], errors="raise").astype(int)
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    if frame["kickoff_time"].isna().any():
        raise ValueError("Official target fixtures contain invalid kickoff_time values.")
    frame["information_cutoff"] = as_of
    frame["source_available_time"] = pd.NaT
    frame["source_available_method"] = "official_fixture_pre_deadline_no_result"
    frame["finished"] = False
    frame["result_valid"] = False
    return frame[
        [
            "season",
            "gameweek",
            "stable_fixture_uid",
            "fixture_key",
            "source_fixture_id",
            "home_team_uid",
            "away_team_uid",
            "source_home_team_id",
            "source_away_team_id",
            "home_team_name",
            "away_team_name",
            "home_team_short_name",
            "away_team_short_name",
            "kickoff_time",
            "information_cutoff",
            "source_available_time",
            "source_available_method",
            "finished",
            "result_valid",
            "source_version",
            "raw_snapshot_path",
        ]
    ].sort_values("source_fixture_id")


def _official_target_players(
    season: str,
    *,
    normalized_dir: Path | str,
    price_variant: str,
    team_identity: pd.DataFrame,
    completed_player_fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    season_dir = Path(normalized_dir) / season
    players = pd.read_parquet(season_dir / "current_players.parquet")
    entity_type = players["entity_type"] if "entity_type" in players.columns else pd.Series("player", index=players.index)
    exclusion_reason = pd.Series("", index=players.index, dtype="string")
    exclusion_reason.loc[~entity_type.eq("player")] = "non_player_entity"
    if "can_select" in players.columns:
        can_select = players["can_select"].astype("boolean")
        exclusion_reason.loc[can_select.notna() & ~can_select] = "official_can_select_false"
    if "removed" in players.columns:
        removed = players["removed"].astype("boolean")
        exclusion_reason.loc[removed.fillna(False)] = "official_removed_true"
    excluded = players.loc[exclusion_reason.ne("")].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = exclusion_reason.loc[excluded.index].to_numpy()
    _write_current_player_exclusions(season_dir, excluded)
    football = players.loc[exclusion_reason.eq("")].copy()
    if football.empty:
        raise ValueError("Official current player population is empty.")
    duplicate_codes = football.loc[football["player_code"].notna() & football["player_code"].duplicated(keep=False)]
    if not duplicate_codes.empty:
        names = ", ".join(duplicate_codes["web_name"].astype(str).head(10))
        raise ValueError(f"Ambiguous current player identities from duplicate player codes: {names}")
    if football["player_code"].isna().any():
        raise ValueError("Official current players without stable player_code cannot be bridged safely.")
    prices = pd.to_numeric(football["price_tenths"], errors="coerce")
    if prices.isna().any() or prices.le(0).any() or prices.mod(1).ne(0).any():
        raise ValueError("Official current players contain missing or invalid price_tenths.")
    team_map = team_identity[["team_id", "team_uid"]].rename(columns={"team_uid": "player_team_uid"})
    frame = football.merge(team_map, on="team_id", how="left")
    if frame["player_team_uid"].isna().any():
        raise ValueError("Official current players reference teams without stable current identities.")
    frame["player_uid"] = "player_code_" + frame["player_code"].astype("Int64").astype(str)
    latest_history = _latest_player_history(normalized_dir)
    frame = frame.merge(latest_history, on="player_uid", how="left", suffixes=("", "_historical"))
    frame["player_name"] = frame["web_name"].astype(str)
    frame["fpl_position"] = frame["position"].astype(str)
    frame["status"] = frame["status"].fillna("a").astype(str)
    frame["news"] = frame["news"].fillna("").astype(str) if "news" in frame.columns else ""
    for optional_column in ("can_select", "can_transact", "removed"):
        if optional_column not in frame.columns:
            frame[optional_column] = pd.NA
    for optional_column in ("chance_of_playing_next_round", "chance_of_playing_this_round"):
        if optional_column not in frame.columns:
            frame[optional_column] = pd.NA
    current_seen: set[str] = set()
    if completed_player_fixtures is not None and not completed_player_fixtures.empty:
        current_seen = set(
            completed_player_fixtures.loc[
                completed_player_fixtures["entity_type"].eq("player"), "player_uid"
            ].dropna().astype(str)
        )
    frame["current_season_history"] = frame["player_uid"].isin(current_seen)
    frame["cold_start_no_history"] = (
        frame["historical_player_team_uid"].isna() & ~frame["current_season_history"]
    )
    frame["fallback_flag"] = frame["cold_start_no_history"]
    frame["transferred_player"] = frame["historical_player_team_uid"].notna() & frame["historical_player_team_uid"].ne(frame["player_team_uid"])
    frame["position_change"] = frame["historical_fpl_position"].notna() & frame["historical_fpl_position"].ne(frame["fpl_position"])
    frame["lineage_note"] = "returning_player"
    frame.loc[frame["cold_start_no_history"], "lineage_note"] = "new_player_cold_start"
    frame.loc[
        frame["historical_player_team_uid"].isna() & frame["current_season_history"],
        "lineage_note",
    ] = "current_season_history_only"
    frame.loc[frame["transferred_player"], "lineage_note"] = "transferred_player"
    frame.loc[frame["position_change"], "lineage_note"] = "position_change"
    if price_variant == "premium_target":
        targets = frame.sort_values(["price_tenths", "player_uid"], ascending=[False, True]).head(3).index
        frame.loc[targets, "price_tenths"] = pd.to_numeric(frame.loc[targets, "price_tenths"], errors="coerce") + 300
    elif price_variant != "base":
        raise ValueError(f"Official current-season mode does not support price variant {price_variant}.")
    identity = frame[
        [
            "season",
            "player_id",
            "player_code",
            "web_name",
            "player_uid",
            "player_team_uid",
            "historical_player_team_uid",
            "fpl_position",
            "historical_fpl_position",
            "cold_start_no_history",
            "current_season_history",
            "transferred_player",
            "position_change",
            "lineage_note",
            "source",
            "source_version",
            "retrieved_at",
            "raw_snapshot_path",
        ]
    ].copy()
    identity.to_parquet(season_dir / "current_player_identities.parquet", index=False)
    pd.DataFrame(columns=["player_id", "player_code", "web_name", "review_reason"]).to_csv(
        season_dir / "current_player_identity_review.csv",
        index=False,
    )
    return frame[
        [
            "season",
            "player_uid",
            "player_name",
            "player_team_uid",
            "fpl_position",
            "price_tenths",
            "status",
            "news",
            "cold_start_no_history",
            "fallback_flag",
            "lineage_note",
            "transferred_player",
            "position_change",
            "player_id",
            "player_code",
            "team_id",
            "position_id",
            "can_select",
            "can_transact",
            "removed",
            "chance_of_playing_next_round",
            "chance_of_playing_this_round",
            "source_version",
            "raw_snapshot_path",
        ]
    ].drop_duplicates("player_uid")


def _write_current_player_exclusions(season_dir: Path, excluded: pd.DataFrame) -> None:
    keep = [
        "season",
        "player_id",
        "player_code",
        "web_name",
        "team_id",
        "position_id",
        "position",
        "entity_type",
        "can_select",
        "can_transact",
        "removed",
        "status",
        "news",
        "exclusion_reason",
        "source",
        "source_version",
        "retrieved_at",
        "raw_snapshot_path",
    ]
    if excluded.empty:
        pd.DataFrame(columns=keep).to_parquet(season_dir / "current_player_exclusions.parquet", index=False)
        return
    excluded[[column for column in keep if column in excluded.columns]].to_parquet(
        season_dir / "current_player_exclusions.parquet",
        index=False,
    )


def _latest_player_history(normalized_dir: Path | str) -> pd.DataFrame:
    fact_path = phase2_dir(normalized_dir) / "fact_player_fixture.parquet"
    fact = pd.read_parquet(fact_path)
    latest = (
        fact.loc[fact["entity_type"].eq("player")]
        .sort_values(["season", "gameweek", "source_available_time"])
        .groupby("player_uid", as_index=False)
        .tail(1)
    )
    return latest[
        ["player_uid", "player_team_uid", "fpl_position"]
    ].rename(
        columns={
            "player_team_uid": "historical_player_team_uid",
            "fpl_position": "historical_fpl_position",
        }
    )


def _official_snapshot_metadata(season_dir: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for table_name, endpoint in (
        ("current_players", "bootstrap_static"),
        ("current_fixtures", "fixtures"),
    ):
        path = season_dir / f"{table_name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["raw_snapshot_path", "retrieved_at", "source_version"])
        if frame.empty:
            continue
        raw_path = Path(str(frame["raw_snapshot_path"].dropna().iloc[0]))
        meta_path = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
        item: dict[str, Any] = {
            "retrieved_at": str(frame["retrieved_at"].dropna().iloc[0]),
            "source_version": str(frame["source_version"].dropna().iloc[0]),
            "raw_snapshot_file": "/".join(raw_path.parts[-4:]),
        }
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            item["sha256"] = meta.get("sha256")
            item["source_url"] = meta.get("source_url")
        metadata[endpoint] = item
    reconstruction_path = season_dir / "current_season_reconstruction.json"
    if reconstruction_path.exists():
        reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
        for endpoint, item in (reconstruction.get("official_snapshots") or {}).items():
            metadata[endpoint] = {
                "retrieved_at": item.get("retrieved_at"),
                "sha256": item.get("sha256"),
                "source_url": item.get("endpoint"),
                "raw_snapshot_file": "/".join(
                    Path(str(item.get("raw_snapshot_path") or "")).parts[-4:]
                ),
            }
    return metadata


def _official_lineage(
    *,
    source_mode: str,
    official_context: dict[str, Any],
    target_players: pd.DataFrame,
    target_fixtures: pd.DataFrame,
    decision_candidates: pd.DataFrame,
    normalized_dir: Path | str,
) -> dict[str, Any]:
    if source_mode != "official_current_season":
        return {
            "current_player_count": int(len(target_players)),
            "current_team_count": 20,
            "target_fixture_count": int(len(target_fixtures)),
            "price_source": "mock_adapter",
            "rules_source": "configured_phase8_rules",
        }
    team_identity = official_context["team_identity"]
    current_identity_path = Path(normalized_dir) / str(target_players["season"].iloc[0]) / "current_player_identities.parquet"
    identity = pd.read_parquet(current_identity_path) if current_identity_path.exists() else pd.DataFrame()
    cold_start_count = int(target_players["cold_start_no_history"].astype(bool).sum())
    neutral_count = int(team_identity["neutral_team_strength_fallback"].astype(bool).sum())
    reconstruction_path = Path(normalized_dir) / str(target_players["season"].iloc[0]) / "current_season_reconstruction.json"
    reconstruction = (
        json.loads(reconstruction_path.read_text(encoding="utf-8"))
        if reconstruction_path.exists()
        else {}
    )
    return {
        "official_deadline": official_context["deadline"].isoformat(),
        "official_snapshots": official_context["snapshot_metadata"],
        "current_player_count": int(len(target_players)),
        "current_team_count": int(len(team_identity)),
        "target_fixture_count": int(len(target_fixtures)),
        "identity_coverage": {
            "players_total": int(len(identity)) if not identity.empty else int(len(target_players)),
            "players_with_stable_uid": int(identity["player_uid"].notna().sum()) if not identity.empty else int(target_players["player_uid"].notna().sum()),
            "teams_total": int(len(team_identity)),
            "teams_with_stable_uid": int(team_identity["team_uid"].notna().sum()),
        },
        "cold_start_count": cold_start_count,
        "neutral_team_fallback_count": neutral_count,
        "price_source": "official_bootstrap_static.now_cost",
        "rules_source": "official_bootstrap_static.game_settings_and_element_types",
        "team_identity_status_counts": team_identity["identity_status"].value_counts().to_dict(),
        "player_lineage_note_counts": target_players["lineage_note"].value_counts().to_dict(),
        "mock_markers_present": False,
        "decision_candidate_count": int(len(decision_candidates)),
        "reconstructed_current_season_events": int(len(reconstruction.get("events") or [])),
        "reconstructed_blank_events": reconstruction.get("blank_events") or [],
        "current_season_temporal_policy": reconstruction.get("temporal_policy"),
        "current_season_source_available_policy": reconstruction.get("source_available_policy"),
    }


def _default_information_cutoff(season: str, *, target_gameweek: int) -> pd.Timestamp:
    if target_gameweek <= 1:
        return pd.Timestamp(f"{season[:4]}-08-01T10:00:00Z")
    day = 15 + (target_gameweek - 1) * 7 - 1
    return pd.Timestamp(f"{season[:4]}-08-{day:02d}T10:00:00Z")


def _target_players(season: str, *, variant: str, normalized_dir: Path | str) -> pd.DataFrame:
    fact = pd.read_parquet(phase2_dir(normalized_dir) / "fact_player_fixture.parquet")
    latest = (
        fact.loc[fact["entity_type"].eq("player")]
        .sort_values(["season", "gameweek", "player_uid"])
        .groupby("player_uid", as_index=False)
        .tail(1)
    )
    teams = sorted(latest["player_team_uid"].dropna().astype(str).unique())[:20]
    rows = []
    used_player_uids: set[str] = set()
    for team_index, team in enumerate(teams):
        team_pool = latest.loc[latest["player_team_uid"].eq(team)].copy()
        if team_pool.empty:
            continue
        for position, count in {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
            group = team_pool.loc[team_pool["fpl_position"].eq(position)].head(count)
            if len(group) < count:
                filler = team_pool.head(count - len(group)).copy()
                filler["fpl_position"] = position
                group = pd.concat([group, filler], ignore_index=True)
            for row in group.head(count).itertuples(index=False):
                player_uid = str(row.player_uid)
                if player_uid in used_player_uids:
                    player_uid = f"{player_uid}_phase8_{team_index}_{position}_{len(rows)}"
                used_player_uids.add(player_uid)
                rows.append(
                    {
                        "season": season,
                        "player_uid": player_uid,
                        "player_name": str(row.player_name),
                        "player_team_uid": team,
                        "fpl_position": position,
                        "price_tenths": _price_for(position, team_index, player_uid, variant),
                        "status": "a",
                        "news": "",
                        "cold_start_no_history": False,
                        "fallback_flag": False,
                        "lineage_note": "returning_player",
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates("player_uid").reset_index(drop=True)
    if not frame.empty:
        first = frame.index[0]
        frame.loc[first, "player_team_uid"] = teams[1]
        frame.loc[first, "lineage_note"] = "transferred_player"
        second = frame.index[1]
        frame.loc[second, "fpl_position"] = "MID" if frame.loc[second, "fpl_position"] != "MID" else "DEF"
        frame.loc[second, "lineage_note"] = "position_change"
    new_rows = [
        {
            "season": season,
            "player_uid": f"player_code_new_{index}",
            "player_name": f"New Player {index}",
            "player_team_uid": teams[index % len(teams)],
            "fpl_position": ["GKP", "DEF", "MID", "FWD"][index % 4],
            "price_tenths": 40 + index,
            "status": "a",
            "news": "Newly added official player code",
            "cold_start_no_history": True,
            "fallback_flag": True,
            "lineage_note": "new_player_cold_start",
        }
        for index in range(8)
    ]
    promoted = {
        "season": season,
        "player_uid": "player_code_promoted_fwd",
        "player_name": "Promoted Team Forward",
        "player_team_uid": "team_promoted_neutral_fallback",
        "fpl_position": "FWD",
        "price_tenths": 45,
        "status": "a",
        "news": "Promoted-team neutral fallback",
        "cold_start_no_history": True,
        "fallback_flag": True,
        "lineage_note": "promoted_team_neutral_fallback",
    }
    frame = pd.concat([frame, pd.DataFrame([*new_rows, promoted])], ignore_index=True)
    return frame


def _price_for(position: str, team_index: int, player_uid: str, variant: str) -> int:
    base = {"GKP": 45, "DEF": 48, "MID": 55, "FWD": 60}[position] + (team_index % 5)
    if variant == "discount_target" and position != "GKP" and team_index < 3:
        return max(40, base - 15)
    if variant == "premium_target" and position != "GKP" and team_index < 3:
        return base + 300
    return base


def _target_player_fixture_rows(fixtures: pd.DataFrame, players: pd.DataFrame, *, as_of: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for fixture in fixtures.itertuples(index=False):
        for side, team, opponent, was_home in (
            ("home", fixture.home_team_uid, fixture.away_team_uid, True),
            ("away", fixture.away_team_uid, fixture.home_team_uid, False),
        ):
            team_players = players.loc[players["player_team_uid"].eq(team)].copy()
            if team_players.empty:
                team_players = players.loc[players["lineage_note"].eq("promoted_team_neutral_fallback")].copy()
            for player in team_players.itertuples(index=False):
                rows.append(
                    {
                        "season": fixture.season,
                        "gameweek": int(fixture.gameweek),
                        "player_uid": player.player_uid,
                        "fixture_key": fixture.stable_fixture_uid,
                        "stable_fixture_uid": fixture.stable_fixture_uid,
                        "player_name": player.player_name,
                        "player_team_uid": team,
                        "opponent_team_uid": opponent,
                        "opponent_official_name": fixture.away_team_name if was_home else fixture.home_team_name,
                        "opponent_short_name": fixture.away_team_short_name if was_home else fixture.home_team_short_name,
                        "home_away": "H" if was_home else "A",
                        "fpl_position": player.fpl_position,
                        "entity_type": "player",
                        "was_home": was_home,
                        "kickoff_time": fixture.kickoff_time,
                        "information_cutoff": as_of,
                        "source_available_time": pd.Timestamp(fixture.kickoff_time) + pd.Timedelta(hours=3),
                        "source_available_method": "future_target_unavailable_until_after_fixture",
                        "minutes": 0,
                        "starts": pd.NA,
                        "source_version": "mock_official_target_bootstrap",
                        "raw_snapshot_path": "mock://phase8/bootstrap",
                        "price_tenths": player.price_tenths,
                        "status": player.status,
                        "news": player.news,
                        "can_select": getattr(player, "can_select", pd.NA),
                        "can_transact": getattr(player, "can_transact", pd.NA),
                        "removed": getattr(player, "removed", pd.NA),
                        "chance_of_playing_next_round": getattr(player, "chance_of_playing_next_round", pd.NA),
                        "chance_of_playing_this_round": getattr(player, "chance_of_playing_this_round", pd.NA),
                        "cold_start_no_history": bool(player.cold_start_no_history),
                        "fallback_flag": bool(player.fallback_flag),
                        "lineage_note": player.lineage_note,
                    }
                )
    return pd.DataFrame(rows)


def _add_gameweek_fixture_metadata(gameweek: pd.DataFrame, target_rows: pd.DataFrame) -> pd.DataFrame:
    if gameweek.empty:
        return gameweek
    metadata = _gameweek_fixture_metadata(target_rows)
    output = gameweek.merge(metadata, on=["season", "gameweek", "player_uid"], how="left")
    fill_values = {
        "fixture_count": 0,
        "opponent_display": "No fixture",
        "opponent_short_names": "",
        "opponent_official_names": "",
        "opponent_team_uids": "",
        "home_away_sequence": "",
        "kickoff_times": "",
    }
    for column, value in fill_values.items():
        if column not in output.columns:
            output[column] = value
        else:
            output[column] = output[column].fillna(value)
    output["fixture_count"] = pd.to_numeric(output["fixture_count"], errors="coerce").fillna(0).astype(int)
    return output


def _add_blank_target_predictions(
    gameweek: pd.DataFrame,
    minutes: pd.DataFrame,
    *,
    target_players: pd.DataFrame,
    season: str,
    target_gameweek: int,
    draw_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted = set(gameweek["player_uid"].astype(str)) if not gameweek.empty else set()
    blank = target_players.loc[~target_players["player_uid"].astype(str).isin(predicted)].copy()
    if blank.empty:
        return gameweek, minutes
    xpoints_rows: list[dict[str, Any]] = []
    minute_rows: list[dict[str, Any]] = []
    pairs = (
        ("M3", "M3_EWMA_MINUTES", "X2_TEAM_CONSTRAINED_SIM_M3"),
        ("M5", "M5_REGULARIZED_STATE_SOFTMAX", "X2_TEAM_CONSTRAINED_SIM_M5"),
        ("M7", "M7_HIERARCHICAL_AVAILABILITY_STATE", "X2_TEAM_CONSTRAINED_SIM_M7"),
    )
    for player in blank.itertuples(index=False):
        population = (
            "cold_start_no_history" if bool(player.cold_start_no_history) else "pre_deadline_history_active"
        )
        for minutes_variant, minutes_model, xpoints_model in pairs:
            minute_rows.append(
                {
                    "season": season,
                    "gameweek": target_gameweek,
                    "player_uid": player.player_uid,
                    "player_name": player.player_name,
                    "player_team_uid": player.player_team_uid,
                    "fpl_position": player.fpl_position,
                    "model_name": minutes_model,
                    "minutes_variant": minutes_variant,
                    "predicted_minutes": 0.0,
                    "p_appearance": 0.0,
                    "p_start": 0.0,
                    "p_reached_60": 0.0,
                }
            )
            xpoints_rows.append(
                {
                    "season": season,
                    "gameweek": target_gameweek,
                    "player_uid": player.player_uid,
                    "model_name": xpoints_model,
                    "pre_deadline_population": population,
                    "expected_points": 0.0,
                    "simulated_expected_points": 0.0,
                    "expected_points_unconditional": 0.0,
                    "raw_expected_points_given_appearance": 0.0,
                    "expected_points_given_appearance": 0.0,
                    "points_std": 0.0,
                    "points_p10": 0.0,
                    "points_p25": 0.0,
                    "points_p50": 0.0,
                    "points_p75": 0.0,
                    "points_p90": 0.0,
                    "prob_points_eq_0": 1.0,
                    "prob_points_ge_1": 0.0,
                    "prob_points_ge_5": 0.0,
                    "prob_points_ge_10": 0.0,
                    "simulation_draw_count": draw_count,
                    "conditional_estimate_source": "zero_fixture_rule",
                    "conditional_coherence_error": 0.0,
                }
            )
    return (
        pd.concat([gameweek, pd.DataFrame(xpoints_rows)], ignore_index=True, sort=False),
        pd.concat([minutes, pd.DataFrame(minute_rows)], ignore_index=True, sort=False),
    )


def _gameweek_fixture_metadata(target_rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "season",
        "gameweek",
        "player_uid",
        "fixture_count",
        "opponent_display",
        "opponent_short_names",
        "opponent_official_names",
        "opponent_team_uids",
        "home_away_sequence",
        "kickoff_times",
    ]
    if target_rows.empty:
        return pd.DataFrame(columns=columns)
    frame = target_rows.copy()
    for column in ("opponent_team_uid", "opponent_official_name", "opponent_short_name", "home_away"):
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[
        [
            "season",
            "gameweek",
            "player_uid",
            "fixture_key",
            "kickoff_time",
            "opponent_team_uid",
            "opponent_official_name",
            "opponent_short_name",
            "home_away",
        ]
    ].drop_duplicates(["season", "gameweek", "player_uid", "fixture_key"])
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    rows = []
    for key, group in frame.sort_values(["kickoff_time", "fixture_key"], na_position="last").groupby(
        ["season", "gameweek", "player_uid"],
        dropna=False,
    ):
        entries = [_fixture_display_entry(row) for row in group.itertuples(index=False)]
        kickoff_times = ["" if pd.isna(value) else str(value) for value in group["kickoff_time"]]
        rows.append(
            {
                "season": key[0],
                "gameweek": key[1],
                "player_uid": key[2],
                "fixture_count": int(group["fixture_key"].nunique()),
                "opponent_display": ", ".join(entries) if entries else "No fixture",
                "opponent_short_names": ",".join(group["opponent_short_name"].astype(str)),
                "opponent_official_names": ",".join(group["opponent_official_name"].astype(str)),
                "opponent_team_uids": ",".join(group["opponent_team_uid"].astype(str)),
                "home_away_sequence": ",".join(group["home_away"].astype(str)),
                "kickoff_times": ",".join(kickoff_times),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _fixture_display_entry(row) -> str:
    short_name = str(row.opponent_short_name or "").strip()
    if short_name.lower() in {"nan", "nat", "none", "<na>"}:
        short_name = ""
    if not short_name:
        short_name = str(row.opponent_official_name or row.opponent_team_uid or "").strip()
    home_away = str(row.home_away or "").strip()
    if home_away:
        return f"{short_name} ({home_away})"
    return short_name or "No fixture"


def _target_minutes_frame(target_rows: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = train.copy()
    train["source_available_time"] = pd.to_datetime(train["source_available_time"], utc=True)
    for row in target_rows.itertuples(index=False):
        history = train.loc[train["player_uid"].eq(row.player_uid)].sort_values("source_available_time")
        hist_minutes = pd.to_numeric(history["actual_minutes"], errors="coerce").fillna(0)
        hist_start = pd.to_numeric(history.get("actual_start", 0), errors="coerce").fillna(0)
        prior_seen = not history.empty
        rows.append(
            {
                **row._asdict(),
                "max_feature_source_available_time": history["source_available_time"].max()
                if prior_seen
                else pd.Timestamp("1900-01-01T00:00:00Z"),
                "lag1_minutes_mean": float(hist_minutes.tail(1).mean()) if prior_seen else 0.0,
                "lag3_minutes_mean": float(hist_minutes.tail(3).mean()) if prior_seen else 0.0,
                "lag5_minutes_mean": float(hist_minutes.tail(5).mean()) if prior_seen else 0.0,
                "lag10_minutes_mean": float(hist_minutes.tail(10).mean()) if prior_seen else 0.0,
                "lag3_appearance_rate": float(hist_minutes.tail(3).gt(0).mean()) if prior_seen else 0.0,
                "lag5_appearance_rate": float(hist_minutes.tail(5).gt(0).mean()) if prior_seen else 0.0,
                "lag3_start_rate": float(hist_start.tail(3).mean()) if prior_seen else 0.0,
                "lag5_start_rate": float(hist_start.tail(5).mean()) if prior_seen else 0.0,
                "season_minutes_before": 0.0,
                "season_appearance_before": 0.0,
                "season_starts_before": 0.0,
                "prior_season_minutes": float(hist_minutes.sum()),
                "prior_season_appearances": float(hist_minutes.gt(0).sum()),
                "prior_season_starts": float(hist_start.sum()) if prior_seen else 0.0,
                "history_fixture_count": float(len(history)),
                "days_since_last_source": 90.0 if prior_seen else 999.0,
                "evaluation_population": "pre_deadline_history_active" if prior_seen else "cold_start_no_history",
                "primary_data_quality_population": "all_observed_players",
                "pre_deadline_history_active": prior_seen,
                "cold_start_no_history": bool(row.cold_start_no_history) or not prior_seen,
                "transferred_player": row.lineage_note == "transferred_player",
                "position_change": row.lineage_note == "position_change",
                "price_tenths": getattr(row, "price_tenths", pd.NA),
                "status": getattr(row, "status", "a"),
                "news": getattr(row, "news", ""),
                "can_select": getattr(row, "can_select", pd.NA),
                "can_transact": getattr(row, "can_transact", pd.NA),
                "removed": getattr(row, "removed", pd.NA),
                "chance_of_playing_next_round": getattr(row, "chance_of_playing_next_round", pd.NA),
                "chance_of_playing_this_round": getattr(row, "chance_of_playing_this_round", pd.NA),
                "starts_exact_available": False,
            }
        )
    return pd.DataFrame(rows)


def _target_xpoints_frame(target_rows: pd.DataFrame) -> pd.DataFrame:
    frame = target_rows.copy()
    for column in [
        "total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "own_goals",
        "bonus",
        "bps",
    ]:
        frame[column] = 0
    frame["actual_total_points"] = 0
    frame["pre_deadline_population"] = np.where(
        frame["cold_start_no_history"], "cold_start_no_history", "pre_deadline_history_active"
    )
    return frame.drop(columns=["cold_start_no_history"])


def _decision_candidates(gameweek: pd.DataFrame, players: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    meta = players[
        [
            "player_uid",
            "player_name",
            "player_team_uid",
            "fpl_position",
            "price_tenths",
            "status",
            "news",
            "cold_start_no_history",
            "fallback_flag",
            "lineage_note",
        ]
    ].drop_duplicates("player_uid")
    probs = (
        minutes.groupby(["season", "gameweek", "player_uid", "minutes_variant"], as_index=False)
        .agg(
            expected_minutes=("predicted_minutes", "mean"),
            p_appearance=("p_appearance", "mean"),
            p_start=("p_start", "mean"),
        )
    )
    probs["model_name"] = probs["minutes_variant"].map(
        {
            "M3": "X2_TEAM_CONSTRAINED_SIM_M3",
            "M5": "X2_TEAM_CONSTRAINED_SIM_M5",
            "M7": "X2_TEAM_CONSTRAINED_SIM_M7",
        }
    )
    probs = probs.dropna(subset=["model_name"])
    out = gameweek.merge(meta, on="player_uid", how="left")
    out = out.merge(probs, on=["season", "gameweek", "player_uid", "model_name"], how="left")
    return out


def _squad_table(solution, candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.set_index("player_uid").loc[list(solution.squad)].reset_index()
    frame["selected_role"] = "squad"
    frame.loc[frame["player_uid"].isin(solution.lineup_decision.lineup), "selected_role"] = "starter"
    frame.loc[frame["player_uid"].eq(solution.lineup_decision.captain), "selected_role"] = "captain"
    frame.loc[frame["player_uid"].eq(solution.lineup_decision.vice_captain), "selected_role"] = "vice_captain"
    frame["bench_order"] = frame["player_uid"].map(
        {player: index for index, player in enumerate(solution.lineup_decision.bench, start=1)}
    )
    return frame


def _model_comparison(candidates: pd.DataFrame) -> pd.DataFrame:
    pivot = candidates.pivot_table(
        index="player_uid",
        columns="model_name",
        values="expected_points",
        aggfunc="first",
    )
    rows = []
    comparisons = [
        ("X2_TEAM_CONSTRAINED_SIM_M7", "X2_TEAM_CONSTRAINED_SIM_M3"),
        ("X2_TEAM_CONSTRAINED_SIM_M7", "X2_TEAM_CONSTRAINED_SIM_M5"),
        ("X2_TEAM_CONSTRAINED_SIM_M3", "X2_TEAM_CONSTRAINED_SIM_M5"),
    ]
    for left, right in comparisons:
        if {left, right}.issubset(pivot.columns):
            diff = pivot[left] - pivot[right]
            rows.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "players": int(diff.notna().sum()),
                    "mean_expected_difference": float(diff.mean()),
                    "max_abs_expected_difference": float(diff.abs().max()),
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _write_chain_outputs(output_dir: Path, **frames) -> None:
    chain_dir = output_dir / "model_chain"
    chain_dir.mkdir(parents=True, exist_ok=True)
    for name, value in frames.items():
        if isinstance(value, pd.DataFrame):
            if len(value) > 0:
                value.to_parquet(chain_dir / f"{name}.parquet", index=False)
            else:
                value.to_csv(chain_dir / f"{name}.csv", index=False)
        else:
            (chain_dir / f"{name}.json").write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _validate_completed_inputs(
    *,
    completed_player_fixtures: pd.DataFrame | None,
    completed_team_fixtures: pd.DataFrame | None,
    information_cutoff: pd.Timestamp,
) -> None:
    for name, frame, key in (
        ("completed player fixtures", completed_player_fixtures, ["season", "fixture_id", "player_uid"]),
        ("completed team fixtures", completed_team_fixtures, ["season", "stable_fixture_uid"]),
    ):
        if frame is None or frame.empty:
            continue
        missing = set(key + ["source_available_time"]).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {', '.join(sorted(missing))}")
        if frame.duplicated(key).any():
            raise ValueError(f"{name} contain duplicate keys.")
        available = pd.to_datetime(frame["source_available_time"], utc=True, errors="coerce")
        if available.isna().any():
            raise ValueError(f"{name} contain invalid source availability timestamps.")
        if available.ge(information_cutoff).any():
            raise ValueError(f"{name} include rows unavailable at the forecast cutoff.")
        if "fixture_completed" in frame.columns and (~frame["fixture_completed"].astype(bool)).any():
            raise ValueError(f"{name} include incomplete fixtures.")
        if "unresolved_source_limitation" in frame.columns and frame["unresolved_source_limitation"].astype(bool).any():
            raise ValueError(f"{name} include unresolved source limitations.")


def _completed_minutes_training_frame(
    frame: pd.DataFrame,
    *,
    historical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    output["player_team_uid"] = output.get("player_team_uid", output.get("team_uid"))
    output["opponent_team_uid"] = output.get("opponent_team_uid", output.get("opponent_uid"))
    output["stable_fixture_uid"] = output.get("stable_fixture_uid", output["season"].astype(str) + ":fixture_" + output["fixture_id"].astype(str))
    output["fixture_key"] = output.get("fixture_key", output["stable_fixture_uid"])
    output["player_name"] = output.get("player_name", output["player_uid"])
    output["information_cutoff"] = pd.to_datetime(output.get("information_cutoff", output["kickoff_time"]), utc=True)
    output["source_available_time"] = pd.to_datetime(output["source_available_time"], utc=True)
    output["actual_minutes"] = pd.to_numeric(output["minutes"], errors="coerce").fillna(0).astype(int)
    output["starts_exact_available"] = output["starts"].notna()
    output["actual_start"] = pd.to_numeric(output["starts"], errors="coerce").fillna(0).astype(int)
    output["actual_appearance"] = output["actual_minutes"].gt(0).astype(int)
    output["actual_reached_60"] = output["actual_minutes"].ge(60).astype(int)
    output["actual_played_90"] = output["actual_minutes"].ge(90).astype(int)
    output["actual_state"] = np.select(
        [
            output["actual_minutes"].eq(0),
            output["actual_start"].eq(0),
            output["actual_minutes"].lt(60),
            output["actual_minutes"].lt(90),
        ],
        ["DNP", "SUB_60_PLUS", "START_UNDER_60", "START_60_TO_89"],
        default="START_90",
    )
    for column in [
        "lag1_minutes_mean",
        "lag3_minutes_mean",
        "lag5_minutes_mean",
        "lag10_minutes_mean",
        "lag1_appearance_rate",
        "lag3_appearance_rate",
        "lag5_appearance_rate",
        "lag10_appearance_rate",
        "lag1_start_rate",
        "lag3_start_rate",
        "lag5_start_rate",
        "lag10_start_rate",
        "season_appearance_before",
        "season_minutes_before",
        "season_starts_before",
        "prior_season_minutes",
        "prior_season_appearances",
        "days_since_last_source",
    ]:
        output[column] = 0.0
    output["lag_source_available_time"] = pd.Series(
        pd.NaT,
        index=output.index,
        dtype="datetime64[ns, UTC]",
    )
    output["prior_seen_before"] = False
    output["max_feature_source_available_time"] = pd.Series(
        pd.Timestamp("1900-01-01T00:00:00Z"),
        index=output.index,
        dtype="datetime64[ns, UTC]",
    )
    output["transferred_player"] = output.get("lineage_note", "").eq("transferred_player")
    output["position_change"] = output.get("lineage_note", "").eq("position_change")
    output = _populate_completed_minutes_features(output, historical=historical)
    output["pre_deadline_history_active"] = output["prior_seen_before"].astype(bool)
    output["cold_start_no_history"] = ~output["prior_seen_before"].astype(bool)
    output["actual_appearances_diagnostic"] = output["actual_appearance"].astype(bool)
    output["evaluation_population"] = np.where(output["actual_appearance"].astype(bool), "pre_deadline_history_active", "cold_start_no_history")
    output["primary_data_quality_population"] = "all_observed_players"
    output["source_available_method"] = output.get("source_available_method", "official_event_live_retrieved_after_fixture_final")
    output["source_version"] = output.get("source_version", "event_live")
    output["raw_snapshot_path"] = output.get("raw_snapshot_path", "")
    return output


def _populate_completed_minutes_features(
    output: pd.DataFrame,
    *,
    historical: pd.DataFrame | None,
) -> pd.DataFrame:
    current = output.copy()
    pool_frames = [current]
    if historical is not None and not historical.empty:
        pool_frames.insert(0, historical.copy())
    pool = pd.concat(pool_frames, ignore_index=True, sort=False)
    pool["source_available_time"] = pd.to_datetime(pool["source_available_time"], utc=True)
    pool["actual_minutes"] = pd.to_numeric(pool["actual_minutes"], errors="coerce").fillna(0)
    pool["actual_start"] = pd.to_numeric(pool.get("actual_start", 0), errors="coerce").fillna(0)
    fallback = pd.Timestamp("1900-01-01T00:00:00Z")
    for index, row in current.iterrows():
        cutoff = pd.Timestamp(row["information_cutoff"])
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        available = pool.loc[
            pool["player_uid"].eq(row["player_uid"])
            & pool["source_available_time"].lt(cutoff)
        ].sort_values(["source_available_time", "kickoff_time", "fixture_key"])
        minutes = available["actual_minutes"]
        starts = available["actual_start"]
        for window in (1, 3, 5, 10):
            current.loc[index, f"lag{window}_minutes_mean"] = (
                float(minutes.tail(window).mean()) if not available.empty else 0.0
            )
            current.loc[index, f"lag{window}_appearance_rate"] = (
                float(minutes.tail(window).gt(0).mean()) if not available.empty else 0.0
            )
            current.loc[index, f"lag{window}_start_rate"] = (
                float(starts.tail(window).mean()) if not available.empty else 0.0
            )
        same_season = available.loc[available["season"].astype(str).eq(str(row["season"]))]
        prior_seasons = available.loc[~available["season"].astype(str).eq(str(row["season"]))]
        current.loc[index, "season_minutes_before"] = float(same_season["actual_minutes"].sum())
        current.loc[index, "season_appearance_before"] = float(same_season["actual_minutes"].gt(0).sum())
        current.loc[index, "season_starts_before"] = float(same_season["actual_start"].sum())
        current.loc[index, "prior_season_minutes"] = float(prior_seasons["actual_minutes"].sum())
        current.loc[index, "prior_season_appearances"] = float(
            prior_seasons["actual_minutes"].gt(0).sum()
        )
        current.loc[index, "prior_seen_before"] = not available.empty
        current.loc[index, "lag_source_available_time"] = (
            available["source_available_time"].max() if not available.empty else pd.NaT
        )
        current.loc[index, "max_feature_source_available_time"] = (
            available["source_available_time"].max() if not available.empty else fallback
        )
        current.loc[index, "days_since_last_source"] = (
            (cutoff - available["source_available_time"].max()).total_seconds() / 86400
            if not available.empty
            else 999.0
        )
    return current


def _completed_xpoints_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["player_team_uid"] = output.get("player_team_uid", output.get("team_uid"))
    output["opponent_team_uid"] = output.get("opponent_team_uid", output.get("opponent_uid"))
    output["stable_fixture_uid"] = output.get("stable_fixture_uid", output["season"].astype(str) + ":fixture_" + output["fixture_id"].astype(str))
    output["fixture_key"] = output.get("fixture_key", output["stable_fixture_uid"])
    output["target_total_points"] = pd.to_numeric(output["total_points"], errors="coerce").fillna(0).astype(int)
    output["actual_total_points"] = output["target_total_points"]
    output["pre_deadline_population"] = "post_match_completed_result"
    for column in ["price_tenths", "team_a_score", "team_h_score"]:
        if column not in output.columns:
            output[column] = 0
    return output


def _completed_team_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["information_cutoff"] = pd.to_datetime(output.get("information_cutoff", output["kickoff_time"]), utc=True)
    output["source_available_time"] = pd.to_datetime(output["source_available_time"], utc=True)
    output["finished"] = True
    output["result_valid"] = True
    output["source_version"] = output.get("source_version", "event_live")
    output["raw_snapshot_path"] = output.get("raw_snapshot_path", "")
    return output


def _max_source_available_time(frame: pd.DataFrame | None) -> str | None:
    if frame is None or frame.empty or "source_available_time" not in frame.columns:
        return None
    return pd.to_datetime(frame["source_available_time"], utc=True).max().isoformat()
