from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.decision.milp import optimize_squad_milp
from fpl_forecast.decision.rules import default_rules
from fpl_forecast.minutes_model.config import load_minutes_config
from fpl_forecast.minutes_model.data import load_minutes_frame
from fpl_forecast.minutes_model.models import fit_predict_minutes_models
from fpl_forecast.panel.common import phase2_dir
from fpl_forecast.team_model.config import load_team_model_config
from fpl_forecast.team_model.data import load_historical_team_fixtures
from fpl_forecast.team_model.models import fit_predict_models
from fpl_forecast.team_model.probabilities import add_probability_columns
from fpl_forecast.xpoints.config import load_xpoints_config
from fpl_forecast.xpoints.data import load_xpoints_frame
from fpl_forecast.xpoints.models import predict_xpoints_models
from fpl_forecast.xpoints.simulation import aggregate_gameweek_draws


PRICE_SENSITIVITY_PLAYER_UIDS = frozenset({"player_code_223094", "player_code_164511", "player_code_233420"})


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
) -> OperationalModelChainResult:
    as_of = as_of or _default_information_cutoff(season, target_gameweek=target_gameweek)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    _validate_completed_inputs(
        completed_player_fixtures=completed_player_fixtures,
        completed_team_fixtures=completed_team_fixtures,
        information_cutoff=as_of,
    )
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
    team_train = load_historical_team_fixtures(seasons=["2022-23", "2023-24", "2024-25"], normalized_dir=normalized_dir)
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
    minutes_train = load_minutes_frame(seasons=["2022-23", "2023-24", "2024-25"], normalized_dir=normalized_dir)
    if completed_player_fixtures is not None and not completed_player_fixtures.empty:
        minutes_train = pd.concat(
            [minutes_train, _completed_minutes_training_frame(completed_player_fixtures)],
            ignore_index=True,
        )
    target_minutes_frame = _target_minutes_frame(target_rows, minutes_train)
    minutes_predictions, minutes_diagnostics = fit_predict_minutes_models(
        minutes_train,
        target_minutes_frame,
        config=minutes_config,
        models=["M3_EWMA_MINUTES", "M5_REGULARIZED_STATE_SOFTMAX"],
    )
    minutes_predictions["minutes_variant"] = minutes_predictions["model_name"].map(
        {"M3_EWMA_MINUTES": "M3", "M5_REGULARIZED_STATE_SOFTMAX": "M5"}
    )

    xpoints_config = load_xpoints_config()
    xpoints_train = load_xpoints_frame(seasons=["2022-23", "2023-24", "2024-25"], normalized_dir=normalized_dir)
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
    decision_candidates = _decision_candidates(gameweek_predictions, target_players, minutes_predictions)
    selected_candidates = decision_candidates.loc[
        decision_candidates["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M3")
    ].copy()
    solution = optimize_squad_milp(selected_candidates, default_rules())
    optimized_squad = _squad_table(solution, selected_candidates)
    optimized_lineup = pd.DataFrame(
        [
            {
                "season": season,
                "gameweek": target_gameweek,
                "model_name": "X2_TEAM_CONSTRAINED_SIM_M3",
                "lineup": ",".join(solution.lineup_decision.lineup),
                "captain": solution.lineup_decision.captain,
                "vice_captain": solution.lineup_decision.vice_captain,
                "bench": ",".join(solution.lineup_decision.bench),
                "formation": solution.lineup_decision.formation,
                "cost_tenths": solution.cost_tenths,
                "bank_tenths": solution.bank_tenths,
                "expected_team_points": solution.objective,
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
        "minutes_models": ["M3_EWMA_MINUTES", "M5_REGULARIZED_STATE_SOFTMAX"],
        "xpoints_models": ["X2_TEAM_CONSTRAINED_SIM_M3", "X2_TEAM_CONSTRAINED_SIM_M5"],
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
    if variant == "premium_target" and player_uid in PRICE_SENSITIVITY_PLAYER_UIDS:
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
                        "cold_start_no_history": bool(player.cold_start_no_history),
                        "fallback_flag": bool(player.fallback_flag),
                        "lineage_note": player.lineage_note,
                    }
                )
    return pd.DataFrame(rows)


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
                "days_since_last_source": 90.0 if prior_seen else 999.0,
                "evaluation_population": "pre_deadline_history_active" if prior_seen else "cold_start_no_history",
                "primary_data_quality_population": "all_observed_players",
                "pre_deadline_history_active": prior_seen,
                "cold_start_no_history": bool(row.cold_start_no_history) or not prior_seen,
                "transferred_player": row.lineage_note == "transferred_player",
                "position_change": row.lineage_note == "position_change",
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
        minutes.groupby(["season", "gameweek", "player_uid"], as_index=False)
        .agg(
            expected_minutes=("predicted_minutes", "mean"),
            p_appearance=("p_appearance", "mean"),
            p_start=("p_start", "mean"),
        )
    )
    out = gameweek.merge(meta, on="player_uid", how="left")
    out = out.merge(probs, on=["season", "gameweek", "player_uid"], how="left")
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
    if {"X2_TEAM_CONSTRAINED_SIM_M3", "X2_TEAM_CONSTRAINED_SIM_M5"}.issubset(pivot.columns):
        diff = pivot["X2_TEAM_CONSTRAINED_SIM_M3"] - pivot["X2_TEAM_CONSTRAINED_SIM_M5"]
        return pd.DataFrame(
            [
                {
                    "left_model": "X2_TEAM_CONSTRAINED_SIM_M3",
                    "right_model": "X2_TEAM_CONSTRAINED_SIM_M5",
                    "players": int(diff.notna().sum()),
                    "mean_expected_difference": float(diff.mean()),
                    "max_abs_expected_difference": float(diff.abs().max()),
                }
            ]
        )
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


def _completed_minutes_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
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
    output["lag_source_available_time"] = pd.NaT
    output["prior_seen_before"] = False
    output["max_feature_source_available_time"] = pd.Timestamp("1900-01-01T00:00:00Z")
    output["transferred_player"] = output.get("lineage_note", "").eq("transferred_player")
    output["position_change"] = output.get("lineage_note", "").eq("position_change")
    output["pre_deadline_history_active"] = output["actual_appearance"].astype(bool)
    output["cold_start_no_history"] = False
    output["actual_appearances_diagnostic"] = output["actual_appearance"].astype(bool)
    output["evaluation_population"] = np.where(output["actual_appearance"].astype(bool), "pre_deadline_history_active", "cold_start_no_history")
    output["primary_data_quality_population"] = "all_observed_players"
    output["source_available_method"] = output.get("source_available_method", "official_event_live_retrieved_after_fixture_final")
    output["source_version"] = output.get("source_version", "event_live")
    output["raw_snapshot_path"] = output.get("raw_snapshot_path", "")
    return output


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
