from __future__ import annotations

import pandas as pd

from fpl_forecast.backtest.config import BacktestConfig


BASELINES = (
    "B0_ZERO",
    "B1_GLOBAL_MEAN",
    "B2_POSITION_MEAN",
    "B3_RECENT_POINTS_P3",
    "B3_RECENT_POINTS_P5",
    "B4_RECENT_MINUTES_P3",
    "B4_RECENT_MINUTES_P5",
    "B5_EB_POINTS_PER90",
    "B6_PREVIOUS_SEASON_GW1",
)


def predict_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> pd.DataFrame:
    if train.empty:
        raise ValueError("Cannot score baselines without training rows.")
    global_mean = float(pd.to_numeric(train["target_total_points"], errors="coerce").mean())
    if pd.isna(global_mean):
        global_mean = 0.0
    position_means = (
        train.assign(target_total_points=pd.to_numeric(train["target_total_points"], errors="coerce"))
        .groupby("fpl_position")["target_total_points"]
        .mean()
        .to_dict()
    )
    position_points_per90 = _points_per90_by_position(train, fallback=global_mean)
    prior_player = _previous_season_player_rates(train, config=config, fallback=global_mean)

    id_columns = [
        "season",
        "gameweek",
        "fixture_key",
        "player_uid",
        "player_name",
        "fpl_position",
        "player_team_uid",
        "opponent_team_uid",
        "information_cutoff",
        "source_available_time",
        "minutes",
        "target_total_points",
        "pre_deadline_population",
    ]
    feature_columns = [
        "prev3_total_points_sum",
        "prev5_total_points_sum",
        "prev3_minutes_sum",
        "prev5_minutes_sum",
        "season_to_date_appearances",
        "prior_season_minutes",
        "prior_season_appearances",
    ]
    if "pre_deadline_population" not in test.columns:
        test = test.assign(pre_deadline_population="unknown_population")
    output = test[[*id_columns, *feature_columns]].copy()
    output["B0_ZERO"] = 0.0
    output["B1_GLOBAL_MEAN"] = global_mean
    output["B2_POSITION_MEAN"] = (
        output["fpl_position"].map(position_means).fillna(global_mean).astype(float)
    )
    for window in (3, 5):
        points = pd.to_numeric(output[f"prev{window}_total_points_sum"], errors="coerce")
        minutes = pd.to_numeric(output[f"prev{window}_minutes_sum"], errors="coerce")
        history_games = (pd.to_numeric(output["season_to_date_appearances"], errors="coerce").fillna(0))
        denominator = history_games.clip(lower=1, upper=window)
        has_history = history_games > 0
        output[f"B3_RECENT_POINTS_P{window}"] = (points / denominator).where(has_history, global_mean)

        expected_minutes = (minutes / float(window)).clip(lower=0, upper=config.minutes_cap)
        per90 = output["fpl_position"].map(position_points_per90).fillna(global_mean)
        output[f"B4_RECENT_MINUTES_P{window}"] = (expected_minutes / 90.0) * per90

    recent_points = pd.to_numeric(output["prev5_total_points_sum"], errors="coerce").fillna(0)
    recent_minutes = pd.to_numeric(output["prev5_minutes_sum"], errors="coerce").fillna(0)
    prior_minutes = float(config.points_per_90_prior_matches * 90)
    global_per90 = _global_points_per90(train, fallback=global_mean)
    shrunk_per90 = ((recent_points * 90.0) + (global_per90 * prior_minutes)) / (
        recent_minutes + prior_minutes
    )
    expected_minutes = (recent_minutes / 5.0).clip(lower=0, upper=config.minutes_cap)
    output["B5_EB_POINTS_PER90"] = (expected_minutes / 90.0) * shrunk_per90

    output = output.merge(prior_player, on=["player_uid", "season"], how="left")
    output["B6_PREVIOUS_SEASON_GW1"] = output["previous_season_b6_prediction"].fillna(global_mean)
    output = output.drop(columns=["previous_season_b6_prediction"])

    melted = output.melt(
        id_vars=id_columns,
        value_vars=list(BASELINES),
        var_name="baseline",
        value_name="prediction",
    )
    melted["prediction"] = pd.to_numeric(melted["prediction"], errors="coerce").fillna(global_mean)
    return melted


def _global_points_per90(train: pd.DataFrame, *, fallback: float) -> float:
    minutes = pd.to_numeric(train["minutes"], errors="coerce").fillna(0)
    points = pd.to_numeric(train["target_total_points"], errors="coerce").fillna(0)
    total_minutes = float(minutes.sum())
    if total_minutes <= 0:
        return fallback
    return float(points.sum() * 90.0 / total_minutes)


def _points_per90_by_position(train: pd.DataFrame, *, fallback: float) -> dict[str, float]:
    frame = train.copy()
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0)
    frame["target_total_points"] = pd.to_numeric(frame["target_total_points"], errors="coerce").fillna(0)
    rows = {}
    for position, group in frame.groupby("fpl_position"):
        minutes = float(group["minutes"].sum())
        rows[position] = fallback if minutes <= 0 else float(group["target_total_points"].sum() * 90.0 / minutes)
    return rows


def _previous_season_player_rates(
    train: pd.DataFrame,
    *,
    config: BacktestConfig,
    fallback: float,
) -> pd.DataFrame:
    frame = train.copy()
    frame["target_total_points"] = pd.to_numeric(frame["target_total_points"], errors="coerce")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0)
    frame["appearance"] = (frame["minutes"] > 0).astype(int)
    frame["season_start"] = frame["season"].str.slice(0, 4).astype(int)
    global_per90 = _global_points_per90(frame, fallback=fallback)
    prior_minutes = float(config.points_per_90_prior_matches * 90)
    aggregates = (
        frame.groupby(["player_uid", "season_start"], as_index=False)
        .agg(
            previous_season_points=("target_total_points", "sum"),
            previous_season_minutes=("minutes", "sum"),
            previous_season_appearances=("appearance", "sum"),
        )
    )
    active = (aggregates["previous_season_minutes"] > 0) & (
        aggregates["previous_season_appearances"] > 0
    )
    shrunk_rate = (
        (aggregates["previous_season_points"] * 90.0) + (global_per90 * prior_minutes)
    ) / (aggregates["previous_season_minutes"] + prior_minutes)
    expected_minutes = (
        aggregates["previous_season_minutes"] / aggregates["previous_season_appearances"].clip(lower=1)
    ).clip(lower=0, upper=config.minutes_cap)
    aggregates["previous_season_b6_prediction"] = ((expected_minutes / 90.0) * shrunk_rate).where(
        active
    )
    aggregates["season"] = (aggregates["season_start"] + 1).map(
        lambda year: f"{year}-{str(year + 1)[-2:]}"
    )
    return aggregates[["player_uid", "season", "previous_season_b6_prediction"]]
