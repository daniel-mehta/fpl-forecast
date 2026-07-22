from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestFold:
    fold_id: str
    season: str
    gameweek: int
    information_cutoff: pd.Timestamp
    train_rows: int
    test_rows: int


def season_sort_key(season: str) -> int:
    return int(str(season).split("-", maxsplit=1)[0])


def add_season_order(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["season_order"] = output["season"].map(season_sort_key)
    return output


def chronological_gameweek_folds(
    frame: pd.DataFrame,
    *,
    test_seasons: list[str],
    mode: str,
    minimum_training_rows: int = 1,
) -> list[BacktestFold]:
    required = {"season", "gameweek", "information_cutoff", "source_available_time", "entity_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Backtest frame missing fold columns: {', '.join(sorted(missing))}")
    if mode not in {"rolling", "gw1"}:
        raise ValueError("mode must be 'rolling' or 'gw1'.")

    players = frame.loc[frame["entity_type"] == "player"].copy()
    players["information_cutoff"] = pd.to_datetime(players["information_cutoff"], utc=True)
    players["source_available_time"] = pd.to_datetime(players["source_available_time"], utc=True)
    groups = (
        players.loc[players["season"].isin(test_seasons)]
        .groupby(["season", "gameweek"], as_index=False)
        .agg(information_cutoff=("information_cutoff", "min"), test_rows=("player_uid", "size"))
        .sort_values(["season", "gameweek"])
    )
    if mode == "gw1":
        groups = groups.loc[pd.to_numeric(groups["gameweek"], errors="coerce") == 1]

    folds: list[BacktestFold] = []
    ordered = add_season_order(players)
    for row in groups.itertuples(index=False):
        cutoff = pd.Timestamp(row.information_cutoff)
        train_mask = (
            (ordered["source_available_time"] < cutoff)
            & ~((ordered["season"] == row.season) & (ordered["gameweek"] == row.gameweek))
        )
        if mode == "gw1":
            train_mask &= ordered["season_order"] < season_sort_key(row.season)
        train_rows = int(train_mask.sum())
        if train_rows < minimum_training_rows:
            continue
        folds.append(
            BacktestFold(
                fold_id=f"{row.season}_GW{int(row.gameweek):02d}",
                season=str(row.season),
                gameweek=int(row.gameweek),
                information_cutoff=cutoff,
                train_rows=train_rows,
                test_rows=int(row.test_rows),
            )
        )
    return folds


def split_fold(frame: pd.DataFrame, fold: BacktestFold, *, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = frame.loc[frame["entity_type"] == "player"].copy()
    players["source_available_time"] = pd.to_datetime(players["source_available_time"], utc=True)
    test = players.loc[
        (players["season"] == fold.season) & (players["gameweek"] == fold.gameweek)
    ].copy()
    train = players.loc[
        (players["source_available_time"] < fold.information_cutoff)
        & ~((players["season"] == fold.season) & (players["gameweek"] == fold.gameweek))
    ].copy()
    if mode == "gw1":
        train = train.loc[train["season"].map(season_sort_key) < season_sort_key(fold.season)].copy()
    if not train["source_available_time"].lt(fold.information_cutoff).all():
        raise ValueError(f"Fold {fold.fold_id} contains training labels unavailable at cutoff.")
    return train, test
