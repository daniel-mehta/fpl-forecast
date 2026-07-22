from __future__ import annotations

import pandas as pd


def build_gameweek_cutoffs(
    fixtures: pd.DataFrame,
    *,
    exact_deadlines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"season", "gameweek", "kickoff_time"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"Fixture data missing cutoff columns: {', '.join(sorted(missing))}")
    frame = fixtures.copy()
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    if frame["kickoff_time"].isna().any():
        raise ValueError("Cannot infer cutoffs when fixture kickoff_time is missing or malformed.")

    inferred = (
        frame.groupby(["season", "gameweek"], as_index=False)
        .agg(information_cutoff=("kickoff_time", "min"))
        .assign(
            cutoff_method="inferred_earliest_gameweek_kickoff",
            cutoff_source="fixture.kickoff_time",
            cutoff_is_exact=False,
        )
    )
    if exact_deadlines is None or exact_deadlines.empty:
        return inferred

    deadlines = exact_deadlines.copy()
    required_deadlines = {"season", "gameweek", "deadline_time", "cutoff_source"}
    missing_deadlines = required_deadlines.difference(deadlines.columns)
    if missing_deadlines:
        raise ValueError(
            f"Exact deadline data missing columns: {', '.join(sorted(missing_deadlines))}"
        )
    deadlines["deadline_time"] = pd.to_datetime(deadlines["deadline_time"], utc=True, errors="coerce")
    if deadlines["deadline_time"].isna().any():
        raise ValueError("Exact deadline_time contains missing or malformed timestamps.")

    output = inferred.merge(
        deadlines[["season", "gameweek", "deadline_time", "cutoff_source"]],
        on=["season", "gameweek"],
        how="left",
    )
    exact = output["deadline_time"].notna()
    output.loc[exact, "information_cutoff"] = output.loc[exact, "deadline_time"]
    output.loc[exact, "cutoff_method"] = "exact_recorded_fpl_deadline"
    output.loc[exact, "cutoff_is_exact"] = True
    output = output.drop(columns=["deadline_time"])
    return output
