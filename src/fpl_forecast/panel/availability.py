from __future__ import annotations

import pandas as pd


CONSERVATIVE_MATCH_RESULT_DELAY = pd.Timedelta(hours=3)
CONSERVATIVE_AVAILABILITY_METHOD = "kickoff_plus_3h_conservative_match_completion"


def match_result_available_time(
    kickoff_time: pd.Series | pd.Timestamp,
    *,
    exact_completion_time: pd.Series | pd.Timestamp | None = None,
) -> pd.Series | pd.Timestamp:
    if exact_completion_time is not None:
        exact = pd.to_datetime(exact_completion_time, utc=True, errors="coerce")
        if isinstance(exact, pd.Series) and exact.notna().all():
            return exact
        if isinstance(exact, pd.Timestamp) and pd.notna(exact):
            return exact
    kickoff = pd.to_datetime(kickoff_time, utc=True, errors="coerce")
    return kickoff + CONSERVATIVE_MATCH_RESULT_DELAY
