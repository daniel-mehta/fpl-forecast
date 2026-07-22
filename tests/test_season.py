from __future__ import annotations

import pytest

from fpl_forecast.ingest.season import (
    SeasonIdentityError,
    infer_and_validate_current_season,
    parse_season_label,
)


def test_parse_season_label_boundary_years():
    assert parse_season_label("2025-26") == (2025, 2026)


def test_infer_season_from_deadlines_and_kickoffs():
    identity = infer_and_validate_current_season(
        requested_season="2025-26",
        bootstrap_payload=_standard_bootstrap(),
        fixtures_payload=_standard_fixtures(),
    )

    assert identity.inferred_season == "2025-26"
    assert identity.first_fixture_kickoff == "2025-08-15T19:00:00Z"
    assert identity.last_fixture_kickoff == "2026-05-24T15:00:00Z"


def test_infer_season_rejects_mismatch():
    with pytest.raises(SeasonIdentityError, match="conflicts"):
        infer_and_validate_current_season(
            requested_season="2026-27",
            bootstrap_payload=_standard_bootstrap(),
            fixtures_payload=_standard_fixtures(),
        )


def test_infer_season_rejects_missing_dates():
    bootstrap = _standard_bootstrap()
    bootstrap["events"][0].pop("deadline_time")

    with pytest.raises(SeasonIdentityError, match="missing"):
        infer_and_validate_current_season(
            requested_season="2025-26",
            bootstrap_payload=bootstrap,
            fixtures_payload=_standard_fixtures(),
        )


def _standard_bootstrap() -> dict:
    return {
        "events": [
            {"id": gameweek, "deadline_time": f"2025-08-{min(10 + gameweek, 28):02d}T11:00:00Z"}
            for gameweek in range(1, 39)
        ],
        "teams": [{"id": team_id} for team_id in range(1, 21)],
    }


def _standard_fixtures() -> list[dict]:
    fixtures = []
    fixture_id = 1
    for gameweek in range(1, 39):
        for match in range(10):
            fixtures.append(
                {
                    "id": fixture_id,
                    "event": gameweek,
                    "team_h": (match % 20) + 1,
                    "team_a": ((match + 1) % 20) + 1,
                    "kickoff_time": "2025-08-15T19:00:00Z"
                    if gameweek == 1
                    else "2026-05-24T15:00:00Z"
                    if gameweek == 38
                    else "2025-12-15T15:00:00Z",
                }
            )
            fixture_id += 1
    return fixtures
