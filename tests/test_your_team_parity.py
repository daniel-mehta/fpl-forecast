from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_forecast.decision.expected_realized import (
    optimize_lineup_expected_realized,
    refine_fixed_squad_lineup,
)
from fpl_forecast.decision.rules import default_rules


FIXTURE_PATH = Path("frontend/src/fixtures/your_team_parity.json")


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"],
    ids=lambda case: case["name"],
)
def test_python_authority_matches_shared_your_team_fixture(case: dict[str, object]) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    rows = []
    for base in fixture["base"]:
        rows.append(
            {
                **base,
                **case.get("all", {}),
                **case.get("overrides", {}).get(base["player_uid"], {}),
            }
        )
    squad = pd.DataFrame(rows)
    initial, _ = optimize_lineup_expected_realized(squad, default_rules(), shortlist=1)
    decision, breakdown, _ = refine_fixed_squad_lineup(squad, initial, default_rules())
    expected = case["expected"]
    tolerance = float(fixture["tolerance"])

    assert list(decision.lineup) == expected["lineup"]
    assert decision.formation == expected["formation"]
    assert decision.bench[0] == expected["bench_goalkeeper"]
    assert list(decision.bench[1:]) == expected["outfield_bench"]
    assert decision.captain == expected["captain"]
    assert decision.vice_captain == expected["vice_captain"]
    assert breakdown.expected_realized_total == pytest.approx(
        expected["expected_realized_total"], abs=tolerance
    )
    assert breakdown.expected_autosub_contribution == pytest.approx(
        expected["expected_autosub_contribution"], abs=tolerance
    )
    assert breakdown.expected_captain_bonus == pytest.approx(
        expected["expected_captain_bonus"], abs=tolerance
    )
    assert breakdown.expected_vice_captain_contingency == pytest.approx(
        expected["expected_vice_captain_contingency"], abs=tolerance
    )
    assert breakdown.expected_automatic_substitutions == pytest.approx(
        expected["expected_automatic_substitutions"], abs=tolerance
    )
    assert breakdown.probability_unreplaced_starter == pytest.approx(
        expected["probability_unreplaced_starter"], abs=tolerance
    )
