from __future__ import annotations

import runpy
from pathlib import Path

import pytest


bundle = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_sloan_bundle.py"))
check_ignore_policy = bundle["check_ignore_policy"]
check_paper_assets = bundle["check_paper_assets"]
public_git_paths = bundle["public_git_paths"]
validate_public_paths = bundle["validate_public_paths"]
validate_aggregate_table = bundle["validate_aggregate_table"]


def test_aggregate_bundle_is_complete_and_hashed() -> None:
    validate_public_paths(public_git_paths())
    check_paper_assets()
    check_ignore_policy()


@pytest.mark.parametrize(
    "path",
    [
        "paper/tables/table8_prospective_gw1_snapshot.csv",
        "paper/figures/figure8_prospective_price_xpoints.svg",
        "paper/tables/player_predictions.parquet",
        "data/raw/vaastav/2024-25/merged_gw.csv",
        "data/normalized/phase2/fact_player_fixture.parquet",
    ],
)
def test_restricted_research_data_cannot_enter_public_bundle(path: str) -> None:
    with pytest.raises(ValueError, match="Restricted or unreviewed public paths"):
        validate_public_paths({path})


def test_row_level_columns_are_rejected_at_an_aggregate_path(tmp_path: Path) -> None:
    table = tmp_path / "table1_historical_data_coverage.csv"
    table.write_text("season,player_uid\n2024-25,p1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Row-level columns"):
        validate_aggregate_table(table, 1)
