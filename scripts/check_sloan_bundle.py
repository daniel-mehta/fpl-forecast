#!/usr/bin/env python3
"""Validate the rights-safe aggregate paper bundle without running research models."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURE1 = "paper/figures/figure1_v7_player_points_pipeline.png"
FIGURE1_SHA256 = "06c05fdd4bacb9301dedda1f3f59d1f00ae933d6dde062467e85b2811c9df896"
PAPER_METADATA = {
    "paper/FIGURE_NOTES.md",
    "paper/evidence_manifest.csv",
    "paper/evidence_supersession.csv",
    "paper/manuscript_value_map.csv",
    "paper/output_hashes.csv",
    "paper/prospective_publication_registry.csv",
    "paper/research_schema.csv",
    "paper/source_input_manifest.csv",
}
PAPER_TABLES = {
    "paper/tables/figure7_p5_calibration_bins.csv",
    *(
        f"paper/tables/table{i}_{slug}.csv"
        for i, slug in {
            1: "historical_data_coverage",
            2: "team_model_comparison",
            3: "minutes_model_comparison",
            4: "xpoints_comparison",
            5: "hybrid_simulation_convergence",
            6: "old_vs_hybrid_gw1_folds",
            7: "decision_system_evidence",
        }.items()
    ),
}
PAPER_FIGURES = {
    FIGURE1,
    *(
        f"paper/figures/figure{i}_{slug}.svg"
        for i, slug in {
            1: "system_architecture",
            2: "evaluation_timeline",
            3: "team_model_comparison",
            4: "xpoints_comparison",
            5: "simulation_convergence",
            6: "simulator_tradeoffs",
            7: "p5_calibration",
        }.items()
    ),
}
PAPER_ALLOWLIST = PAPER_METADATA | PAPER_TABLES | PAPER_FIGURES
REQUIRED_CLAIMS = {
    "historical-panel",
    "team-folds",
    "team-goal-mae",
    "team-bootstrap-ci",
    "team-outcome-logloss",
    "xpoints-76-fold",
    "fcxp-gw1",
    "sim-closure",
    "d1-optimality",
    "d2-comparison",
}
AGGREGATE_ROW_COUNTS = {
    "figure7_p5_calibration_bins.csv": 13,
    "table1_historical_data_coverage.csv": 4,
    "table2_team_model_comparison.csv": 4,
    "table3_minutes_model_comparison.csv": 6,
    "table4_xpoints_comparison.csv": 11,
    "table5_hybrid_simulation_convergence.csv": 7,
    "table6_old_vs_hybrid_gw1_folds.csv": 8,
    "table7_decision_system_evidence.csv": 2,
}
ROW_LEVEL_COLUMNS = {
    "player",
    "player_name",
    "player_uid",
    "stable_player_id",
    "fixture_id",
    "price_tenths",
    "recommended_squad",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_aggregate_table(path: Path, expected_rows: int) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or ROW_LEVEL_COLUMNS.intersection(reader.fieldnames):
            raise ValueError(f"Row-level columns in aggregate table: {path.name}")
        if sum(1 for _ in reader) != expected_rows:
            raise ValueError(f"Unexpected aggregate row count: {path.name}")


def validate_public_paths(paths: set[str]) -> None:
    forbidden = set()
    for path in paths:
        if path.startswith("paper/") and path not in PAPER_ALLOWLIST:
            forbidden.add(path)
        if path.startswith(("data/raw/", "data/normalized/", "reports/operational/")):
            if not path.endswith("/.gitkeep"):
                forbidden.add(path)
    if forbidden:
        raise ValueError(f"Restricted or unreviewed public paths: {', '.join(sorted(forbidden))}")


def public_git_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {
        name
        for part in result.stdout.split(b"\0")
        if part
        for name in [part.decode()]
        if (ROOT / name).exists()
    }


def check_ignore_policy() -> None:
    def ignored(path: str) -> bool:
        return (
            subprocess.run(
                ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, check=False
            ).returncode
            == 0
        )

    if ignored("paper/new_reviewed_asset.csv"):
        raise ValueError("New paper assets are still hidden by .gitignore")
    if not ignored("paper/manuscript_draft.docx"):
        raise ValueError("Temporary manuscript editing files must stay ignored")
    if not ignored("data/raw/vaastav/restricted.csv"):
        raise ValueError("Raw third-party inputs must stay ignored")


def check_paper_assets() -> None:
    for name in PAPER_ALLOWLIST:
        if not (ROOT / name).is_file():
            raise ValueError(f"Missing reviewed paper asset: {name}")
    if sha256(ROOT / FIGURE1) != FIGURE1_SHA256:
        raise ValueError("The v7 embedded Figure 1 bytes changed")
    for name, count in AGGREGATE_ROW_COUNTS.items():
        validate_aggregate_table(PAPER / "tables" / name, count)

    rows = read_csv(PAPER / "output_hashes.csv")
    expected = PAPER_ALLOWLIST - {"paper/output_hashes.csv"}
    found = [row["path"] for row in rows]
    if set(found) != expected or len(found) != len(expected):
        raise ValueError("Output hash inventory does not cover each reviewed paper asset once")
    for row in rows:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Output hash mismatch: {row['path']}")

    evidence = read_csv(PAPER / "evidence_manifest.csv")
    evidence_ids = {row["evidence_id"] for row in evidence}
    if len(evidence_ids) != len(evidence) or not {"E-M01", "E-M02", "E-F01"} <= evidence_ids:
        raise ValueError("Missing or duplicated manuscript evidence mapping")
    if any(row["figure_or_table_id"] in {"Table 8", "Figure 8"} for row in evidence):
        raise ValueError("Restricted prospective assets remain in the evidence manifest")
    xpoints = next(row for row in evidence if row["evidence_id"] == "E-M01")
    decision = next(row for row in evidence if row["evidence_id"] == "E-M02")
    if xpoints["source_run_id"] != "phase6_xpoints_rolling_goalkeeper_corrected_exact":
        raise ValueError("76-fold xPoints claim uses the wrong run")
    if decision["source_run_id"] != (
        "phase7_goalkeeper_scoring_corrected_decisions_rolling_real_clean_034830b041c1"
    ):
        raise ValueError("380-decision D1 claim uses the wrong run")
    for row in (xpoints, decision):
        retained = ROOT / row["source_path"]
        expected = row["source_output_sha256"].split("=", 1)[1]
        if retained.is_file() and sha256(retained) != expected:
            raise ValueError(f"Retained evidence hash mismatch: {row['evidence_id']}")
    table4 = read_csv(PAPER / "tables/table4_xpoints_comparison.csv")
    if {row["folds"] for row in table4 if row["comparison_block"] == "rolling_legacy_80"} != {
        "114"
    }:
        raise ValueError("Figure 4 rolling scope must remain separate from v7's 76-fold claim")

    claims = read_csv(PAPER / "manuscript_value_map.csv")
    if len({row["claim_id"] for row in claims}) != len(claims):
        raise ValueError("Duplicate manuscript claim IDs")
    if not REQUIRED_CLAIMS <= {row["claim_id"] for row in claims}:
        raise ValueError("Headline manuscript claim mapping is incomplete")
    for row in claims:
        artifact = row["public_artifact"].split("#", 1)[0]
        if artifact not in PAPER_ALLOWLIST:
            raise ValueError(f"Claim has no reviewed public artifact: {row['claim_id']}")
        if row["evidence_id"] not in evidence_ids:
            raise ValueError(f"Claim has no evidence manifest row: {row['claim_id']}")
        if sha256(ROOT / artifact) != row["public_artifact_sha256"]:
            raise ValueError(f"Claim artifact hash mismatch: {row['claim_id']}")

    inputs = read_csv(PAPER / "source_input_manifest.csv")
    if len({row["input_id"] for row in inputs}) != len(inputs):
        raise ValueError("Duplicate source input IDs")
    for row in inputs:
        if not re.fullmatch(r"[0-9a-f]{64}", row["expected_sha256"]):
            raise ValueError(f"Invalid source hash: {row['input_id']}")
        if row["origin"] in {"Vaastav / FPL", "Official FPL API"} and not row[
            "source_url"
        ].startswith("https://"):
            raise ValueError(f"Unpinned upstream URL: {row['input_id']}")
        if row["input_id"].startswith("VAASTAV-") and row["upstream_revision"] != (
            "f2090d378ebd1b0c3d14884770dde95f38c50a0d"
        ):
            raise ValueError(f"Unexpected Vaastav revision: {row['input_id']}")

    registry = read_csv(PAPER / "prospective_publication_registry.csv")
    if [row["gameweek"] for row in registry] != [str(i) for i in range(1, 7)]:
        raise ValueError("Prospective publication registry must cover GW1-GW6")
    for row in registry:
        if row["bundle_freeze_at_utc"] >= row["deadline_utc"]:
            raise ValueError(f"Late frozen publication: GW{row['gameweek']}")


def main() -> None:
    validate_public_paths(public_git_paths())
    check_ignore_policy()
    check_paper_assets()
    print("Sloan aggregate bundle policy, mappings, assets, and hashes: PASS")


if __name__ == "__main__":
    main()
