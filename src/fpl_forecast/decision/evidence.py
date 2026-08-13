from __future__ import annotations

import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


EVIDENCE_REGISTRY_PATH = Path(__file__).with_name("evidence_registry.json")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
D1_VARIANT = "D1_MEAN_ONLY_MILP"
D2_VARIANT = "D2_EXPECTED_REALIZED_POINTS"


def load_decision_evidence_registry(path: Path = EVIDENCE_REGISTRY_PATH) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 2:
        raise ValueError("Unsupported decision evidence registry schema.")
    xpoints_evidence = registry.get("xpoints_evidence")
    if not isinstance(xpoints_evidence, dict) or not xpoints_evidence:
        raise ValueError("Decision evidence registry is missing xPoints evidence scopes.")
    configurations = registry.get("decision_configurations")
    if not isinstance(configurations, dict) or not configurations:
        raise ValueError("Decision evidence registry is missing active decision configurations.")
    evidence = registry.get("decision_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"rolling_benchmark", "table7_gw1"}:
        raise ValueError("Decision evidence registry is missing required scopes.")
    authoritative = []
    superseded = []
    for scope, record in {**xpoints_evidence, **evidence}.items():
        if record.get("status") != "authoritative":
            raise ValueError(f"Evidence scope is not authoritative: {scope}")
        authoritative.append(str(record["authoritative_run_id"]))
        for item in record.get("supersedes", []):
            if item.get("preservation") != "immutable_historical_record":
                raise ValueError(f"Superseded evidence lacks immutable preservation status: {scope}")
            superseded.append(str(item["run_id"]))
    if len(authoritative) != len(set(authoritative)):
        raise ValueError("Authoritative decision run ids must be unique.")
    if set(authoritative).intersection(superseded):
        raise ValueError("An authoritative run is also marked superseded.")
    for config_path, record in configurations.items():
        if record.get("status") not in {"active_default", "active_correction_replay"}:
            raise ValueError(f"Decision configuration has unsupported status: {config_path}")
        scopes = record.get("xpoints_evidence")
        if not isinstance(scopes, dict) or set(scopes) != {"rolling", "gw1"}:
            raise ValueError(f"Decision configuration is missing xPoints scopes: {config_path}")
        unknown = set(scopes.values()).difference(xpoints_evidence)
        if unknown:
            raise ValueError(
                f"Decision configuration references unknown xPoints evidence: {config_path}: "
                f"{sorted(unknown)}"
            )
        divergences = record.get("documented_divergences")
        if not isinstance(divergences, dict):
            raise ValueError(f"Decision configuration divergences must be a mapping: {config_path}")
    return registry


def active_decision_configurations(
    path: Path = EVIDENCE_REGISTRY_PATH,
) -> dict[Path, dict[str, Any]]:
    registry = load_decision_evidence_registry(path)
    return {
        PROJECT_ROOT / config_path: dict(record)
        for config_path, record in registry["decision_configurations"].items()
    }


def authoritative_xpoints_run(scope: str, path: Path = EVIDENCE_REGISTRY_PATH) -> str:
    registry = load_decision_evidence_registry(path)
    try:
        return str(registry["xpoints_evidence"][scope]["authoritative_run_id"])
    except KeyError as exc:
        raise ValueError(f"Unknown xPoints evidence scope: {scope}") from exc


def superseded_evidence_run_ids(path: Path = EVIDENCE_REGISTRY_PATH) -> set[str]:
    registry = load_decision_evidence_registry(path)
    records = [
        *registry["xpoints_evidence"].values(),
        *registry["decision_evidence"].values(),
    ]
    return {
        str(item["run_id"])
        for record in records
        for item in record.get("supersedes", [])
    }


def authoritative_decision_run(scope: str, path: Path = EVIDENCE_REGISTRY_PATH) -> str:
    registry = load_decision_evidence_registry(path)
    try:
        return str(registry["decision_evidence"][scope]["authoritative_run_id"])
    except KeyError as exc:
        raise ValueError(f"Unknown decision evidence scope: {scope}") from exc


def publication_round(value: float, places: int = 2) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def build_decision_evidence_table(run_dir: Path, source_run_id: str) -> pd.DataFrame:
    return build_decision_evidence_table_from_frames(
        metrics=pd.read_csv(run_dir / "decision_metrics.csv"),
        comparison=pd.read_csv(run_dir / "model_comparison.csv"),
        scored=pd.read_csv(run_dir / "scored_decisions.csv"),
        source_run_id=source_run_id,
    )


def build_decision_evidence_table_from_frames(
    *,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    scored: pd.DataFrame,
    source_run_id: str,
) -> pd.DataFrame:
    metric_index = metrics.set_index("optimizer_variant")
    if not metric_index.index.is_unique:
        raise ValueError("Decision metrics contain duplicate optimizer variants.")
    missing_variants = {D1_VARIANT, D2_VARIANT}.difference(metric_index.index)
    if missing_variants:
        raise ValueError(f"Decision evidence is missing variants: {sorted(missing_variants)}")
    if len(comparison) != 1:
        raise ValueError("Decision evidence must contain one D2-versus-D1 comparison row.")
    comparison_row = comparison.iloc[0]
    if not (
        str(comparison_row["left_model"]).endswith(f":{D2_VARIANT}")
        and str(comparison_row["right_model"]).endswith(f":{D1_VARIANT}")
    ):
        raise ValueError("Decision comparison is not ordered as D2 minus D1.")

    differences = paired_realized_differences(scored)
    mean_difference = sum(differences) / len(differences)
    if not math.isclose(
        mean_difference,
        float(comparison_row["mean_realized_difference"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Paired fold differences do not reconcile to model_comparison.csv.")
    fold_differences = json.dumps([_plain_number(value) for value in differences])

    rows = []
    for variant in (D2_VARIANT, D1_VARIANT):
        metric = metric_index.loc[variant]
        rows.append(
            {
                "decision_model": variant,
                "gameweeks": int(metric["decisions"]),
                "mean_expected_realized_points": float(metric["mean_expected_score"]),
                "mean_realized_points": float(metric["mean_realized_points"]),
                "mean_autosub_points": float(metric["mean_autosub_points"]),
                "unreplaced_rate": float(metric["unreplaced_starter_rate"]),
                "captain_agreement_d1_vs_d2": float(comparison_row["captain_agreement"]),
                "mean_lineup_overlap_d1_vs_d2": float(comparison_row["mean_lineup_overlap"]),
                "d2_minus_d1_fold_realized_differences": fold_differences,
                "d2_minus_d1_mean_realized_difference": mean_difference,
                "d2_minus_d1_mean_realized_difference_2dp": publication_round(mean_difference),
                "paired_difference_ci_low": float(comparison_row["bootstrap_ci_low"]),
                "paired_difference_ci_high": float(comparison_row["bootstrap_ci_high"]),
                "source_run_id": source_run_id,
                "population": "weekly_reset_historical_gw1_decisions",
                "grain": "one_decision_per_gameweek",
                "limitation": "Only three GW1 decisions; interval is descriptive and very unstable.",
            }
        )
    return pd.DataFrame(rows)


def paired_realized_differences(scored: pd.DataFrame) -> list[float]:
    keys = ["season", "gameweek"]
    left = scored.loc[scored["optimizer_variant"].eq(D2_VARIANT), keys + ["realized_points"]]
    right = scored.loc[scored["optimizer_variant"].eq(D1_VARIANT), keys + ["realized_points"]]
    merged = left.merge(right, on=keys, suffixes=("_d2", "_d1"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("D1 and D2 scored decisions do not form a complete paired set.")
    merged = merged.sort_values(keys)
    return (merged["realized_points_d2"] - merged["realized_points_d1"]).astype(float).tolist()


def decision_evidence_supersession_table(path: Path = EVIDENCE_REGISTRY_PATH) -> pd.DataFrame:
    registry = load_decision_evidence_registry(path)
    rows = []
    for scope, record in registry["decision_evidence"].items():
        for item in record["supersedes"]:
            rows.append(
                {
                    "evidence_scope": scope,
                    "authoritative_run_id": record["authoritative_run_id"],
                    "superseded_run_id": item["run_id"],
                    "superseded_status": item["preservation"],
                    "reason": item["reason"],
                }
            )
    return pd.DataFrame(rows)


def _plain_number(value: float) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric
