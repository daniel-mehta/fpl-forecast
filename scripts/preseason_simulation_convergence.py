from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_forecast.xpoints.config import load_xpoints_config
from fpl_forecast.xpoints.simulation import simulate_component_points


DRAW_COUNTS = (80, 500, 1_000, 5_000, 10_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.artifact)
    frame = frame.loc[frame["model_name"].eq("X2_TEAM_CONSTRAINED_SIM_M7")].copy()
    frame["defensive_contribution_threshold"] = frame["fpl_position"].map(
        {"DEF": 10, "MID": 12, "FWD": 12}
    ).fillna(10**9)
    frame["defensive_contribution_points"] = np.where(
        frame["defensive_contribution_threshold"].lt(10**9),
        2,
        0,
    )
    team_keys = ["season", "stable_fixture_uid", "player_team_uid"]
    assisted = (
        frame.groupby(team_keys, as_index=False)
        .agg(expected_assists=("expected_assists", "sum"), team_xg=("team_expected_goals", "first"))
    )
    assisted["assisted_goal_rate"] = np.divide(
        assisted["expected_assists"],
        assisted["team_xg"],
        out=np.zeros(len(assisted)),
        where=assisted["team_xg"].gt(0),
    ).clip(0, 1)
    frame = frame.merge(assisted[[*team_keys, "assisted_goal_rate"]], on=team_keys, how="left")

    base_config = load_xpoints_config()
    results: dict[int, dict[str, object]] = {}
    summaries: dict[int, pd.DataFrame] = {}
    for draw_count in DRAW_COUNTS:
        config = replace(base_config, draw_count=draw_count)
        started = time.perf_counter()
        summary, draws = simulate_component_points(
            frame,
            config=config,
            seed=base_config.random_seed + 4,
            seed_namespace="X2_TEAM_CONSTRAINED_SIM_M7",
        )
        elapsed = time.perf_counter() - started
        summaries[draw_count] = summary
        rerun, rerun_draws = simulate_component_points(
            frame,
            config=config,
            seed=base_config.random_seed + 4,
            seed_namespace="X2_TEAM_CONSTRAINED_SIM_M7",
        )
        results[draw_count] = {
            "runtime_seconds": elapsed,
            "draw_matrix_megabytes": draws.nbytes / (1024**2),
            "process_peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "reproducible_summary": bool(summary.equals(rerun)),
            "reproducible_draws": bool(np.array_equal(draws, rerun_draws)),
            "analytic_vs_simulated_mean_median_abs": float(
                (summary["expected_points"] - summary["simulated_expected_points"]).abs().median()
            ),
            "analytic_vs_simulated_mean_p95_abs": float(
                (summary["expected_points"] - summary["simulated_expected_points"]).abs().quantile(0.95)
            ),
            "analytic_vs_simulated_mean_max_abs": float(
                (summary["expected_points"] - summary["simulated_expected_points"]).abs().max()
            ),
            "component_reconciliation_max_abs": float(summary["component_reconciliation_error"].abs().max()),
        }

    reference = summaries[10_000]
    for draw_count, summary in summaries.items():
        mean_diff = (summary["expected_points"] - reference["expected_points"]).abs()
        p5_diff = (summary["prob_points_ge_5"] - reference["prob_points_ge_5"]).abs()
        zero_diff = (summary["prob_points_eq_0"] - reference["prob_points_eq_0"]).abs()
        interval_diff = pd.concat(
            [
                (summary["points_p10"] - reference["points_p10"]).abs(),
                (summary["points_p90"] - reference["points_p90"]).abs(),
            ],
            ignore_index=True,
        )
        conditional_diff = (
            summary["expected_points_given_appearance"] - reference["expected_points_given_appearance"]
        ).abs()
        results[draw_count].update(
            {
                "reference_mean_median_abs": float(mean_diff.median()),
                "reference_mean_p95_abs": float(mean_diff.quantile(0.95)),
                "reference_mean_max_abs": float(mean_diff.max()),
                "reference_p5_p95_abs": float(p5_diff.quantile(0.95)),
                "reference_zero_p95_abs": float(zero_diff.quantile(0.95)),
                "reference_interval_endpoint_p95_abs": float(interval_diff.quantile(0.95)),
                "reference_conditional_p95_abs": float(conditional_diff.quantile(0.95)),
                "rank_spearman": float(summary["expected_points"].corr(reference["expected_points"], method="spearman")),
                "top_15_overlap": _top_overlap(summary, reference, 15),
                "top_30_overlap": _top_overlap(summary, reference, 30),
                "top_50_overlap": _top_overlap(summary, reference, 50),
                "decision_inputs_identical": bool(
                    summary[["expected_points", "expected_points_given_appearance"]].equals(
                        reference[["expected_points", "expected_points_given_appearance"]]
                    )
                ),
            }
        )
    print(json.dumps({"rows": len(frame), "reference_draw_count": 10_000, "results": results}, indent=2))


def _top_overlap(left: pd.DataFrame, right: pd.DataFrame, k: int) -> float:
    left_ids = set(left.nlargest(k, "expected_points").index)
    right_ids = set(right.nlargest(k, "expected_points").index)
    return len(left_ids & right_ids) / k


if __name__ == "__main__":
    main()
