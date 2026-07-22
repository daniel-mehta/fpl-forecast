# Phase 3 Backtesting Baselines Report

## 1. Phase Scope And Non-Goals

Phase 3 implements an honest rolling-origin backtesting framework and simple baseline
forecast families. It does not implement predictive models, team-strength models,
expected-minutes ML, component models, Monte Carlo simulation, a scoring engine,
squad optimization, transfer planning, a dashboard, or a scheduler.

The audited seasons are `2022-23`, `2023-24`, and `2024-25`. Seasons `2023-24` and
`2024-25` are historical validation seasons, not an untouched final test set.

## 2. Fold Design

Backtest folds are chronological season-gameweek blocks. For a test fold, training
labels are admitted only when `source_available_time < information_cutoff`, and the
same season-gameweek block is excluded from training. Assistant managers are excluded
from standard player backtests through the Phase 2 `entity_type == "player"` field.

Real rolling run `phase3_rolling_hardened` produced 76 folds across `2023-24` and
`2024-25`, with cutoff range `2023-08-11T19:00:00+00:00` to
`2025-05-25T15:00:00+00:00`. Training rows ranged from 26,505 to 82,729 per fold;
test rows ranged from 344 to 1,132 standard-player fixture rows.

## 3. GW1 Design

Dedicated GW1 mode tests cold-start behavior at the season boundary. It trains only
on earlier seasons and scores only gameweek 1 of the requested test seasons.

Real GW1 run `phase3_gw1_hardened` produced 2 folds, one each for `2023-24` and
`2024-25`. Training rows ranged from 26,505 to 56,230; test rows ranged from 616 to
658.

## 4. Baselines

Implemented baselines:

- `B0_ZERO`: constant zero points.
- `B1_GLOBAL_MEAN`: historical global mean target points.
- `B2_POSITION_MEAN`: historical mean by current-season FPL position, with global fallback.
- `B3_RECENT_POINTS_P3` and `B3_RECENT_POINTS_P5`: shifted recent points averages.
- `B4_RECENT_MINUTES_P3` and `B4_RECENT_MINUTES_P5`: shifted recent minutes converted through historical points per 90.
- `B5_EB_POINTS_PER90`: empirical-Bayes shrunk points-per-90 using same-season lagged points and minutes, with `prior_matches=5`.
- `B6_PREVIOUS_SEASON_GW1`: previous-season shrunk points-per-90 rate times a transparent previous-season minutes heuristic.

All baselines use fixed, versioned parameters in
`src/fpl_forecast/backtest/config.json`.

## 5. B6 Formula

`B6_PREVIOUS_SEASON_GW1` is not a tuned model. For player `i` entering season `s`,
using the immediately previous season `s-1`:

```text
global_points_per90 = 90 * sum(training_points) / sum(training_minutes)
prior_minutes = points_per_90_prior_matches * 90
shrunk_player_points_per90 =
  (90 * previous_season_player_points + global_points_per90 * prior_minutes)
  / (previous_season_player_minutes + prior_minutes)
expected_minutes =
  min(minutes_cap, previous_season_player_minutes / previous_season_player_appearances)
B6 = expected_minutes / 90 * shrunk_player_points_per90
```

If a player has no previous-season minutes or appearances, B6 falls back to the
training global mean. The formula is deliberately transparent and is used for
returning, transferred, and position-change players through stable `player_uid`;
current-season position and team context remain season-specific.

## 6. Availability Decisions

The backtester consumes Phase 2 features only after the Phase 2 leakage audit passes.
It excludes FPL expected-points fields (`xP`/expected points), unverified form fields,
and target-row price fields from baseline inputs. Current-season position, team, and
fixture context are season-specific and come from the Phase 2 panel.

## 7. Evaluation Populations

Primary metrics use all observed standard-player player-gameweek rows. Phase 3 now
also reports a decision-relevant `pre_deadline_history_active` population using only
lagged information:

- In-season folds: `prev5_minutes_sum > 0` or `season_to_date_appearances > 0` before the cutoff.
- GW1 folds: `prior_season_minutes > 0` or `prior_season_appearances > 0`.
- `cold_start_no_history`: no same-season lagged activity and no prior-season minutes or appearances.
- `actual_appearances_diagnostic`: target player-gameweek minutes greater than zero, reported only as a retrospective diagnostic.

Coverage by season in rolling validation:

| Season | All observed | History-active | Cold-start no history | Actual appearances |
| --- | ---: | ---: | ---: | ---: |
| 2023-24 | 28,742 | 17,999 | 7,446 | 11,086 |
| 2024-25 | 26,919 | 18,179 | 5,409 | 11,431 |

GW1 coverage:

| Season | All observed | History-active | Cold-start no history | Actual appearances |
| --- | ---: | ---: | ---: | ---: |
| 2023-24 GW1 | 658 | 403 | 255 | 302 |
| 2024-25 GW1 | 616 | 410 | 206 | 307 |

Phase 2 source counts:

| Season | Players | Assistant managers |
| --- | ---: | ---: |
| 2022-23 | 26,505 | 0 |
| 2023-24 | 29,725 | 0 |
| 2024-25 | 27,283 | 322 |

## 8. Overall Metrics

Rolling player-gameweek metrics, sorted by MAE:

| Baseline | Rows | MAE | RMSE | Bias | Median AE | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B5_EB_POINTS_PER90 | 55,661 | 1.0060 | 2.0528 | -0.0980 | 0.1547 | 0.6740 |
| B4_RECENT_MINUTES_P3 | 55,661 | 1.0169 | 2.0511 | -0.0551 | 0.0591 | 0.6944 |
| B4_RECENT_MINUTES_P5 | 55,661 | 1.0234 | 2.0643 | -0.0849 | 0.1669 | 0.6758 |
| B0_ZERO | 55,661 | 1.1386 | 2.6159 | -1.1240 | 0.0000 | n/a |
| B3_RECENT_POINTS_P5 | 55,661 | 1.4529 | 2.2424 | 0.4500 | 1.1341 | 0.4493 |
| B3_RECENT_POINTS_P3 | 55,661 | 1.4543 | 2.3009 | 0.4221 | 1.1322 | 0.4168 |
| B2_POSITION_MEAN | 55,661 | 1.4753 | 2.3519 | 0.0540 | 1.0751 | 0.0885 |
| B1_GLOBAL_MEAN | 55,661 | 1.4818 | 2.3568 | 0.0528 | 1.1349 | -0.0068 |
| B6_PREVIOUS_SEASON_GW1 | 55,661 | 1.7880 | 2.4170 | 0.8312 | 1.1736 | 0.3414 |

GW1 player-gameweek metrics:

| Baseline | Rows | MAE | RMSE | Bias | Median AE | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_ZERO | 1,274 | 1.2818 | 2.5895 | -1.2692 | 0.0000 | n/a |
| B4_RECENT_MINUTES_P3 | 1,274 | 1.2818 | 2.5895 | -1.2692 | 0.0000 | n/a |
| B4_RECENT_MINUTES_P5 | 1,274 | 1.2818 | 2.5895 | -1.2692 | 0.0000 | n/a |
| B5_EB_POINTS_PER90 | 1,274 | 1.2818 | 2.5895 | -1.2692 | 0.0000 | n/a |
| B2_POSITION_MEAN | 1,274 | 1.4531 | 2.2573 | -0.1073 | 1.0828 | 0.0779 |
| B1_GLOBAL_MEAN | 1,274 | 1.4577 | 2.2609 | -0.1094 | 1.1203 | -0.0354 |
| B3_RECENT_POINTS_P3 | 1,274 | 1.4577 | 2.2609 | -0.1094 | 1.1203 | -0.0354 |
| B3_RECENT_POINTS_P5 | 1,274 | 1.4577 | 2.2609 | -0.1094 | 1.1203 | -0.0354 |
| B6_PREVIOUS_SEASON_GW1 | 1,274 | 1.6797 | 2.2003 | 0.7715 | 1.1969 | 0.4238 |

## 9. Pre-Deadline Candidate Metrics

Rolling `pre_deadline_history_active` metrics:

| Baseline | Rows | MAE | RMSE | Bias | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| B5_EB_POINTS_PER90 | 36,178 | 1.5170 | 2.5233 | -0.1204 | 0.5420 |
| B4_RECENT_MINUTES_P3 | 36,178 | 1.5338 | 2.5212 | -0.0543 | 0.5757 |
| B4_RECENT_MINUTES_P5 | 36,178 | 1.5438 | 2.5377 | -0.1002 | 0.5448 |
| B3_RECENT_POINTS_P5 | 36,178 | 1.6050 | 2.6318 | 0.0876 | 0.5392 |
| B3_RECENT_POINTS_P3 | 36,178 | 1.6071 | 2.7084 | 0.0448 | 0.5562 |
| B1_GLOBAL_MEAN | 36,178 | 1.6495 | 2.7814 | -0.5233 | 0.0221 |
| B2_POSITION_MEAN | 36,178 | 1.6499 | 2.7771 | -0.5108 | 0.0763 |
| B0_ZERO | 36,178 | 1.7210 | 3.2267 | -1.6988 | n/a |
| B6_PREVIOUS_SEASON_GW1 | 36,178 | 2.0204 | 2.7617 | 0.5711 | 0.2444 |

Rolling `pre_deadline_history_active` top-k:

| Baseline | Top 15 points | Top 15 overlap | Top 30 points | Top 30 overlap |
| --- | ---: | ---: | ---: | ---: |
| B5_EB_POINTS_PER90 | 67.37 | 0.137 | 122.17 | 0.196 |
| B3_RECENT_POINTS_P5 | 65.71 | 0.132 | 120.16 | 0.193 |
| B4_RECENT_MINUTES_P3 | 61.29 | 0.118 | 117.71 | 0.193 |
| B4_RECENT_MINUTES_P5 | 61.70 | 0.125 | 116.72 | 0.191 |
| B3_RECENT_POINTS_P3 | 62.28 | 0.118 | 114.75 | 0.171 |
| B6_PREVIOUS_SEASON_GW1 | 63.36 | 0.148 | 105.99 | 0.192 |
| B2_POSITION_MEAN | 31.54 | 0.049 | 66.67 | 0.112 |
| B1_GLOBAL_MEAN | 27.16 | 0.033 | 56.84 | 0.074 |
| B0_ZERO | 23.74 | 0.028 | 49.87 | 0.061 |

GW1 `pre_deadline_history_active` metrics:

| Baseline | Rows | MAE | RMSE | Bias | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1_GLOBAL_MEAN | 813 | 1.6436 | 2.5927 | -0.4949 | -0.0025 |
| B3_RECENT_POINTS_P3 | 813 | 1.6436 | 2.5927 | -0.4949 | -0.0025 |
| B3_RECENT_POINTS_P5 | 813 | 1.6436 | 2.5927 | -0.4949 | -0.0025 |
| B2_POSITION_MEAN | 813 | 1.6461 | 2.5921 | -0.4847 | 0.0311 |
| B0_ZERO | 813 | 1.6679 | 3.0344 | -1.6531 | n/a |
| B4_RECENT_MINUTES_P3 | 813 | 1.6679 | 3.0344 | -1.6531 | n/a |
| B4_RECENT_MINUTES_P5 | 813 | 1.6679 | 3.0344 | -1.6531 | n/a |
| B5_EB_POINTS_PER90 | 813 | 1.6679 | 3.0344 | -1.6531 | n/a |
| B6_PREVIOUS_SEASON_GW1 | 813 | 1.9915 | 2.5097 | 0.8855 | 0.4247 |

GW1 `pre_deadline_history_active` top-k is operationally most useful for B6:
top-15 actual points 63.5 with 0.167 overlap, and top-30 actual points 114.0
with 0.217 overlap.

## 10. Calibration

Representative rolling calibration bins:

| Baseline | Bin | Rows | Mean prediction | Mean actual |
| --- | ---: | ---: | ---: | ---: |
| B4_RECENT_MINUTES_P3 | 0 | 5,567 | 0.000 | 0.234 |
| B4_RECENT_MINUTES_P3 | 5 | 5,566 | 0.189 | 0.740 |
| B4_RECENT_MINUTES_P3 | 7 | 5,566 | 2.192 | 2.066 |
| B4_RECENT_MINUTES_P3 | 9 | 5,566 | 4.147 | 3.534 |
| B5_EB_POINTS_PER90 | 0 | 5,567 | 0.000 | 0.218 |
| B5_EB_POINTS_PER90 | 5 | 5,566 | 0.348 | 0.947 |
| B5_EB_POINTS_PER90 | 7 | 5,566 | 1.921 | 1.945 |
| B5_EB_POINTS_PER90 | 9 | 5,566 | 4.157 | 3.612 |
| B6_PREVIOUS_SEASON_GW1 | 0 | 5,567 | 0.728 | 0.376 |
| B6_PREVIOUS_SEASON_GW1 | 5 | 5,566 | 1.570 | 0.997 |
| B6_PREVIOUS_SEASON_GW1 | 7 | 5,566 | 2.708 | 1.537 |
| B6_PREVIOUS_SEASON_GW1 | 9 | 5,566 | 4.470 | 2.849 |

B4 and B5 underpredict the lower-middle bins and overpredict the top bin. B5 is
near calibrated around bin 7. B6 overpredicts across the displayed bins, which is
consistent with its positive rolling bias of 0.8312.

## 11. Ranking Metrics

Rolling top-k summary for all observed standard players:

| Baseline | Top 15 points | Top 15 overlap | Top 30 points | Top 30 overlap |
| --- | ---: | ---: | ---: | ---: |
| B5_EB_POINTS_PER90 | 67.66 | 0.138 | 122.36 | 0.198 |
| B3_RECENT_POINTS_P5 | 65.99 | 0.132 | 119.41 | 0.191 |
| B4_RECENT_MINUTES_P3 | 61.21 | 0.120 | 117.66 | 0.193 |
| B4_RECENT_MINUTES_P5 | 61.43 | 0.121 | 116.17 | 0.189 |
| B3_RECENT_POINTS_P3 | 61.99 | 0.125 | 114.87 | 0.171 |
| B6_PREVIOUS_SEASON_GW1 | 61.04 | 0.148 | 98.59 | 0.179 |
| B2_POSITION_MEAN | 23.37 | 0.043 | 43.43 | 0.072 |
| B1_GLOBAL_MEAN | 14.28 | 0.018 | 29.01 | 0.037 |
| B0_ZERO | 11.89 | 0.011 | 24.43 | 0.027 |

## 12. Block-Bootstrap Comparisons

Differences are MAE minus `B1_GLOBAL_MEAN`, resampling whole season-gameweek blocks
with 300 bootstrap samples and seed `20260722`.

Rolling intervals:

| Baseline | Mean diff | 95% CI | Blocks |
| --- | ---: | --- | ---: |
| B0_ZERO | -0.3375 | [-0.3568, -0.3159] | 76 |
| B2_POSITION_MEAN | -0.0064 | [-0.0071, -0.0058] | 76 |
| B3_RECENT_POINTS_P3 | -0.0213 | [-0.0471, 0.0009] | 76 |
| B3_RECENT_POINTS_P5 | -0.0231 | [-0.0485, 0.0017] | 76 |
| B4_RECENT_MINUTES_P3 | -0.4596 | [-0.4855, -0.4331] | 76 |
| B4_RECENT_MINUTES_P5 | -0.4533 | [-0.4770, -0.4308] | 76 |
| B5_EB_POINTS_PER90 | -0.4707 | [-0.4954, -0.4476] | 76 |
| B6_PREVIOUS_SEASON_GW1 | 0.3055 | [0.2953, 0.3158] | 76 |

GW1 intervals use only 2 blocks, so they are descriptive rather than strong evidence.

## 13. Run Provenance

Each run writes:

- `manifest.json`
- `frozen_fixture_predictions.parquet`
- `scored_fixture_predictions.parquet`
- `scored_player_gameweek_predictions.parquet`
- `metrics_overall.csv`
- `metrics_by_season.csv`
- `metrics_by_gameweek.csv`
- `metrics_by_position.csv`
- `metrics_by_population.csv`
- `ranking_by_population_topk.csv`
- `population_coverage.csv`
- `calibration.csv`
- `ranking_topk.csv`
- `bootstrap_mae_differences.csv`

The manifest records run ID, UTC creation time, git commit, dirty-worktree state,
source table hashes, feature registry/config paths, seasons, folds, leakage audit
result, baseline parameters, random seed, and package version.

## 14. Limitations

The panel evaluates rows present in historical player-fixture data. This is a strong
pre-deadline candidate proxy because it includes non-playing squad rows, but it is not
yet a full prospective squad-registration reconstruction.

`B0_ZERO` is competitive on MAE because many player-gameweek rows are zero-point,
zero-minute rows. Ranking, pre-deadline candidate metrics, appearance-only diagnostics,
and RMSE provide important counterweights.

GW1 has only two validation blocks here, so interval estimates are narrow artifacts of
the small block count. Later seasons should be added before drawing stronger cold-start
claims.

No baseline here is a production forecasting model.

## 15. Exact Rebuild Commands

```text
uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25
uv run fpl backtest-baselines --seasons 2022-23,2023-24,2024-25 --test-seasons 2023-24,2024-25 --mode rolling --run-id phase3_rolling_hardened
uv run fpl backtest-baselines --seasons 2022-23,2023-24,2024-25 --test-seasons 2023-24,2024-25 --mode gw1 --run-id phase3_gw1_hardened
uv run fpl compare-baselines --run-id phase3_rolling_hardened
uv run fpl compare-baselines --run-id phase3_gw1_hardened
uv run fpl inspect-backtest --run-id phase3_rolling_hardened
uv run fpl inspect-backtest --run-id phase3_gw1_hardened
```
