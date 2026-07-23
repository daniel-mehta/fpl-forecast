# Phase 5 Expected-Minutes Report

Phase 5 implements leakage-safe player expected-minutes and appearance-probability backtests over
the Phase 2 player-fixture panel. It does not implement player expected-points projections,
simulation, optimization, or production forecasting.

## Scope

Implemented command family:

- `uv run fpl backtest-minutes`
- `uv run fpl compare-minutes`
- `uv run fpl inspect-minutes`
- `uv run fpl forecast-minutes`

Generated outputs are ignored under `reports/minutes_backtests/<run_id>/`.

## Leakage Controls

Historical labels come from `fact_player_fixture.parquet` and are included only after scoring.
Frozen predictions exclude actual minutes, appearance, starts, role-duration state, target points,
price, transfer popularity, lineup status, and other post-deadline fields.

Per-row Phase 5 lag features are rebuilt with:

```text
source_available_time < information_cutoff
```

This is stricter than sorting by kickoff or by row order. For every target player-fixture row, prior
minutes, appearances, starts, and recent means are computed only from that player's rows whose
match-derived source availability is strictly before the target cutoff. When no eligible history
exists, the feature source time is the documented sentinel `1900-01-01T00:00:00Z`.

Exact starts are present for all audited player rows:

```text
starts_exact_coverage=1.0000
```

No start labels were fabricated.

## Outcome Labels

Role-duration states:

- `DNP`: minutes = 0
- `SUB_UNDER_60`: minutes > 0, exact start = 0, minutes < 60
- `SUB_60_PLUS`: exact start = 0, minutes >= 60
- `START_UNDER_60`: exact start = 1, minutes < 60
- `START_60_TO_89`: exact start = 1, 60 <= minutes < 90
- `START_90`: exact start = 1, minutes >= 90

Derived binary targets:

- `appearance = minutes > 0`
- `start = exact starts == 1`
- `reached_60 = minutes >= 60`
- `played_90 = minutes >= 90`

## Evaluation Populations

The candidate-population rule is version-controlled in
`src/fpl_forecast/minutes_model/config.json`.

- Primary data-quality population: all observed football-player rows.
- Decision-relevant population: `pre_deadline_history_active`.
- Cold starts: `cold_start_no_history`.
- Actual appearances: scored only as a retrospective diagnostic.

`pre_deadline_history_active` means:

- In-season folds: positive minutes in the previous five available fixtures, or positive
  season-to-date appearances before the cutoff.
- GW1 folds: positive prior-season minutes or appearances.

Rolling coverage:

```text
season   population                    rows   appearances
2023-24  cold_start_no_history          7734   221
2023-24  other_history_inactive         3055   73
2023-24  pre_deadline_history_active   18936   11090
2024-25  cold_start_no_history          5426   191
2024-25  other_history_inactive         3088   81
2024-25  pre_deadline_history_active   18769   11294
```

GW1 coverage:

```text
season   population                    rows  appearances
2023-24  cold_start_no_history          255   76
2023-24  pre_deadline_history_active    403   226
2024-25  cold_start_no_history          180   57
2024-25  other_history_inactive          26   16
2024-25  pre_deadline_history_active    410   234
```

## Models

Baselines:

- `M0_ZERO_MINUTES`: predicts 0 minutes and state `DNP`.
- `M1_RECENT_MEAN_P3`: mean minutes from the previous three available fixtures.
- `M2_RECENT_MEAN_P5`: mean minutes from the previous five available fixtures.
- `M3_EWMA_MINUTES`: `alpha * previous_fixture_minutes + (1 - alpha) * (0.65 * previous_3_mean + 0.35 * previous_10_mean)`, clipped to 0-90.
- `M4_PREVIOUS_SEASON_ROLE_GW1`: shrunk prior-season minutes-per-appearance multiplied by
  prior-season appearance probability:

```text
role_rate = prior_season_minutes / prior_season_appearances
appearance_probability = prior_season_appearances / (prior_season_appearances + shrink_matches)
prediction = role_rate * appearance_probability
```

Learned challengers:

- `M5_REGULARIZED_STATE_SOFTMAX`: L2-regularized multinomial softmax over the six states.
- `M6_NONLINEAR_RECENCY_ENSEMBLE`: shrunk nonlinear bucket ensemble using position, recent minutes,
  recent starts, and prior-season history buckets.

The learned models use a deterministic capped recent training window
(`max_model_training_rows=5000`) so rolling verification remains practical. Baselines still use the
full leakage-eligible fold history.

## Real Runs

Rolling:

```text
run_id=phase5_minutes_rolling_real
folds=76
frozen_rows=399056
scored_rows=399056
cutoff_min=2023-08-11T19:00:00+00:00
cutoff_max=2025-05-25T15:00:00+00:00
```

GW1:

```text
run_id=phase5_minutes_gw1_real
folds=2
frozen_rows=8918
scored_rows=8918
cutoff_min=2023-08-11T19:00:00+00:00
cutoff_max=2024-08-16T19:00:00+00:00
```

## Rolling Metrics

All observed player rows:

```text
model                            rows   MAE      RMSE     bias     Spearman
M3_EWMA_MINUTES                  57008  11.8929  23.3096   0.0720  0.7751
M1_RECENT_MEAN_P3                57008  12.4368  24.6904   0.0688  0.7778
M2_RECENT_MEAN_P5                57008  13.5420  25.2703   0.0720  0.7600
M5_REGULARIZED_STATE_SOFTMAX     57008  15.8654  23.7152   0.5445  0.7483
M6_NONLINEAR_RECENCY_ENSEMBLE    57008  24.2364  28.5393   0.5170  0.7103
M4_PREVIOUS_SEASON_ROLE_GW1      57008  25.1531  37.5661  -2.2285  0.3570
M0_ZERO_MINUTES                  57008  26.2611  46.0018 -26.2611  NaN
```

Decision-relevant `pre_deadline_history_active` rows:

```text
model                            rows   MAE      RMSE     bias     Spearman
M3_EWMA_MINUTES                  37705  17.3712  27.9478   0.6002  0.7260
M1_RECENT_MEAN_P3                37705  18.2248  29.6773   0.6367  0.7038
M2_RECENT_MEAN_P5                37705  19.8986  30.4144   0.6409  0.6765
M5_REGULARIZED_STATE_SOFTMAX     37705  21.4296  28.4477  -0.8184  0.7042
M6_NONLINEAR_RECENCY_ENSEMBLE    37705  29.6344  33.4092  -5.4655  0.6349
M4_PREVIOUS_SEASON_ROLE_GW1      37705  33.6132  43.7935  -6.8034  0.2384
M0_ZERO_MINUTES                  37705  39.1322  56.2045 -39.1322  NaN
```

Rolling binary metrics selected examples:

```text
target      best / notable model              Brier   log_loss  predicted  actual
appearance  M5_REGULARIZED_STATE_SOFTMAX      0.1109  0.3618    0.3906     0.4026
start       M3_EWMA_MINUTES                   0.0840  0.3348    0.2929     0.2933
reached_60  M3_EWMA_MINUTES                   0.0892  0.4407    0.2926     0.2749
played_90   M5_REGULARIZED_STATE_SOFTMAX      0.0893  0.2823    0.2018     0.1899
```

Rolling top-k selection averages:

```text
population                   k   best model                         avg actual minutes
all_observed_players         15  M5_REGULARIZED_STATE_SOFTMAX       1264.86
all_observed_players         30  M5_REGULARIZED_STATE_SOFTMAX       2511.36
pre_deadline_history_active  15  M3_EWMA_MINUTES                    1267.71
pre_deadline_history_active  30  M3_EWMA_MINUTES                    2513.92
```

## GW1 Metrics

All observed GW1 rows:

```text
model                            rows  MAE      RMSE     bias      Spearman
M2_RECENT_MEAN_P5                1274  22.8913  37.7077  -2.6500   0.4616
M1_RECENT_MEAN_P3                1274  23.0984  38.4017  -2.7924   0.4601
M3_EWMA_MINUTES                  1274  23.3735  38.2217  -2.6500   0.4567
M5_REGULARIZED_STATE_SOFTMAX     1274  25.4630  34.1476  -0.6946   0.5062
M4_PREVIOUS_SEASON_ROLE_GW1      1274  25.4631  36.3439  -3.5685   0.4249
M6_NONLINEAR_RECENCY_ENSEMBLE    1274  28.8375  33.7046  -0.0700   0.5008
M0_ZERO_MINUTES                  1274  30.9074  49.9419 -30.9074   NaN
```

GW1 `pre_deadline_history_active` rows:

```text
model                            rows  MAE      RMSE     bias      Spearman
M2_RECENT_MEAN_P5                 813  25.4715  37.7910   4.1995   0.5000
M1_RECENT_MEAN_P3                 813  25.6624  38.5781   4.0083   0.4931
M3_EWMA_MINUTES                   813  26.2031  38.5809   4.1917   0.4975
M5_REGULARIZED_STATE_SOFTMAX      813  27.5371  35.4104   0.7324   0.5057
M4_PREVIOUS_SEASON_ROLE_GW1       813  29.6360  35.5609   4.6738   0.5091
M6_NONLINEAR_RECENCY_ENSEMBLE     813  30.8234  34.8350  -1.2169   0.5189
M0_ZERO_MINUTES                   813  38.1673  55.7063 -38.1673   NaN
```

GW1 top-k selection averages:

```text
population                   k   best model                         avg actual minutes
all_observed_players         15  M3_EWMA_MINUTES                    1076.0
all_observed_players         30  M5_REGULARIZED_STATE_SOFTMAX       2133.0
pre_deadline_history_active  15  M3_EWMA_MINUTES                    1121.0
pre_deadline_history_active  30  M5_REGULARIZED_STATE_SOFTMAX       2169.0
```

## Calibration

Rolling calibration examples:

```text
model                         bin       rows   mean_pred  mean_actual  actual_minus_pred
M3_EWMA_MINUTES               0-1       24800  0.04       1.39         1.35
M3_EWMA_MINUTES               75-90     10163  85.25      77.61       -7.65
M5_REGULARIZED_STATE_SOFTMAX  1-15      31651  5.40       3.16        -2.24
M5_REGULARIZED_STATE_SOFTMAX  60-75      8390  68.91      70.14        1.23
M6_NONLINEAR_RECENCY_ENSEMBLE 1-15      19730  12.24      1.13       -11.11
M6_NONLINEAR_RECENCY_ENSEMBLE 45-60     11384  51.30      71.44       20.14
```

Interpretation: `M3` underpredicts very low bins but overpredicts high-minute bins. `M5` is the
best-behaved of the learned state models, with smaller midrange gaps but still low-bin
overprediction. `M6` is directionally useful for nonlinear ranking in places but poorly calibrated
at the extremes.

## Lineup Diagnostics

Rolling team-fixture start sums:

```text
model                            groups  raw_mean  adjusted_mean  adjusted_groups
M1_RECENT_MEAN_P3                  1520  10.9385   10.9855        1518
M2_RECENT_MEAN_P5                  1520  10.9680   10.9855        1518
M3_EWMA_MINUTES                    1520  10.9466   10.9855        1518
M4_PREVIOUS_SEASON_ROLE_GW1        1520  11.4239   10.9711        1516
M5_REGULARIZED_STATE_SOFTMAX       1520  12.4479   11.0000        1520
M6_NONLINEAR_RECENCY_ENSEMBLE      1520  11.0659   11.0000        1520
```

The adjustment is target-free and operates within `model_name, season, fixture, team`. It uses a
logit shift on `p_start`, preserves within-team ranking, and rescales starter/nonstarter state
probabilities so expected minutes and role-state probabilities remain coherent.

## Current Inference

`forecast-minutes` validates official current fixtures and players before any player forecast is
attempted. With the cached local data, `2026-27` normalized current files infer as `2025-26`, so the
command fails rather than fabricating a 2026-27 player-minutes forecast from stale snapshots.

## Limitations

- The historical candidate set is reconstructed from observed player-fixture rows. It is
  result-leakage-safe but can still differ from a prospective pre-deadline FPL player list.
- Baseline state probabilities for pure minutes baselines are intentionally simple and less
  calibrated than their minutes rankings.
- `M5` and `M6` are transparent, deterministic challengers, not tuned production models.
- The learned models use a capped recent training window for verification speed.
- Current player-minute inference remains guarded until genuine official future-season launch data
  and stable current player identities are available.

## Verdict

Phase 5 is implemented and verified as a leakage-safe expected-minutes backtesting layer. It is not a
production expected-points or squad-decision system.
