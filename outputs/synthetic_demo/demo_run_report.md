# FPL Forecast MVP Run Report: `project_20260722T045210Z`

## Assumptions And Cutoffs

- Synthetic deterministic data was generated for 2023-24 through 2025-26.
- The demo target is the synthetic 2025-26 GW1-GW4 horizon.
- 2026/27 production mode remains blocked until official target-season inputs are available.

## Commands

```bash
uv run fpl run-demo --seasons 2023-24,2024-25,2025-26 --gameweeks 6 --draws 500
```

## Projection Metrics

| pool | mae | rmse | spearman |
| --- | --- | --- | --- |
| all_listed | 1.904 | 2.636 | 0.591 |
| meaningful_appearance | 1.969 | 2.692 | 0.547 |
| realistic_candidate_pool | 2.244 | 2.946 | 0.250 |

## Baseline Metrics

| model | mae | rmse | spearman |
| --- | --- | --- | --- |
| fpl_form_snapshot | 3.301 | 4.037 | 0.007 |
| position_mean | 2.269 | 2.975 | 0.186 |
| previous_season_shrunk_points_per90 | 2.014 | 2.644 | 0.598 |
| price_position_bucket_mean | 1.958 | 2.648 | 0.575 |
| recent_3_match | 1.991 | 2.970 | 0.569 |
| recent_5_match | 1.914 | 2.863 | 0.578 |
| zero_points | 2.726 | 4.014 | 0.000 |

## Top Player-Gameweek Projections

| gameweek | player_name | team | position | price | mean_xp | p_appearance |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | BIR MID2 | BIR | MID | 88 | 5.354 | 1.000 |
| 4 | BIR MID1 | BIR | MID | 101 | 5.110 | 1.000 |
| 4 | BIR MID2 | BIR | MID | 88 | 5.072 | 1.000 |
| 3 | DOR MID2 | DOR | MID | 90 | 4.934 | 1.000 |
| 4 | BIR MID3 | BIR | MID | 75 | 4.850 | 1.000 |
| 2 | BIR MID1 | BIR | MID | 101 | 4.774 | 1.000 |
| 1 | DOR MID2 | DOR | MID | 90 | 4.752 | 1.000 |
| 1 | DOR MID1 | DOR | MID | 103 | 4.582 | 1.000 |
| 3 | DOR MID1 | DOR | MID | 103 | 4.544 | 1.000 |
| 1 | ALB MID1 | ALB | MID | 100 | 4.520 | 1.000 |
| 4 | BIR FWD1 | BIR | FWD | 92 | 4.480 | 1.000 |
| 4 | FAL MID1 | FAL | MID | 105 | 4.446 | 1.000 |

## Optimized Squad

| player_name | team | position | price | horizon_xp |
| --- | --- | --- | --- | --- |
| FAL DEF3 | FAL | DEF | 63 | 14.934 |
| FAL DEF4 | FAL | DEF | 55 | 13.306 |
| DOR DEF4 | DOR | DEF | 53 | 10.066 |
| COV DEF4 | COV | DEF | 52 | 8.776 |
| ALB DEF5 | ALB | DEF | 42 | 3.974 |
| BIR FWD2 | BIR | FWD | 74 | 15.826 |
| ALB FWD2 | ALB | FWD | 73 | 13.612 |
| EXE FWD2 | EXE | FWD | 77 | 12.176 |
| FAL GKP1 | FAL | GKP | 60 | 14.180 |
| ALB GKP1 | ALB | GKP | 55 | 10.886 |
| DOR MID2 | DOR | MID | 90 | 17.652 |
| BIR MID3 | BIR | MID | 75 | 17.448 |

## GW1 Lineup

| gameweek | player_uid | player_name | team | position | price | mean_xp | role | bench_order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | code:10054 | DOR MID2 | DOR | MID | 90 | 4.752 | captain | 0 |
| 1 | code:10024 | BIR MID2 | BIR | MID | 88 | 4.266 | starter | 0 |
| 1 | code:10025 | BIR MID3 | BIR | MID | 75 | 3.952 | starter | 0 |
| 1 | code:10056 | DOR MID4 | DOR | MID | 64 | 3.840 | starter | 0 |
| 1 | code:10029 | BIR FWD2 | BIR | FWD | 74 | 3.586 | starter | 0 |
| 1 | code:10014 | ALB FWD2 | ALB | FWD | 73 | 3.552 | starter | 0 |
| 1 | code:10040 | COV MID3 | COV | MID | 76 | 3.486 | starter | 0 |
| 1 | code:10076 | FAL GKP1 | FAL | GKP | 60 | 3.202 | starter | 0 |
| 1 | code:10051 | DOR DEF4 | DOR | DEF | 53 | 3.118 | starter | 0 |
| 1 | code:10081 | FAL DEF4 | FAL | DEF | 55 | 2.906 | starter | 0 |
| 1 | code:10080 | FAL DEF3 | FAL | DEF | 63 | 3.218 | vice_captain | 0 |
| 1 | code:10074 | EXE FWD2 | EXE | FWD | 77 | 3.154 | bench | 1 |

## Known Limitations

- Synthetic demo data is not evidence of live model quality.
- Bonus, cards, saves, defensive contributions, and penalty events use residual empirical approximations.
- 2026/27 production optimization is blocked until official target-season rules and prices are available.

## Plot

![Projection error by pool](/Users/daniel/Documents/Codex/2026-07-22/files-mentioned-by-the-user-fpl/reports/figures/project_20260722T045210Z_metric_plot.png)
