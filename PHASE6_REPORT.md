# Phase 6 Component xPoints Report

Phase 6 adds a leakage-safe, component-based FPL expected-points backtesting layer. It reconstructs
historical FPL points from components, estimates component rates from fold-local history, integrates
Phase 4 T2 team expectations and Phase 5 minutes outputs, and writes discrete simulated point
distributions at player-fixture and player-gameweek grain.

Phase 6 does not implement prices, squad optimization, transfer planning, chips, deployment,
scheduling, dashboards, or production current-season forecasts.

## Source Coverage

The audited Phase 2 fact table contains 83,513 football-player rows and 322 assistant-manager rows.
Assistant managers remain preserved upstream and are excluded from the standard Phase 6 player model.

All required player component columns are present with zero nulls across `2022-23`, `2023-24`, and
`2024-25`: minutes, starts, goals, assists, clean sheets, goals conceded, saves, penalty saves,
penalty misses, yellow cards, red cards, own goals, bonus, BPS, and total points.

Observed component maxima include: 4 goals, 4 assists, 13 saves, 2 penalty saves, 2 penalty misses,
1 yellow, 1 red, 2 own goals, 3 bonus, and BPS from -25 to 102.

## Scoring Rules

The versioned rule set is `fpl_standard_2022_2025`:

- appearance: 1 point for playing, plus 1 point for 60+ minutes;
- goals: GKP/DEF 6, MID 5, FWD 4;
- assists: 3;
- clean sheets: GKP/DEF 4, MID 1, requiring 60+ minutes;
- saves: 1 per 3 saves;
- penalty saves: 5;
- penalty misses: -2;
- goals conceded: GKP/DEF -1 per 2 conceded;
- yellow cards: -1;
- red cards: -3;
- own goals: -2;
- bonus: source bonus points.

Reconstruction audit:

```text
rows=83513
exact_matches=83513
exact_match_pct=1.0000
mean_abs_difference=0.0
max_abs_difference=0
difference_counts: {0: 83513}
```

Every season-position group reconstructed exactly.

## Component Targets

Targets are actual historical columns from `fact_player_fixture.parquet`; `total_points` is used
only after frozen prediction scoring. Vaastav `xP` and price are never used as Phase 6 features.

Component definitions:

- minutes/role: Phase 5 M3 or M5 frozen probabilities;
- goals: player goals per 90, shrunk to position/league priors;
- assists: player assists per 90, shrunk to priors;
- clean-sheet points: Phase 4 T2 clean-sheet probability times Phase 5 60+ probability;
- goals-conceded deduction: T2 opponent expected goals, approximated per conceded-pair exposure;
- saves and penalty saves: goalkeeper-only shrunk per-90 processes;
- penalty misses, cards, own goals: strongly bounded rare-event per-90 processes;
- bonus: bounded shrunk bonus-per-90 process with explicit tie-rule tests for the BPS allocator.

## Models

- `X0_PHASE3_B5_EB_POINTS_PER90`: frozen Phase 3 B5 reference.
- `X1_INDEPENDENT_COMPONENT_RATES_M3`: shrunk independent component rates with Phase 5 M3 minutes.
- `X2_TEAM_CONSTRAINED_SIM_M3`: T2 team-goal-constrained component simulation with Phase 5 M3 minutes.
- `X2_TEAM_CONSTRAINED_SIM_M5`: same X2 component model with Phase 5 M5 state probabilities.

X2 normalizes scorer and assister shares within each target team-fixture. Expected allocated player
goals reconcile exactly to T2 team expected goals. Point percentiles and threshold probabilities are
computed from discrete Monte Carlo draws, not a normal approximation.

## Real Runs

Rolling:

```text
run_id=phase6_xpoints_rolling_real
folds=76
frozen_rows=228032
scored_rows=228032
player_gameweek_rows=222664
max_goal_conservation_abs_error=0.00000000
```

GW1:

```text
run_id=phase6_xpoints_gw1_real
folds=2
frozen_rows=5096
scored_rows=5096
player_gameweek_rows=5096
max_goal_conservation_abs_error=0.00000000
```

Frozen row counts equal target player-fixture rows multiplied by four model variants:

- rolling: 57,008 x 4 = 228,032;
- GW1: 1,274 x 4 = 5,096.

No duplicate `season, stable_fixture_uid, player_uid, model_name` keys were found.

## Rolling Metrics

```text
model                         rows   MAE     RMSE    bias     Spearman
X1_INDEPENDENT_COMPONENT_M3   57008  0.8992  1.9584 -0.2440   0.7049
X2_TEAM_CONSTRAINED_SIM_M3    57008  0.9060  1.9543 -0.2149   0.7044
X2_TEAM_CONSTRAINED_SIM_M5    57008  0.9463  1.9406 -0.1601   0.6752
X0_PHASE3_B5_EB_POINTS_PER90  57008  0.9824  2.0295 -0.1017   0.6501
```

Decision-relevant `pre_deadline_history_active`:

```text
X1_M3  MAE=1.1315 RMSE=2.1996 bias=-0.2974 Spearman=0.6704
X2_M3  MAE=1.1401 RMSE=2.1949 bias=-0.2602 Spearman=0.6702
X2_M5  MAE=1.1643 RMSE=2.1795 bias=-0.2209 Spearman=0.6559
X0_B5  MAE=1.2374 RMSE=2.2803 bias=-0.1161 Spearman=0.5966
```

Rolling top-k actual points:

```text
population                   k   best model       avg top-k actual points
all_observed_players         15  X2_M5            76.14
all_observed_players         30  X2_M5            134.32
pre_deadline_history_active  15  X2_M5            76.01
pre_deadline_history_active  30  X2_M5            134.32
```

Block bootstrap, default X2-M3 versus Phase 3 B5:

```text
mean_mae_difference=-0.0752
ci95=[-0.0878, -0.0660]
evaluated_gameweeks=76
```

## GW1 Metrics

Only two GW1 folds are available, so this is weak evidence.

```text
model                         rows  MAE     RMSE    bias     Spearman
X2_TEAM_CONSTRAINED_SIM_M5    1274  1.1716  2.0763 -0.2675   0.4749
X1_INDEPENDENT_COMPONENT_M3   1274  1.1896  2.1623 -0.3298   0.4532
X2_TEAM_CONSTRAINED_SIM_M3    1274  1.2013  2.1887 -0.2955   0.4501
X0_PHASE3_B5_EB_POINTS_PER90  1274  1.2818  2.5895 -1.2692   NaN
```

The Phase 3 B5 frozen GW1 reference predicts zero for all rows in the available hardened artifact,
so its GW1 ranking correlation is undefined.

## Distribution Metrics

Rolling:

```text
model   P(>=5) Brier  central80 coverage  zero_pred  zero_actual
X0_B5   0.0871        0.5787              0.5730     0.6223
X1_M3   0.0655        0.8823              0.7037     0.6223
X2_M3   0.0654        0.8843              0.7037     0.6223
X2_M5   0.0651        0.9142              0.6097     0.6223
```

X2-M5 has the best threshold calibration for 5+ points and zero-rate calibration, but it trails M3
on rolling MAE. X2-M3 is the selected default because it preserves team-goal conservation and stays
near X1's error frontier.

## Component Calibration

Rolling X2-M3 selected component means:

```text
component      predicted  actual   MAE
goals          0.0438     0.0399   0.0697
assists        0.0392     0.0359   0.0660
saves          0.0635     0.0848   0.0587
bonus          0.0874     0.0848   0.1483
yellow_cards   0.0527     0.0551   0.0928
red_cards      0.0015     0.0019   0.0034
own_goals      0.0015     0.0015   0.0030
```

Major calibration limitations:

- low predicted-point bins still underpredict actual points;
- high predicted-point bins have small sample sizes and can overpredict;
- saves are underpredicted for goalkeepers;
- bonus is bounded and explicit, but still simplified relative to full BPS ranking.

## Conservation

Rolling conservation:

```text
model   team-fixture groups  max_abs_goal_error  allocated_goals
X2_M3   1520                 4.44e-16            2496.39
X2_M5   1520                 4.44e-16            2496.39
```

Assists are allocated as a shrunk assisted-goal process and include an implicit no-assist outcome.
Rolling allocated expected assists total 2,237.45 versus allocated goals 2,496.39.

## Examples And Edge Cases

The tests cover:

- scoring examples across goals, assists, clean sheets, saves, penalties, cards, own goals and bonus;
- 59 versus 60 minutes;
- bonus tie cases;
- coherent goal allocation with no self-assist;
- DNP/ineligible bonus exclusion;
- deterministic simulation seed behavior;
- gameweek aggregation by summing fixture draws rather than summing percentiles;
- case-insensitive forbidden `xP` rejection;
- invalid standard position rejection.

The real panel includes transfers, position changes, cold starts and double-gameweek fixture rows
from earlier Phase 2/3/5 evidence. Historical candidate rows are observed player-fixture rows, not
a fully archived pre-deadline player universe.

## Current Inference Guard

`forecast-xpoints` validates current fixture season identity through the Phase 4 current-fixture
guard before writing anything. With the cached local data, a request for `2026-27` fails because the
payload infers as `2025-26`. No fabricated 2026-27 xPoints outputs are produced.

## Limitations

- X2 conserves expected team goals, but the stored artifacts do not preserve raw fixture-level draw
  matrices by default.
- Event timing is approximated through role/minute state and team outcome context; exact historical
  substitutions and goal times are not reconstructed from aggregate rows.
- Bonus uses bounded component rates and tested tie logic, not a full simulated BPS engine.
- Rare events use conservative priors and clipping.
- Current-season inference remains blocked until genuine target-season launch data and prerequisite
  team/minutes forecasts exist.

## Default Decision

Default for Phase 6: `X2_TEAM_CONSTRAINED_SIM_M3`.

Reason: it is close to the best rolling MAE, improves materially over Phase 3 B5, preserves team-goal
conservation, and uses the strongest rolling minutes baseline. `X1_INDEPENDENT_COMPONENT_RATES_M3`
is retained as the simplest error baseline. `X2_TEAM_CONSTRAINED_SIM_M5` is retained as a
probabilistic/GW1 challenger because it has better rolling distribution calibration and better GW1
MAE in the two available folds.

Phase 7 should consume player-fixture and player-gameweek expected points, distribution percentiles,
threshold probabilities, component breakdowns, cold-start flags, and model lineage. It should not
treat Phase 6 current inference as available until the launch-data guard passes.
