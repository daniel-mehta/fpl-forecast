# Phase 7 Prices And Decision Optimization Report

Phase 7 adds a leakage-safe decision layer on top of frozen Phase 6 xPoints. It handles
integer-tenths FPL prices, official squad and lineup constraints, captain and vice-captain choice,
bench ordering, autosub scoring, an exact weekly-reset squad benchmark, and an exhaustive no-chip
multi-gameweek transfer planner for small certified cases.

The historical decision backtests are weekly-reset squad benchmarks: every gameweek starts from a
fresh legal squad. They are not realistic season-management simulations.

Phase 7 does not implement chips, live production forecasting, dashboard work, deployment, or
scheduling.

## Rules And Price Handling

The versioned decision rule set is `fpl_2025_26_api_verified`. It was checked against the cached
official `bootstrap-static` snapshot at
`data/raw/fpl_api/2025-26/bootstrap_static/20260722T164209744637Z.json`.

Rules enforced:

- 15-player squad: 2 GKP, 5 DEF, 5 MID, 3 FWD;
- 11-player lineup with exactly 1 GKP, 3-5 DEF, 2-5 MID, and 1-3 FWD;
- maximum 3 players per Premier League team;
- 100.0 budget represented as integer tenths;
- captain and vice-captain must be distinct;
- bench is one goalkeeper plus three ordered outfield substitutes;
- transfer hits are 4 points after free transfers;
- selling price uses integer-tenths FPL sell-on logic.

Prices come from the audited historical player-fixture fact table for backtests. Price is a decision
constraint and a simple market baseline, not a predictive feature in xPoints models.

## Exact Weekly Squad Optimizer

The real historical runner uses SciPy HiGHS MILP over the full gameweek candidate set. Binary
variables select:

- 15-player ownership;
- 11 starters;
- one captain.

The MILP enforces position quotas, formation limits, budget, max-three-per-team, starter ownership,
and captain ownership. The objective is starter expected points plus captain expected bonus. In
appearance-aware mode the captain bonus is multiplied by pre-deadline appearance probability; in
mean-only mode it is not. Vice-captain and bench order are then derived deterministically from the
optimal selected squad.

Small MILP cases are tested against exhaustive brute-force enumeration. Selection stability is also
tested under tiny forecast perturbations.

Solver certificate summary:

```text
rolling statuses={'optimal': 380}
rolling max_gap=2.96e-16
rolling max_runtime_seconds=0.7678
rolling mean_runtime_seconds=0.0892

gw1 statuses={'optimal': 10}
gw1 max_gap=0.0
gw1 max_runtime_seconds=0.2246
gw1 mean_runtime_seconds=0.0761
```

## Real Runs

Rolling:

```text
run_id=phase7_decisions_rolling_real
decisions=380
models=5
gameweek decisions per model=76
all_legal=True
max_cost_tenths=1000
min_bank_tenths=0
```

GW1:

```text
run_id=phase7_decisions_gw1_real
decisions=10
models=5
gameweek decisions per model=2
all_legal=True
max_cost_tenths=1000
min_bank_tenths=0
```

Frozen decision artifacts contain only decision-time columns such as model name, objective, cost,
bank, lineup, captain, vice-captain, bench, formation, solver status, solver bound, solver gap,
candidate count, runtime, and search scope. Actual points, actual minutes, and autosub outcomes are
written only to scored artifacts.

## Rolling Decision Metrics

```text
model                         n   expected realized mean_only app_adv autosubs feasible exact bank
X2_TEAM_CONSTRAINED_SIM_M3    76  69.34    63.51    69.55    -0.20   0.24     1.00     1.00  16.13
X2_TEAM_CONSTRAINED_SIM_M5    76  60.36    62.53    60.71    -0.35   0.34     1.00     1.00  16.82
X1_INDEPENDENT_COMPONENT_M3   76  63.80    61.00    63.98    -0.18   0.17     1.00     1.00  16.99
X0_PHASE3_B5_EB_POINTS_PER90  76  63.79    54.67    63.79     0.00   0.32     1.00     1.00 119.96
D0_PRICE_VALUE_BASELINE       76  48.19    41.11    48.51    -0.32   0.78     1.00     1.00   0.16
```

Default Phase 6 `X2_TEAM_CONSTRAINED_SIM_M3` had the highest realized rolling weekly-reset score in
this pass. The direct matched comparison against `X2_TEAM_CONSTRAINED_SIM_M5` was:

```text
matched_decisions=76
mean_realized_difference=0.9868
bootstrap_ci_low=-2.0658
bootstrap_ci_high=4.2243
captain_agreement=0.6842
mean_lineup_overlap=0.5502
```

The confidence interval crosses zero, so the realized-score difference is not decisive.

## GW1 Decision Metrics

Only two GW1 folds are available, so this is weak evidence.

```text
model                         n  expected realized mean_only app_adv autosubs feasible exact bank
X1_INDEPENDENT_COMPONENT_M3   2  60.30    57.50    60.42    -0.12   1.00     1.00     1.00  0.00
D0_PRICE_VALUE_BASELINE       2  36.00    47.50    46.00   -10.00   0.00     1.00     1.00  0.00
X2_TEAM_CONSTRAINED_SIM_M3    2  70.12    46.50    70.13    -0.00   1.00     1.00     1.00  2.50
X2_TEAM_CONSTRAINED_SIM_M5    2  53.60    36.00    54.32    -0.72   0.00     1.00     1.00  0.00
X0_PHASE3_B5_EB_POINTS_PER90  2   0.00    16.00     0.00     0.00   2.50     1.00     1.00 237.50
```

`D0_PRICE_VALUE_BASELINE` is the added nonzero market-price baseline. It is a meaningful squad-value
comparator for GW1, unlike the retained zero-valued B5 reference.

## Selected-Player Calibration

Selected-player calibration shows the expected-versus-realized gap. Rolling X2-M3 selected players:

```text
bin selected mean_prediction mean_actual bias
0   228      0.0168          0.1096     -0.0928
1   228      3.2011          3.0658      0.1353
2   228      4.7281          4.0658      0.6623
3   228      5.5684          5.1096      0.4587
4   228      7.5236          6.9474      0.5762
```

The optimizer amplifies calibration error because it deliberately selects high predictions. X2-M3
overpredicts in the upper selected bins, which explains part of the gap between mean expected score
69.34 and mean realized score 63.51. The same effect is visible for the price baseline, whose top
bin overpredicts by 2.86 points per selected player.

## Transfer Planning Evidence

`phase7_transfer_demo` writes a two-gameweek no-chip transfer plan solved by exhaustive search over a
small universe. It tracks ownership, purchases, sales, bank, purchase/selling prices, free-transfer
rollover and cap, hits, legal squads, lineups, captaincy, vice-captaincy, and bench order in every
gameweek.

```text
solver_status=exhaustive_optimal
objective_bound=103.6475
objective_gap=0.0
states_evaluated=992

GW1 transfers_out=('DEF_0',) transfers_in=('DEF_6',) bank_before=10 bank_after=10 FT 1->1 hit=0
GW2 transfers_out=('MID_0',) transfers_in=('MID_6',) bank_before=10 bank_after=10 FT 1->1 hit=0
```

Honest historical multi-gameweek transfer backtesting is not possible with the current archived
evidence. The project does not yet have pre-deadline manager state, purchase prices, bank, free
transfers, transfer history, or a prospective player availability universe. Reconstructing those
inputs retrospectively would require unsupported assumptions.

## Current Inference Guard

`forecast-decisions` validates current fixture season identity through the Phase 4 current-fixture
guard before writing anything. With the cached local data, a request for `2026-27` fails because the
fixture payload infers as `2025-26`.

```text
Decision forecast failed: Current fixture season mismatch: requested 2026-27, inferred 2025-26.
```

No fabricated 2026-27 decision outputs are produced.

## Test Coverage

Tests cover:

- official squad and lineup rule shape;
- integer-tenths selling price and budget arithmetic;
- legal lineup, captain, vice-captain, bench, autosub, and captain fallback behavior;
- exact pruned squad optimization against brute force on a small universe;
- full-candidate MILP optimization against brute force on a small universe;
- selection stability under tiny forecast perturbations;
- exhaustive multi-gameweek transfer planning with bank, rollover, hits, legal squads and lineups;
- frozen decision artifact rejection of target/future columns;
- current decision guard failure before output.

## Limitations

- The weekly-reset benchmark is not a realistic season-management strategy.
- The MILP objective optimizes expected starter points plus captain bonus; vice-captain fallback and
  bench autosub value are evaluated after selection rather than embedded in the MILP objective.
- Historical candidate universes are derived from observed Phase 6 player-gameweek predictions, not
  archived official pre-deadline availability snapshots.
- Historical multi-gameweek transfer backtesting remains blocked by missing manager-state evidence.
- Chips are not implemented.
- Current-season decisions remain blocked until genuine target-season current data and prerequisite
  current xPoints artifacts exist.

## Decision

Default decision model for Phase 7 remains `X2_TEAM_CONSTRAINED_SIM_M3`. It is the Phase 6 default,
produces certified optimal legal weekly-reset decisions in all rolling and GW1 folds, and had the
highest realized rolling weekly-reset score in this pass, though the X2-M3 versus X2-M5 paired
bootstrap interval crosses zero.
