# Phase 7 Prices And Decision Optimization Report

Phase 7 adds a leakage-safe historical decision layer on top of frozen Phase 6 xPoints. It handles
integer-tenths FPL prices, official squad and lineup constraints, captain and vice-captain choice,
bench ordering, autosub scoring, and a bounded single-transfer planner.

Phase 7 does not implement chips, live production forecasting, dashboard work, deployment,
scheduling, or a full multi-gameweek transfer optimizer.

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

Prices come from the audited historical player-fixture fact table for backtests. The implementation
does not use price as a predictive feature; price is a decision constraint.

## Optimizers

The exact small-case optimizer enumerates a deterministic pruned candidate universe and is covered by
hand-built tests against brute force.

The real historical decision runner uses a deterministic greedy-repair squad constructor:

1. seed a legal cheap squad by position quota;
2. improve same-position replacements from a capped high-projection/value/cheap upgrade pool while
   preserving budget and team constraints;
3. optimize the legal lineup, captain, vice-captain, and bench exactly for the selected squad;
4. score actual historical outcomes only after frozen decisions are written.

This is intentionally documented as a feasible deterministic decision heuristic, not a claim of
global mixed-integer optimality.

## Real Runs

Rolling:

```text
run_id=phase7_decisions_rolling_real
decisions=304
models=4
gameweek decisions per model=76
all_legal=True
max_cost_tenths=1000
min_bank_tenths=0
```

GW1:

```text
run_id=phase7_decisions_gw1_real
decisions=8
models=4
gameweek decisions per model=2
all_legal=True
max_cost_tenths=1000
min_bank_tenths=0
```

Frozen decision artifacts contain only decision-time columns such as model name, objective, cost,
bank, lineup, captain, vice-captain, bench, formation, solver status, candidate count, and search
scope. Actual points, actual minutes, and autosub outcomes are written only to scored artifacts.

## Rolling Decision Metrics

```text
model                         decisions expected realized captain autosubs feasible exact_opt bank
X2_TEAM_CONSTRAINED_SIM_M3    76        65.46    61.34    7.61    0.49     1.00     0.00      8.20
X2_TEAM_CONSTRAINED_SIM_M5    76        55.60    60.41    6.87    0.57     1.00     0.00      6.74
X1_INDEPENDENT_COMPONENT_M3   76        59.81    59.26    7.58    0.51     1.00     0.00      7.86
X0_PHASE3_B5_EB_POINTS_PER90  76        64.24    56.18    7.51    0.61     1.00     0.00    106.05
```

Default Phase 6 `X2_TEAM_CONSTRAINED_SIM_M3` had the highest realized rolling decision score in this
Phase 7 pass. The direct matched comparison against `X2_TEAM_CONSTRAINED_SIM_M5` was:

```text
matched_decisions=76
mean_realized_difference=0.9342
captain_agreement=0.7237
```

## GW1 Decision Metrics

Only two GW1 folds are available, so this is weak evidence.

```text
model                         decisions expected realized captain autosubs feasible exact_opt bank
X2_TEAM_CONSTRAINED_SIM_M3    2         69.63    45.50   10.50    1.00     1.00     0.00      0.00
X2_TEAM_CONSTRAINED_SIM_M5    2         49.36    43.00    5.00    1.00     1.00     0.00      0.00
X1_INDEPENDENT_COMPONENT_M3   2         55.11    36.00    5.00    1.50     1.00     0.00      0.00
X0_PHASE3_B5_EB_POINTS_PER90  2          0.00     2.50    0.00    1.00     1.00     0.00    360.00
```

`X0_PHASE3_B5_EB_POINTS_PER90` is retained as a reference but remains unsuitable for GW1 decision
use in these artifacts because the Phase 6 GW1 frozen B5 reference predicts zero.

## Transfer Planning Evidence

`phase7_transfer_demo` writes a deterministic toy transfer plan that verifies position-preserving
replacement, bank arithmetic, free-transfer hit handling, and acceptance only when expected gain is
positive.

```text
transfers_out=('DEF_0',)
transfers_in=('DEF_6',)
bank_before=10
bank_after=10
points_hit=0
expected_gain=6.8450
accepted=True
solver_status=exact_single_transfer_scan
```

Historical transfer backtesting is not claimed complete because the project does not yet archive
pre-deadline manager state, purchase prices, bank, free transfers, or transfer history. Without
those inputs, reconstructing honest historical transfer availability would require unsupported
assumptions.

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
- deterministic single-transfer planning with bank and hit accounting;
- frozen decision artifact rejection of target/future columns;
- current decision guard failure before output.

## Limitations

- The historical squad optimizer is a deterministic feasible heuristic for real runs, not a global
  optimizer over the full player universe.
- Historical candidate universes are derived from observed Phase 6 player-gameweek predictions, not
  archived official pre-deadline availability snapshots.
- Purchase prices, manager-specific selling prices, bank, free transfers, and transfer history are
  not available historically; only rule mechanics and a small transfer planner are implemented.
- Chips are not implemented.
- Current-season decisions remain blocked until genuine target-season current data and prerequisite
  current xPoints artifacts exist.

## Decision

Default decision model for Phase 7 remains `X2_TEAM_CONSTRAINED_SIM_M3`. It is the Phase 6 default,
produces legal decisions in all rolling and GW1 folds, and had the highest realized rolling decision
score in this pass.
