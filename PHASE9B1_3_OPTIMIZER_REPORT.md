# Phase 9B1.3 Optimizer Report

Phase 9B1.3 hardens the weekly GW1 optimizer only. It does not add transfer planning, chips,
multi-gameweek optimization, public automation, deployment, player-specific rules, or manual
exclusions.

## Repository State

- Starting point: local Phase 9B work on `main`, currently ahead of `origin/main` by two commits.
- No commit, push, deploy, scheduled workflow, or frontend redesign was performed.
- Generated raw data, normalized data, decision reports, operational outputs, synced frontend data,
  build output, and dependencies remain ignored by Git.

## Previous Objective

The previous full-candidate weekly optimizer was `D1_MEAN_ONLY_MILP`.

For a selected squad and nominal starting XI it maximized:

```text
sum(expected_points for nominal starters)
+ expected_points(captain) * p_appearance(captain)
```

Bench order was assigned after solving. Bench players, ordinary automatic substitutions,
vice-captain contingency, goalkeeper substitution, unreplaced starters, and unused budget were not
genuine parts of the optimized football objective.

## Hardened Objective

The new challenger is `D2_EXPECTED_REALIZED_POINTS`.

It uses:

1. The exact full-candidate D1 MILP solution as a deterministic seed.
2. A deterministic full-pool one-swap local search over legal squads.
3. A lineup, bench-order, captain, and vice-captain evaluator that maximizes expected realized FPL
   points after ordinary automatic substitutions and captain fallback.

D2's optimized value is:

```text
expected nominal active starter points
+ expected automatic-substitution points
+ expected captain bonus when captain appears
+ expected vice-captain fallback bonus when captain does not appear and vice-captain appears
```

The expected-realized evaluator uses M7 appearance probabilities and stabilized conditional
expected points generated directly from the Phase 6 simulation draws. It enumerates all
`2^15 = 32,768` binary appearance states. Scenario probability mass is checked against one within
`1e-12`; the official run produced `1.0`.
Automatic substitutions, captaincy, and vice-captain fallback are applied in every state.

The previous 512-draw deterministic Monte Carlo path was removed from D2. The result is exact only
conditional on the supplied forecasts and independent player appearances. Independence is not a
claim about the real world and remains a limitation.

The previous evaluator reconstructed conditional points by dividing unconditional xPoints by
`p_appearance`. That was unstable when appearance probability was very small and the unconditional
simulation mean had little appearance-draw support; downstream rounding could amplify it further.

The simulation now records:

```text
raw_conditional_xpoints =
    sum(points in appearance draws) / appearance_draw_count

shrunk_conditional_xpoints =
    (
        appearance_draw_count * raw_conditional_xpoints
        + 5 * leave_one_out_position_prior
    )
    / (appearance_draw_count + 5)
```

The prior uses only simulated forecast draws from other players in the same position, falling back
to a leave-one-out global prior and then a configured `2.0` point prior. It uses no target outcomes.
Five pseudo-appearances strongly regularize one- or two-draw estimates while preserving
well-supported estimates. The raw simulation quantities satisfy:

```text
expected_points_unconditional
= simulation_appearance_probability * raw_conditional_xpoints
```

within floating-point tolerance. D2 uses M7 appearance probability multiplied by the stabilized
conditional estimate. It never divides public or rounded xPoints and does not apply appearance
probability twice.

### Pecsi / Raya Diagnosis

The original small-denominator reconstruction was not the mathematical reason Pecsi started:

| Player | Unconditional xP | M7 appearance | Old reconstructed conditional |
| --- | ---: | ---: | ---: |
| Pecsi | 0.0500 | 0.014499 | 3.4486 |
| Raya | 3.3875 | 0.884068 | 3.8317 |

Pecsi's reconstructed conditional value was already lower than Raya's. Under the same exact
evaluator, simply starting Raya and benching Pecsi improved the old squad by approximately
`0.0049`. The order came from the bounded approximate lineup shortlist, not from Pecsi having a
higher conditional expectation.

The direct stabilized estimates in the closure run are:

| Player | Appearance draws | Raw conditional | Prior | Reliability | Stabilized conditional |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pecsi | 2 / 80 | 2.0000 | 2.6526 | 0.2857 | 2.4661 |
| Raya | 69 / 80 | 3.9275 | 2.5938 | 0.9324 | 3.8374 |

The statistical instability is fixed, but the bounded lineup-search limitation remains: starting
Raya scores `50.0837`, while the returned Pecsi-start lineup scores `50.0662`, a `+0.0176`
improvement. The search algorithm was intentionally not changed in this closure pass.

D2 is not globally proven optimal for the expected-realized objective. It is globally optimal only
for the D1 seed objective, then heuristic-feasible after bounded expected-realized local search.

## Search Coverage

The earlier `25` figure meant one D1 seed plus a shortlist of 24 neighbours. It was not complete
one-swap coverage and has been removed.

At each iteration, D2 now considers every selected-player/unselected-player pair from the complete
eligible pool. It rejects position mismatches, over-budget squads, club-limit violations, and other
illegal proposals explicitly. Every remaining unique legal one-swap squad is scored by the exact
expected-realized evaluator using a deterministic inherited lineup. An accepted squad receives a
separate lineup refinement before the next iteration.

The official run reported:

| Search diagnostic | Count |
| --- | ---: |
| Eligible players | 554 |
| Players in D1 seed | 15 |
| Unselected replacement players | 539 |
| Raw proposals over 3 iterations | 24,255 |
| Rejected: position mismatch | 17,076 |
| Rejected: budget | 891 |
| Rejected: club limit | 227 |
| Rejected: other illegal | 0 |
| Feasible proposals, cumulative | 6,061 |
| Unique squads exactly evaluated | 5,907 |
| Exact squad-evaluation calls | 6,062 |
| Repeated feasible squads re-evaluated | 155 |
| Accepted improving moves | 3 |
| Iterations | 3 |

The counts reconcile: `17,076 + 891 + 227 + 0 + 6,061 = 24,255`. Repeated squads across iterations
are re-evaluated because their inherited lineup context can change; they explain why 6,062
evaluation calls cover 5,907 unique squads. Termination was `configured_iteration_bound_reached`,
not a local- or global-optimality proof.

## Substitution Rules

The evaluator implements ordinary no-chip autosub rules:

- A non-playing starting goalkeeper can be replaced only by the bench goalkeeper.
- Outfield substitutes are considered in bench order.
- An outfield substitute enters only if the resulting formation remains legal.
- Legal outfield formations require at least 3 DEF, 2 MID, and 1 FWD.
- A starter may remain unreplaced if no appearing bench player can enter legally.
- Bench players who do not appear cannot substitute.
- Points from unused substitutes are not counted.

## Captaincy Rules

- If the captain appears, the captain receives the captain bonus.
- If the captain does not appear and the vice-captain appears, the vice-captain receives the
  captain bonus.
- If neither appears, no captain bonus is awarded.
- Captain and vice-captain are distinct and selected from the nominal starting XI, preserving the
  existing project restriction.

## Deterministic Tie-Breaking

The expected-realized lineup evaluator orders candidates by:

1. Expected realized total.
2. Expected nominal starting-XI points.
3. Expected autosub contribution.
4. Expected vice-captain contingency.
5. Lower unreplaced-starter probability.
6. Higher bench expected-points sum.
7. Lower cost when all football values are equivalent.
8. Stable player-ID, bench-order, captain, and vice-captain ordering.

The squad-level D2 search applies an analogous deterministic ordering and does not reward spending
budget for its own sake.

## Tests Added

Focused Phase 9B1.3 coverage was added for:

- Hand-calculated expected-realized captain and vice-captain fallback.
- Exact probability mass across all 32,768 states.
- Unconditional versus conditional xPoints without double appearance discounting.
- Direct appearance-draw conditional xPoints and unconditional coherence.
- Zero appearance draws and one-draw rare estimates.
- General-prior shrinkage and preservation of well-supported estimates.
- Independence from rounded public xPoints.
- Stabilized two-goalkeeper ordering by hand.
- Captain and vice-captain both absent.
- Goalkeeper-only substitution.
- Formation-preserving outfield substitutions.
- An unreplaced starter when no legal bench player appears.
- Bench-order materiality.
- Expected-realized lineup selection.
- Complete legal one-swap coverage and reconciled search counters.
- D2 no-worse-than-D1 common-evaluator acceptance.
- Stronger-bench acceptance case.
- Unused-bank acceptance case.
- Cheap low-projection player acceptance without a named blacklist.
- Repeated-run determinism.

The existing Phase 7 constraints and brute-force MILP tests remain intact.

## Historical GW1 Comparison

Run:

```bash
uv run fpl backtest-decisions \
  --seasons 2023-24,2024-25,2025-26 \
  --mode gw1 \
  --run-id phase9b13_conditional_stabilized_decisions_gw1
```

Scope: weekly-reset GW1 decisions only, using frozen `X2_TEAM_CONSTRAINED_SIM_M7` predictions.
This is not a transfer-aware season simulation.

| Optimizer | Decisions | Mean expected realized | Mean realized | Mean autosub points | Unreplaced-starter rate | Mean bank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 expected-realized | 3 | 49.3856 | 61.00 | 6.33 | 0.667 | 0.0 |
| D1 mean-only MILP | 3 | 48.6920 | 59.33 | 1.33 | 0.667 | 10.0 |

Paired D2 minus D1 realized comparison:

- Mean common-evaluator expected-realized difference: `+0.6936`.
- Median common-evaluator expected-realized difference: `+0.4565`.
- Mean realized difference: `+1.67`.
- Median realized difference: `+2.00`.
- Improved / tied / worse: `66.7% / 33.3% / 0.0%`.
- Block-bootstrap CI: `[0.0, 3.0]`.
- Captain agreement: `1.000`.
- Vice-captain agreement: `1.000`.
- Mean squad overlap: `0.800`.
- Mean lineup overlap: `0.970`.
- Mean bench overlap: `0.250`.

Because this contains only three GW1 folds, it is useful acceptance evidence but not a definitive
performance claim. The bootstrap interval is especially weak with three observations. This remains
a weekly-reset benchmark, not a transfer-aware season simulation.

## Official 2026/27 GW1 Run

Run:

```bash
uv run fpl refresh-operational \
  --season 2026-27 \
  --target-gameweek 1 \
  --run-id phase9b_real_2026_27_gw1_optimizer_conditional_stabilized \
  --force
```

Result:

- State: `SUCCEEDED`.
- Optimizer: `D2_EXPECTED_REALIZED_POINTS`.
- Solver/search status: `heuristic_feasible`.
- Unique legal squads evaluated: `5,907`.
- Scenario count per evaluator call: `32,768`.
- Probability mass: `1.0`.
- Search termination: `configured_iteration_bound_reached`.
- Formation: `3-5-2`.
- Cost: `995`.
- Bank: `5`.
- Nominal starting-XI xPoints: `39.3165`.
- Expected active-starter points: `39.3165`.
- Expected autosub contribution: `5.5892`.
- Expected captain bonus: `4.7069`.
- Expected vice-captain fallback contribution: `0.4536`.
- Expected realized total: `50.0662`.
- Probability all starters appear: `0.0057`.
- Expected automatic substitutions: `1.6587`.
- Probability of an unreplaced starter: `0.1779`.

Selected squad:

| Player | Position | Team | Opponent | Price | xPts | Role |
| --- | --- | --- | --- | ---: | ---: | --- |
| Mbeumo | MID | Manchester United | HUL (A) | 80 | 4.7250 | Captain |
| B.Fernandes | MID | Manchester United | HUL (A) | 120 | 4.9625 | Vice-captain |
| Watkins | FWD | Aston Villa | BHA (A) | 80 | 5.0000 | Starter |
| Pecsi | GKP | Liverpool | NEW (A) | 40 | 0.0500 | Starter |
| Tarkowski | DEF | Everton | CRY (H) | 60 | 3.0000 | Starter |
| Gabriel | DEF | Arsenal | COV (H) | 80 | 4.2250 | Starter |
| Muñoz | DEF | Crystal Palace | EVE (A) | 55 | 2.8250 | Starter |
| Eze | MID | Arsenal | COV (H) | 65 | 3.8500 | Starter |
| Cunha | MID | Manchester United | HUL (A) | 80 | 4.0250 | Starter |
| Semenyo | MID | Manchester City | BOU (H) | 85 | 4.3875 | Starter |
| Igor Jesus | FWD | Nottingham Forest | LEE (H) | 60 | 3.8250 | Starter |
| Raya | GKP | Arsenal | COV (H) | 60 | 3.3875 | Bench GKP |
| Kayode | DEF | Brentford | TOT (H) | 45 | 2.7875 | Bench 1 |
| O'Shea | DEF | Ipswich | SUN (H) | 40 | 2.5125 | Bench 2 |
| Mateo Joseph | FWD | Leeds | NFO (A) | 45 | 0.1625 | Bench 3 |

## D1 Versus D2 Under One Evaluator

Both official selections were evaluated with the same exact 32,768-state evaluator, player
forecasts, independence assumption, substitution rules, and tolerances:

| Metric | D1 seed | D2 final | D2 minus D1 |
| --- | ---: | ---: | ---: |
| Nominal starting-XI xPoints | 42.6615 | 39.3165 | -3.3450 |
| Expected active-starter points | 42.6615 | 39.3165 | -3.3450 |
| Expected autosub contribution | 0.6530 | 5.5892 | +4.9362 |
| Expected captain bonus | 4.7459 | 4.7069 | -0.0390 |
| Expected vice fallback | 0.1147 | 0.4536 | +0.3388 |
| Expected realized total | 48.1751 | 50.0662 | **+1.8911** |
| Expected autosubs | 0.3201 | 1.6587 | +1.3386 |
| All starters appear | 0.3610 | 0.0057 | -0.3553 |
| At least one unreplaced starter | 0.4505 | 0.1779 | -0.2726 |
| Cost | 995 | 995 | 0 |
| Bank | 5 | 5 | 0 |

Both formations are `3-5-2`. D1 captains Watkins with B.Fernandes vice; D2 captains Mbeumo with
B.Fernandes vice. D1 benches Pecsi, McNair, Amenda, and Mateo Joseph. D2 starts Pecsi and benches
Raya first, followed by Kayode, O'Shea, and Mateo Joseph.

The Pecsi/Raya order is legal but is not the best order for the returned squad under the exact
evaluator. This is a bounded approximate lineup-search limitation, not support for Pecsi having
higher conditional xPoints.

## McNair Acceptance Result

McNair remains eligible and was present in the D1 seed. D2 replaced him through ordinary legal
one-swap search because another squad scored better under the exact evaluator. No player-specific
blacklist, projection threshold, team rule, name rule, or code rule was added.

## Unused-Bank Acceptance Result

D2 does not reward budget spending by itself. Tests retain a cheaper squad when football value is
equivalent. The official D1 seed and D2 both leave `5` tenths; D2 improves the common-evaluator
expected realized total by `1.8911`.

## Frontend Contract

The existing `phase9_frontend_v1` contract is preserved. `optimized_lineup.csv` now carries additive
expected-realized diagnostics, including optimizer variant, autosub contribution, captain bonus,
vice-captain fallback contribution, expected substitutions, unreplaced-starter probability, solver
name, and optimality scope.

The frontend displays these values compactly above the squad table. The page design and opponent
display were otherwise preserved.

## Verification

Commands run:

```bash
uv run pytest -q tests/test_phase6_xpoints.py tests/test_phase7_decision.py
uv run pytest -q
uv run ruff check .
uv run fpl validate-data
uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25,2025-26
uv run fpl backtest-xpoints --seasons 2022-23,2023-24,2024-25,2025-26 --test-seasons 2023-24,2024-25,2025-26 --mode gw1 --run-id phase9b12_xpoints_gw1
uv run fpl backtest-decisions --seasons 2023-24,2024-25,2025-26 --mode gw1 --run-id phase9b13_conditional_stabilized_decisions_gw1
uv run fpl refresh-operational --season 2026-27 --target-gameweek 1 --run-id phase9b_real_2026_27_gw1_optimizer_conditional_stabilized --force
git diff --check
git check-ignore -v ...
```

Observed results:

- Focused conditional-xPoints and optimizer tests: exit `0`, `40 passed in 48.37s`.
- `uv run pytest -q`: exit `0`, `162 passed in 960.13s`.
- `uv run ruff check .`: exit `0`.
- `uv run fpl validate-data`: exit `0`, `1 warning, 0 errors`; warning is raw Vaastav `xP`
  presence only.
- `uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25,2025-26`: exit `0`.
- Refreshed three-fold GW1 xPoints artifact: exit `0`, 3 folds.
- Frozen conditional-stabilized GW1 comparison: exit `0`, 6 decisions.
- Official conditional-stabilized operational refresh: exit `0`, state `SUCCEEDED`.
- Frontend source and contract were unchanged, so npm lint/build were not rerun.
- `git diff --check`: exit `0`.
- Representative `git check-ignore -v`: confirmed raw, normalized, report, operational, frontend
  synced, build, and dependency artifacts are ignored.

## Known Limitations

- D2 uses independent player appearance scenarios; it does not yet model correlated absences.
- D2 is a bounded local-search challenger, not a globally proven stochastic MILP optimum.
- The official search reached its configured three-iteration bound; it did not establish even a
  one-swap local optimum.
- The returned official lineup is not optimal even within its selected squad: swapping Raya into
  the starting lineup improves the exact evaluator by `0.0176`. Correcting the bounded lineup
  shortlist was outside this closure scope.
- Complete exact evaluation and full-pool search increase runtime. The official D2 search took about
  114 seconds, and the full test suite took about 16 minutes on the development machine.
- Historical evidence here is GW1-only across three seasons.
- Weekly-reset historical decisions are not equivalent to a transfer-aware season strategy.
- Bench Boost, Triple Captain, Free Hit, Wildcard, transfers, hits, price-change mechanics, and
  multi-gameweek planning remain out of scope.
- The public site should not be automated until this local result is reviewed.

## Verdict

The conditional-xPoints fix is sound, but D2 should remain an experimental challenger because the
bounded lineup shortlist returned a goalkeeper order that is `0.0176` worse under its own exact
evaluator. Phase 9B2 must wait.
