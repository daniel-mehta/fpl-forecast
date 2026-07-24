# Phase 9B1.3 Optimizer Report

Phase 9B1.3 hardens the weekly GW1 optimizer only. It does not add transfer planning, chips,
multi-gameweek optimization, public automation, deployment, player-specific rules, or manual
exclusions.

## Repository State

- Starting point: local Phase 9B work on `main`, ahead of `origin/main`.
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

The expected-realized evaluator uses M7 appearance probabilities and conditional expected points
given appearance. It enumerates all `2^15 = 32,768` binary appearance states. Scenario probability
mass is checked against one within `1e-12`; the official run produced `0.9999999999999999`.
Automatic substitutions, captaincy, and vice-captain fallback are applied in every state.

The previous 512-draw deterministic Monte Carlo path was removed from D2. The result is exact only
conditional on the supplied forecasts and independent player appearances. Independence is not a
claim about the real world and remains a limitation.

Player `expected_points` is treated as unconditional. Scenario scoring safely derives:

```text
conditional_points_given_appearance =
    expected_points / p_appearance, when p_appearance > 0
```

and uses zero when `p_appearance` is zero. Exact enumeration recovers the original unconditional
expectation, so appearance is not applied twice.

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
  --run-id phase9b13_exact_eval_decisions_gw1
```

Scope: weekly-reset GW1 decisions only, using frozen `X2_TEAM_CONSTRAINED_SIM_M7` predictions.
This is not a transfer-aware season simulation.

| Optimizer | Decisions | Mean expected realized | Mean realized | Mean autosub points | Unreplaced-starter rate | Mean bank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 expected-realized | 3 | 51.9662 | 61.67 | 7.00 | 0.333 | 0.0 |
| D1 mean-only MILP | 3 | 51.1715 | 59.33 | 1.33 | 0.667 | 10.0 |

Paired D2 minus D1 realized comparison:

- Mean common-evaluator expected-realized difference: `+0.7947`.
- Median common-evaluator expected-realized difference: `+0.4653`.
- Mean realized difference: `+2.33`.
- Median realized difference: `+2.00`.
- Improved / tied / worse: `66.7% / 33.3% / 0.0%`.
- Block-bootstrap CI: `[0.0, 5.0]`.
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
  --run-id phase9b_real_2026_27_gw1_optimizer_exact_eval \
  --force
```

Result:

- State: `SUCCEEDED`.
- Optimizer: `D2_EXPECTED_REALIZED_POINTS`.
- Solver/search status: `heuristic_feasible`.
- Unique legal squads evaluated: `5,907`.
- Scenario count per evaluator call: `32,768`.
- Probability mass: `0.9999999999999999`.
- Search termination: `configured_iteration_bound_reached`.
- Formation: `3-5-2`.
- Cost: `1000`.
- Bank: `0`.
- Nominal starting-XI xPoints: `41.0875`.
- Expected active-starter points: `41.0875`.
- Expected autosub contribution: `5.6264`.
- Expected captain bonus: `4.9625`.
- Expected vice-captain fallback contribution: `0.2667`.
- Expected realized total: `51.9431`.
- Probability all starters appear: `0.0059`.
- Expected automatic substitutions: `1.6511`.
- Probability of an unreplaced starter: `0.1595`.

Selected squad:

| Player | Position | Team | Opponent | Price | xPts | Role |
| --- | --- | --- | --- | ---: | ---: | --- |
| B.Fernandes | MID | Manchester United | HUL (A) | 120 | 4.9625 | Captain |
| Watkins | FWD | Aston Villa | BHA (A) | 80 | 5.0000 | Vice-captain |
| Pecsi | GKP | Liverpool | NEW (A) | 40 | 0.0500 | Starter |
| Tarkowski | DEF | Everton | CRY (H) | 60 | 3.0000 | Starter |
| Gabriel | DEF | Arsenal | COV (H) | 80 | 4.2250 | Starter |
| Muñoz | DEF | Crystal Palace | EVE (A) | 55 | 2.8250 | Starter |
| Szoboszlai | MID | Liverpool | NEW (A) | 70 | 4.0625 | Starter |
| Cunha | MID | Manchester United | HUL (A) | 80 | 4.0250 | Starter |
| Semenyo | MID | Manchester City | BOU (H) | 85 | 4.3875 | Starter |
| Mbeumo | MID | Manchester United | HUL (A) | 80 | 4.7250 | Starter |
| Igor Jesus | FWD | Nottingham Forest | LEE (H) | 60 | 3.8250 | Starter |
| Raya | GKP | Arsenal | COV (H) | 60 | 3.3875 | Bench GKP |
| Kayode | DEF | Brentford | TOT (H) | 45 | 2.7875 | Bench 1 |
| O'Shea | DEF | Ipswich | SUN (H) | 40 | 2.5125 | Bench 2 |
| Walle Egeli | FWD | Ipswich | SUN (H) | 45 | 0.8250 | Bench 3 |

## D1 Versus D2 Under One Evaluator

Both official selections were evaluated with the same exact 32,768-state evaluator, player
forecasts, independence assumption, substitution rules, and tolerances:

| Metric | D1 seed | D2 final | D2 minus D1 |
| --- | ---: | ---: | ---: |
| Nominal starting-XI xPoints | 44.4250 | 41.0875 | -3.3375 |
| Expected active-starter points | 44.4250 | 41.0875 | -3.3375 |
| Expected autosub contribution | 0.7345 | 5.6264 | +4.8919 |
| Expected captain bonus | 5.0000 | 4.9625 | -0.0375 |
| Expected vice fallback | 0.1191 | 0.2667 | +0.1476 |
| Expected realized total | 50.2786 | 51.9431 | **+1.6645** |
| Expected autosubs | 0.3201 | 1.6511 | +1.3310 |
| All starters appear | 0.3610 | 0.0059 | -0.3551 |
| At least one unreplaced starter | 0.4505 | 0.1595 | -0.2910 |
| Cost | 995 | 1000 | +5 |
| Bank | 5 | 0 | -5 |

Both formations are `3-5-2`. D1 captains Watkins with B.Fernandes vice; D2 captains B.Fernandes
with Watkins vice. D1 benches Pecsi, McNair, Amenda, and Mateo Joseph. D2 starts Pecsi and benches
Raya first, followed by Kayode, O'Shea, and Walle Egeli.

The Pecsi/Raya ordering is not a named rule: under independent appearances, D2 values Raya's
goalkeeper-only autosub when Pecsi does not appear. It is legal and internally coherent, but it also
shows how strongly the optimizer can exploit supplied appearance probabilities and the independence
assumption.

## McNair Acceptance Result

McNair remains eligible and was present in the D1 seed. D2 replaced him through ordinary legal
one-swap search because another squad scored better under the exact evaluator. No player-specific
blacklist, projection threshold, team rule, name rule, or code rule was added.

## Unused-Bank Acceptance Result

D2 does not reward budget spending by itself. Tests retain a cheaper squad when football value is
equivalent. The official D1 seed leaves `5` tenths and D2 leaves zero only because its selected legal
squad improves common-evaluator expected realized points by `1.6645`.

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
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run fpl validate-data
uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25,2025-26
uv run fpl backtest-decisions --seasons 2023-24,2024-25,2025-26 --mode gw1 --run-id phase9b13_exact_eval_decisions_gw1
uv run fpl refresh-operational --season 2026-27 --target-gameweek 1 --run-id phase9b_real_2026_27_gw1_optimizer_exact_eval --force
cd frontend && npm run sync-data
cd frontend && npm run lint
cd frontend && npm run build
git diff --check
git check-ignore -v ...
```

Observed results:

- `uv sync --locked`: exit `0`.
- Focused optimizer tests: exit `0`, `11 passed, 9 deselected in 6.85s`.
- `uv run pytest -q`: exit `0`, `158 passed in 592.79s`.
- `uv run ruff check .`: exit `0`.
- `uv run fpl validate-data`: exit `0`, `1 warning, 0 errors`; warning is raw Vaastav `xP`
  presence only.
- `uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25,2025-26`: exit `0`.
- Frozen exact GW1 comparison: exit `0`, 6 decisions.
- Official exact operational refresh: exit `0`, state `SUCCEEDED`.
- `npm run sync-data`: exit `0`, synced from `phase9b_real_2026_27_gw1_optimizer_exact_eval`.
- `npm run lint`: exit `0`.
- `npm run build`: exit `0`.
- `git diff --check`: exit `0`.
- Representative `git check-ignore -v`: confirmed raw, normalized, report, operational, frontend
  synced, build, and dependency artifacts are ignored.

## Known Limitations

- D2 uses independent player appearance scenarios; it does not yet model correlated absences.
- D2 is a bounded local-search challenger, not a globally proven stochastic MILP optimum.
- The official search reached its configured three-iteration bound; it did not establish even a
  one-swap local optimum.
- Complete exact evaluation and full-pool search increase runtime. The official D2 search took about
  61 seconds, and the full test suite took about 11.5 minutes on the development machine.
- Historical evidence here is GW1-only across three seasons.
- Weekly-reset historical decisions are not equivalent to a transfer-aware season strategy.
- Bench Boost, Triple Captain, Free Hit, Wildcard, transfers, hits, price-change mechanics, and
  multi-gameweek planning remain out of scope.
- The public site should not be automated until this local result is reviewed.

## Verdict

The optimizer layer is credible enough to commit as Phase 9B1.3. Phase 9B2 automation should still
wait for local review of the official GW1 output.
