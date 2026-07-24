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
given appearance. It uses deterministic independent appearance scenarios with a fixed seed when
Monte Carlo truncation is requested. This independence assumption is explicit and remains a
limitation.

D2 is not globally proven optimal for the expected-realized objective. It is globally optimal only
for the D1 seed objective, then heuristic-feasible after bounded expected-realized local search.

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
- Goalkeeper-only substitution.
- Formation-preserving outfield substitutions.
- Bench-order materiality.
- Expected-realized lineup selection.
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
  --run-id phase9b13_decisions_gw1
```

Scope: weekly-reset GW1 decisions only, using frozen `X2_TEAM_CONSTRAINED_SIM_M7` predictions.
This is not a transfer-aware season simulation.

| Optimizer | Decisions | Mean expected realized | Mean realized | Mean autosub points | Unreplaced-starter rate | Mean bank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 expected-realized | 3 | 51.52 | 61.33 | 2.00 | 0.333 | 5.0 |
| D1 mean-only MILP | 3 | 50.91 | 59.33 | 1.33 | 0.667 | 10.0 |

Paired D2 minus D1 realized comparison:

- Mean difference: `+2.00`.
- Median difference: `+1.00`.
- Improved / tied / worse: `66.7% / 33.3% / 0.0%`.
- Block-bootstrap CI: `[0.0, 5.0]`.
- Captain agreement: `1.000`.
- Vice-captain agreement: `1.000`.
- Mean lineup overlap: `0.970`.
- Mean bench overlap: `0.750`.

Because this contains only three GW1 folds, it is useful acceptance evidence but not a definitive
performance claim.

## Official 2026/27 GW1 Run

Run:

```bash
uv run fpl refresh-operational \
  --season 2026-27 \
  --target-gameweek 1 \
  --run-id phase9b_real_2026_27_gw1_optimizer_hardened \
  --force
```

Result:

- State: `SUCCEEDED`.
- Optimizer: `D2_EXPECTED_REALIZED_POINTS`.
- Solver/search status: `heuristic_feasible`.
- Candidate squads evaluated: `25`.
- Scenario count: `512`.
- Formation: `3-5-2`.
- Cost: `1000`.
- Bank: `0`.
- Expected nominal starting-XI points: `44.2656`.
- Expected autosub contribution: `1.8535`.
- Expected captain bonus: `4.8940`.
- Expected vice-captain fallback contribution: `0.3402`.
- Expected realized total: `51.3533`.
- Probability all starters appear: `0.3379`.
- Expected automatic substitutions: `0.6211`.
- Probability of an unreplaced starter: `0.2949`.

Selected squad:

| Player | Position | Team | Opponent | Price | xPts | Role |
| --- | --- | --- | --- | ---: | ---: | --- |
| B.Fernandes | MID | Manchester United | HUL (A) | 120 | 4.9625 | Captain |
| Watkins | FWD | Aston Villa | BHA (A) | 80 | 5.0000 | Vice-captain |
| Raya | GKP | Arsenal | COV (H) | 60 | 3.3875 | Starter |
| Tarkowski | DEF | Everton | CRY (H) | 60 | 3.0000 | Starter |
| Gabriel | DEF | Arsenal | COV (H) | 80 | 4.2250 | Starter |
| Muñoz | DEF | Crystal Palace | EVE (A) | 55 | 2.8250 | Starter |
| Szoboszlai | MID | Liverpool | NEW (A) | 70 | 4.0625 | Starter |
| Cunha | MID | Manchester United | HUL (A) | 80 | 4.0250 | Starter |
| Semenyo | MID | Manchester City | BOU (H) | 85 | 4.3875 | Starter |
| Mbeumo | MID | Manchester United | HUL (A) | 80 | 4.7250 | Starter |
| Igor Jesus | FWD | Nottingham Forest | LEE (H) | 60 | 3.8250 | Starter |
| Pecsi | GKP | Liverpool | NEW (A) | 40 | 0.0500 | Bench GKP |
| Kayode | DEF | Brentford | TOT (H) | 45 | 2.7875 | Bench 1 |
| Mateo Joseph | FWD | Leeds | NFO (A) | 45 | 0.1625 | Bench 2 |
| McNair | DEF | Hull City | MUN (H) | 40 | 0.7125 | Bench 3 |

## Old Versus New Official GW1

Compared with the Phase 9B1.2 M7 mean-only run:

- Starting XI stayed the same.
- Captain changed from Watkins to B.Fernandes.
- Vice-captain changed from B.Fernandes to Watkins.
- Bench changed from Pecsi, McNair, Amenda, Mateo Joseph to Pecsi, Kayode, Mateo Joseph, McNair.
- Cost changed from `995` to `1000`; this was a football-value consequence of selecting Kayode, not
  a budget-spending rule.
- Expected team points changed from `49.3050` mean-only to `51.3533` expected-realized.

## McNair Acceptance Result

McNair remains selectable and remains selected as the final outfield bench player. No player-specific
blacklist, projection threshold, team rule, name rule, or code rule was added.

The hardened optimizer improved the bench ahead of him by selecting Kayode as first outfield bench.
McNair's continued selection reflects the current forecast universe and bounded search, not manual
suppression or forced replacement.

## Unused-Bank Acceptance Result

D2 does not reward budget spending by itself. The historical tests include a case where a cheaper
squad is retained when football value is equivalent. The official GW1 run spends the full budget only
because the selected legal squad improved the expected-realized objective.

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
uv run fpl backtest-decisions --seasons 2023-24,2024-25,2025-26 --mode gw1 --run-id phase9b13_decisions_gw1
uv run fpl refresh-operational --season 2026-27 --target-gameweek 1 --run-id phase9b_real_2026_27_gw1_optimizer_hardened --force
cd frontend && npm run sync-data
cd frontend && npm run lint
cd frontend && npm run build
git diff --check
git check-ignore -v ...
```

Observed results:

- `uv sync --locked`: exit `0`.
- `uv run pytest -q`: exit `0`, `154 passed in 221.48s`.
- `uv run ruff check .`: exit `0`.
- `uv run fpl validate-data`: exit `0`, `1 warning, 0 errors`; warning is raw Vaastav `xP`
  presence only.
- `uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25,2025-26`: exit `0`.
- `npm run sync-data`: exit `0`, synced from `phase9b_real_2026_27_gw1_optimizer_hardened`.
- `npm run lint`: exit `0`.
- `npm run build`: exit `0`.
- `git diff --check`: exit `0`.
- Representative `git check-ignore -v`: confirmed raw, normalized, report, operational, frontend
  synced, build, and dependency artifacts are ignored.

## Known Limitations

- D2 uses independent player appearance scenarios; it does not yet model correlated absences.
- D2 is a bounded local-search challenger, not a globally proven stochastic MILP optimum.
- Historical evidence here is GW1-only across three seasons.
- Weekly-reset historical decisions are not equivalent to a transfer-aware season strategy.
- Bench Boost, Triple Captain, Free Hit, Wildcard, transfers, hits, price-change mechanics, and
  multi-gameweek planning remain out of scope.
- The public site should not be automated until this local result is reviewed.

## Verdict

The optimizer layer is credible enough to commit as Phase 9B1.3. Phase 9B2 automation should still
wait for local review of the official GW1 output.
