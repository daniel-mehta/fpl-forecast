# Phase 4 Team Strength Report

## 1. Scope And Non-Goals

Phase 4 builds leakage-safe fixture-grain team goal and probability baselines. It
does not model player minutes, player expected points, shots, bonus, simulations,
optimization, scheduling, dashboards, paid data, or Phase 5.

The Phase 4 target is team fixture goals and fixture probabilities. These metrics
must not be compared numerically with Phase 3 player-point baselines.

## 2. Data Grain And Leakage Controls

The modeling table has one row per fixture with stable home and away team UIDs,
source team IDs, kickoff time, information cutoff, source availability time,
finished/result-valid flags, goals, source version, and raw snapshot path.

Real historical source: `data/normalized/phase2/dim_fixture.parquet`. It contains
1,140 fixtures across `2022-23`, `2023-24`, and `2024-25`.

Training fixtures are included only when:

```text
source_available_time < information_cutoff
```

All fixtures in a season-gameweek fold are predicted from the same cutoff. A
result from an early match in the same deadline block cannot train a later match
in that block. Frozen predictions exclude `home_goals`, `away_goals`, match
outcome, and clean-sheet target columns.

Historical schedule limitation: Vaastav historical fixture schedules are
retrospective. The backtest is result-leakage-safe, but it may know final
postponed or rearranged fixture assignments that were not known at the original
deadline. Prospective operation should use archived pre-deadline FPL fixture
snapshots.

## 3. Model Formulas

`T0_LEAGUE_HOME_AWAY`:

```text
lambda_home = mean(training home goals)
lambda_away = mean(training away goals)
```

`T1_SHRUNK_ROLLING_TEAM_RATE` uses the last 8 available team fixtures by side and
shrinks toward league home/away rates with 6 pseudo-matches:

```text
shrunk_rate = (observed_goals + shrink_matches * league_rate)
              / (observed_matches + shrink_matches)
lambda_home = mean(home_team_home_attack, away_team_away_conceding)
lambda_away = mean(away_team_away_attack, home_team_home_conceding)
```

`T2_REGULARIZED_ATTACK_DEFENCE` is a weighted ridge-penalized Poisson
attack-defence model:

```text
eta_home = intercept + home_advantage + attack[home] + defence[away]
eta_away = intercept + attack[away] + defence[home]
lambda = exp(eta)
```

For goal observations `i`, the fitted objective is:

```text
loss = sum_i w_i * (exp(eta_i) - y_i * eta_i + log(y_i!))
       + 0.5 * ridge * (sum_t attack_t^2 + sum_t defence_t^2)
```

Only `attack` and `defence` are penalized. `intercept` and `home_advantage` are
unpenalized. Identifiability is enforced by parameterizing the final team's
attack and defence as the negative sum of the other team effects, so
`sum(attack) = 0` and `sum(defence) = 0`. Higher `attack` means stronger scoring.
Higher `defence` means a team concedes more and is therefore defensively weaker.

Recency weights use `source_available_time` relative to the fold cutoff:

```text
w_i = 0.5 ** (age_days_i / 365)
```

The solver is deterministic Newton iteration with backtracking line search,
`t2_ridge_penalty = 10.0`, `t2_max_iterations = 100`, and
`t2_convergence_tolerance = 1e-7`. A nonconverged fit raises an error instead of
emitting predictions. Expected goals are clipped to `[0.05, 5.0]`; the regenerated
rolling and GW1 runs clipped zero home and zero away predictions.

T2 convergence diagnostics:

| Run | T2 folds | Iterations | Max gradient | Home clipped | Away clipped |
| --- | ---: | --- | ---: | ---: | ---: |
| `phase4_team_rolling_poisson_v2` | 76 | 4 to 4 | 5.49e-08 | 0 | 0 |
| `phase4_team_gw1_poisson_v2` | 2 | 4 to 4 | 2.68e-08 | 0 | 0 |

## 4. Probability Outputs

Each frozen fixture prediction contains expected goals, marginal home and away
goal probabilities for 0 through 8 plus `9+`, home and away clean-sheet
probabilities, and home/draw/away probabilities. Goal tails are retained. Artifact
checks verified probability sums with maximum deviations below `4e-16`.

## 5. Rolling Backtest

Run ID: `phase4_team_rolling_poisson_v2`.

```text
folds=76
frozen_rows=2280
scored_rows=2280
cutoff_min=2023-08-11T19:00:00+00:00
cutoff_max=2025-05-25T15:00:00+00:00
```

Expected-goal metrics:

| Model | Fixtures | Goal MAE | Goal RMSE | Bias | Poisson NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | 760 | 1.0269 | 1.2864 | -0.0540 | 1.5793 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 760 | 0.9847 | 1.2421 | -0.0183 | 1.5430 |
| T2_REGULARIZED_ATTACK_DEFENCE | 760 | 0.9489 | 1.2060 | -0.0421 | 1.5162 |

Home and away splits:

| Model | Side | Fixtures | Goal MAE | Goal RMSE | Bias | Poisson NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | home | 760 | 1.0898 | 1.3296 | 0.0172 | 1.6236 |
| T0_LEAGUE_HOME_AWAY | away | 760 | 0.9641 | 1.2417 | -0.1253 | 1.5349 |
| T1_SHRUNK_ROLLING_TEAM_RATE | home | 760 | 1.0450 | 1.2873 | 0.0211 | 1.5896 |
| T1_SHRUNK_ROLLING_TEAM_RATE | away | 760 | 0.9243 | 1.1951 | -0.0576 | 1.4965 |
| T2_REGULARIZED_ATTACK_DEFENCE | home | 760 | 0.9975 | 1.2424 | 0.0131 | 1.5547 |
| T2_REGULARIZED_ATTACK_DEFENCE | away | 760 | 0.9003 | 1.1685 | -0.0973 | 1.4777 |

Season splits:

| Season | Model | Fixtures | Goal MAE | Goal RMSE | Bias | Poisson NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2023-24 | T0_LEAGUE_HOME_AWAY | 380 | 1.0517 | 1.3309 | -0.1644 | 1.6179 |
| 2023-24 | T1_SHRUNK_ROLLING_TEAM_RATE | 380 | 1.0099 | 1.2790 | -0.0985 | 1.5754 |
| 2023-24 | T2_REGULARIZED_ATTACK_DEFENCE | 380 | 0.9717 | 1.2373 | -0.1354 | 1.5477 |
| 2024-25 | T0_LEAGUE_HOME_AWAY | 380 | 1.0022 | 1.2403 | 0.0563 | 1.5406 |
| 2024-25 | T1_SHRUNK_ROLLING_TEAM_RATE | 380 | 0.9594 | 1.2041 | 0.0620 | 1.5106 |
| 2024-25 | T2_REGULARIZED_ATTACK_DEFENCE | 380 | 0.9262 | 1.1739 | 0.0512 | 1.4847 |

Fallback splits:

| Fallback involved | Model | Fixtures | Goal MAE | Goal RMSE | Bias | Poisson NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| False | T0_LEAGUE_HOME_AWAY | 735 | 1.0268 | 1.2788 | -0.0537 | 1.5767 |
| False | T1_SHRUNK_ROLLING_TEAM_RATE | 735 | 0.9853 | 1.2362 | -0.0178 | 1.5422 |
| False | T2_REGULARIZED_ATTACK_DEFENCE | 735 | 0.9510 | 1.2019 | -0.0413 | 1.5170 |
| True | T0_LEAGUE_HOME_AWAY | 25 | 1.0308 | 1.4921 | -0.0635 | 1.6549 |
| True | T1_SHRUNK_ROLLING_TEAM_RATE | 25 | 0.9655 | 1.4034 | -0.0334 | 1.5680 |
| True | T2_REGULARIZED_ATTACK_DEFENCE | 25 | 0.8889 | 1.3224 | -0.0653 | 1.4927 |

Clean-sheet metrics:

| Model | Team-fixture rows | Brier | Log loss | Predicted rate | Observed rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | 1520 | 0.1727 | 0.5298 | 0.2269 | 0.2204 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 1520 | 0.1675 | 0.5140 | 0.2230 | 0.2204 |
| T2_REGULARIZED_ATTACK_DEFENCE | 1520 | 0.1643 | 0.5028 | 0.2454 | 0.2204 |

Match-outcome metrics:

| Model | Fixtures | Log loss | Brier | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | 760 | 1.0696 | 0.6477 | 0.4342 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 760 | 1.0118 | 0.6044 | 0.5184 |
| T2_REGULARIZED_ATTACK_DEFENCE | 760 | 0.9679 | 0.5748 | 0.5500 |

Block-bootstrap goal-MAE differences versus T0:

| Model | Mean diff | 95% CI | Blocks |
| --- | ---: | --- | ---: |
| T1_SHRUNK_ROLLING_TEAM_RATE | -0.0411 | [-0.0526, -0.0304] | 76 |
| T2_REGULARIZED_ATTACK_DEFENCE | -0.0754 | [-0.0967, -0.0538] | 76 |

## 6. GW1 Backtest

Run ID: `phase4_team_gw1_poisson_v2`.

```text
folds=2
frozen_rows=60
scored_rows=60
cutoff_min=2023-08-11T19:00:00+00:00
cutoff_max=2024-08-16T19:00:00+00:00
```

Expected-goal metrics:

| Model | Fixtures | Goal MAE | Goal RMSE | Bias | Poisson NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | 20 | 0.9145 | 1.1556 | 0.2546 | 1.4398 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 20 | 0.8742 | 1.1210 | 0.3064 | 1.4083 |
| T2_REGULARIZED_ATTACK_DEFENCE | 20 | 0.7571 | 0.9986 | 0.2471 | 1.3064 |

Home and away splits:

| Model | Side | Fixtures | Goal MAE | Goal RMSE | Bias | Poisson NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | home | 20 | 1.1286 | 1.3782 | 0.4257 | 1.5763 |
| T0_LEAGUE_HOME_AWAY | away | 20 | 0.7004 | 0.8782 | 0.0836 | 1.3033 |
| T1_SHRUNK_ROLLING_TEAM_RATE | home | 20 | 1.0912 | 1.3491 | 0.4716 | 1.5472 |
| T1_SHRUNK_ROLLING_TEAM_RATE | away | 20 | 0.6572 | 0.8327 | 0.1412 | 1.2693 |
| T2_REGULARIZED_ATTACK_DEFENCE | home | 20 | 0.9916 | 1.2318 | 0.4071 | 1.4321 |
| T2_REGULARIZED_ATTACK_DEFENCE | away | 20 | 0.5226 | 0.6907 | 0.0872 | 1.1808 |

Season splits:

| Season | Model | Fixtures | Goal MAE | Goal RMSE | Bias | Poisson NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2023-24 | T0_LEAGUE_HOME_AWAY | 10 | 0.9289 | 1.2652 | 0.0263 | 1.5067 |
| 2023-24 | T1_SHRUNK_ROLLING_TEAM_RATE | 10 | 0.8607 | 1.1913 | 0.0675 | 1.4365 |
| 2023-24 | T2_REGULARIZED_ATTACK_DEFENCE | 10 | 0.7550 | 1.0924 | 0.0007 | 1.3574 |
| 2024-25 | T0_LEAGUE_HOME_AWAY | 10 | 0.9000 | 1.0344 | 0.4829 | 1.3730 |
| 2024-25 | T1_SHRUNK_ROLLING_TEAM_RATE | 10 | 0.8878 | 1.0461 | 0.5454 | 1.3801 |
| 2024-25 | T2_REGULARIZED_ATTACK_DEFENCE | 10 | 0.7591 | 0.8949 | 0.4936 | 1.2555 |

Fallback splits:

| Fallback involved | Model | Fixtures | Goal MAE | Goal RMSE | Bias | Poisson NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| False | T0_LEAGUE_HOME_AWAY | 16 | 0.8237 | 1.0582 | 0.2988 | 1.3706 |
| False | T1_SHRUNK_ROLLING_TEAM_RATE | 16 | 0.8021 | 1.0547 | 0.3513 | 1.3607 |
| False | T2_REGULARIZED_ATTACK_DEFENCE | 16 | 0.7168 | 0.9623 | 0.2845 | 1.2782 |
| True | T0_LEAGUE_HOME_AWAY | 4 | 1.2776 | 1.4824 | 0.0780 | 1.7170 |
| True | T1_SHRUNK_ROLLING_TEAM_RATE | 4 | 1.1626 | 1.3543 | 0.1271 | 1.5984 |
| True | T2_REGULARIZED_ATTACK_DEFENCE | 4 | 0.9183 | 1.1323 | 0.0978 | 1.4195 |

Clean-sheet metrics:

| Model | Team-fixture rows | Brier | Log loss | Predicted rate | Observed rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | 40 | 0.1964 | 0.5871 | 0.2325 | 0.2500 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 40 | 0.1774 | 0.5356 | 0.2266 | 0.2500 |
| T2_REGULARIZED_ATTACK_DEFENCE | 40 | 0.1541 | 0.4707 | 0.2543 | 0.2500 |

Match-outcome metrics:

| Model | Fixtures | Log loss | Brier | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | 20 | 1.0886 | 0.6615 | 0.4000 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 20 | 0.9867 | 0.5843 | 0.5000 |
| T2_REGULARIZED_ATTACK_DEFENCE | 20 | 0.8514 | 0.4861 | 0.7000 |

GW1 has only two validation blocks, so this is descriptive evidence only. The
corrected T2 fit still overpredicts total goals in GW1 by +0.2471 per team, with
the bias concentrated in the 2024-25 fold (+0.4936). It should not be described
as fully calibrated.

## 7. Clean-Sheet Calibration

Rolling run, all nonempty bins:

| Model | Bin | Lower | Upper | Rows | Mean predicted | Observed | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | 1 | 0.1000 | 0.2000 | 760 | 0.1876 | 0.2092 | 0.0216 |
| T0_LEAGUE_HOME_AWAY | 2 | 0.2000 | 0.3000 | 760 | 0.2662 | 0.2316 | -0.0347 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 0 | 0.0000 | 0.1000 | 5 | 0.0917 | 0.2000 | 0.1083 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 1 | 0.1000 | 0.2000 | 569 | 0.1661 | 0.1283 | -0.0378 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 2 | 0.2000 | 0.3000 | 791 | 0.2439 | 0.2693 | 0.0254 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 3 | 0.3000 | 0.4000 | 149 | 0.3259 | 0.3154 | -0.0104 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 4 | 0.4000 | 0.5000 | 6 | 0.4232 | 0.1667 | -0.2566 |
| T2_REGULARIZED_ATTACK_DEFENCE | 0 | 0.0000 | 0.1000 | 106 | 0.0733 | 0.0755 | 0.0022 |
| T2_REGULARIZED_ATTACK_DEFENCE | 1 | 0.1000 | 0.2000 | 442 | 0.1534 | 0.1290 | -0.0244 |
| T2_REGULARIZED_ATTACK_DEFENCE | 2 | 0.2000 | 0.3000 | 522 | 0.2488 | 0.2337 | -0.0151 |
| T2_REGULARIZED_ATTACK_DEFENCE | 3 | 0.3000 | 0.4000 | 324 | 0.3440 | 0.3241 | -0.0199 |
| T2_REGULARIZED_ATTACK_DEFENCE | 4 | 0.4000 | 0.5000 | 114 | 0.4369 | 0.3333 | -0.1036 |
| T2_REGULARIZED_ATTACK_DEFENCE | 5 | 0.5000 | 0.6000 | 12 | 0.5253 | 0.4167 | -0.1086 |

GW1 run, all nonempty bins:

| Model | Bin | Lower | Upper | Rows | Mean predicted | Observed | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0_LEAGUE_HOME_AWAY | 1 | 0.1000 | 0.2000 | 20 | 0.1873 | 0.3000 | 0.1127 |
| T0_LEAGUE_HOME_AWAY | 2 | 0.2000 | 0.3000 | 20 | 0.2776 | 0.2000 | -0.0776 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 0 | 0.0000 | 0.1000 | 1 | 0.0995 | 0.0000 | -0.0995 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 1 | 0.1000 | 0.2000 | 13 | 0.1620 | 0.1538 | -0.0082 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 2 | 0.2000 | 0.3000 | 21 | 0.2434 | 0.2857 | 0.0423 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 3 | 0.3000 | 0.4000 | 4 | 0.3173 | 0.2500 | -0.0673 |
| T1_SHRUNK_ROLLING_TEAM_RATE | 4 | 0.4000 | 0.5000 | 1 | 0.4788 | 1.0000 | 0.5212 |
| T2_REGULARIZED_ATTACK_DEFENCE | 0 | 0.0000 | 0.1000 | 3 | 0.0689 | 0.0000 | -0.0689 |
| T2_REGULARIZED_ATTACK_DEFENCE | 1 | 0.1000 | 0.2000 | 11 | 0.1639 | 0.0000 | -0.1639 |
| T2_REGULARIZED_ATTACK_DEFENCE | 2 | 0.2000 | 0.3000 | 13 | 0.2593 | 0.1538 | -0.1055 |
| T2_REGULARIZED_ATTACK_DEFENCE | 3 | 0.3000 | 0.4000 | 9 | 0.3269 | 0.6667 | 0.3398 |
| T2_REGULARIZED_ATTACK_DEFENCE | 4 | 0.4000 | 0.5000 | 4 | 0.4618 | 0.5000 | 0.0382 |

T2 improves clean-sheet Brier and log loss in both runs, but the bin tables show
some overprediction in higher rolling bins and noisy GW1 bin gaps. This is useful
benchmark evidence, not a claim of complete calibration.

## 8. Leicester Re-Entry Identity Trace

The 2024-25 GW38 fold uses cutoff `2025-05-25T15:00:00+00:00`, with 1,130
eligible training fixtures.

| Club | Season | Source team ID | Source name | Stable team UID |
| --- | --- | ---: | --- | --- |
| Southampton | 2022-23 | 17 | Southampton | `team_southampton` |
| Southampton | 2024-25 | 17 | Southampton | `team_southampton` |
| Leicester | 2022-23 | 10 | Leicester | `team_leicester` |
| Leicester | 2024-25 | 11 | Leicester | `team_leicester` |

Fold-level T2 rating diagnostics:

| Club | Training fixtures | Positive-weight fixtures | Effective sample size | Fallback |
| --- | ---: | ---: | ---: | --- |
| Southampton | 75 | 75 | 53.8044 | False |
| Leicester | 75 | 75 | 53.8155 | False |

Both clubs contribute 38 fixtures from 2022-23 and 37 fixtures from 2024-25 before
the GW38 cutoff. Leicester's source team ID changed from 10 to 11, but its stable
UID remained `team_leicester`; the earlier 2022-23 history is not lost. A
regression test now covers a relegated/re-entered team whose source ID changes
across seasons.

## 9. Opponent-Adjustment Evidence

The corrected illustration holds the attacking team, cutoff, home/away status, and
attack effect constant, and changes only the opponent defence effect.

Fold: `2024-25_GW38`.

```text
intercept = 0.3188355680
home_advantage = 0.1418311826
Bournemouth attack = -0.0145038139
base without opponent defence = 0.4461629368
```

Counterfactual Bournemouth home expected goals:

| Scenario | Opponent defence effect | Eta | Expected home goals |
| --- | ---: | ---: | ---: |
| vs Arsenal defence | -0.4589828617 | -0.0128199249 | 0.9873 |
| vs Leicester defence | 0.2166577401 | 0.6628206768 | 1.9403 |

Calculation:

```text
eta = intercept + home_advantage + Bournemouth_attack + opponent_defence
lambda = exp(eta)
```

This isolates opponent adjustment. Arsenal's negative defence effect lowers the
same Bournemouth home attack expectation; Leicester's positive defence effect
raises it.

## 10. Current-Team Mapping

The inference command keeps season-mismatch protection:

```text
uv run fpl forecast-team-fixtures --season 2026-27 --gameweek 1 --as-of 2026-07-22T00:00:00Z
Team-fixture forecast failed: Current fixture season mismatch: requested 2026-27, inferred 2025-26.
```

For the local inferred `2025-26` snapshot, it also blocks because Sunderland is
not yet mapped in the audited Phase 2 team identities:

```text
Team-fixture forecast failed: Current teams lack stable identity mapping: Sunderland
```

No `2026-27` forecast was fabricated. A regression test demonstrates the intended
path for a genuinely new promoted team: add the team to stable `dim_team`, map the
current team name to that stable UID, and the forecast then proceeds with the
neutral no-history fallback flag instead of failing the entire command.

## 11. Price Boundary

Current FPL player normalization preserves official price as integer
`price_tenths` with snapshot provenance. Phase 4 team-model feature matrices do
not include price. Public current price, personalized purchase/selling prices,
budget arithmetic, and price-as-feature leakage audits belong to a later optimizer
or decision phase.

## 12. Provenance And Git Hygiene

Each run writes a manifest, frozen fixture predictions, scored fixture
predictions, fit diagnostics, team ratings, expected-goal metrics, side splits,
season splits, fallback splits, clean-sheet metrics, clean-sheet calibration,
match-outcome metrics, and bootstrap comparisons under
`reports/team_backtests/<run_id>/`.

Generated Phase 4 outputs are ignored by Git. Phase 3 remains uncommitted in this
working tree, and Phase 3 plus Phase 4 changes are currently mixed in
`.gitignore`, `README.md`, and `src/fpl_forecast/cli.py`. Use partial staging if
separate Phase 3 and Phase 4 commits are desired.

## 13. Remaining Limitations

The model is still a fixture-grain team baseline, not a player projection model.
It does not ingest lineups, injuries, bookmaker prices, expected-goal event data,
or manually archived prospective schedule snapshots. GW1 evaluation covers only
two validation blocks, so GW1 calibration and bias should be treated as early
evidence rather than stable model truth.
