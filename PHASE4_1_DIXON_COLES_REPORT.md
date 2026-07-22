# Phase 4.1 Dixon-Coles Challenger Report

## 1. Hypothesis And Scope

Phase 4.1 adds one isolated fixture-level challenger, `T3_DIXON_COLES`, to test
whether a Dixon-Coles low-score dependence correction improves proper
probabilistic scores relative to the Phase 4 independent-Poisson benchmark,
`T2_REGULARIZED_ATTACK_DEFENCE`.

This experiment does not replace T2. It does not add expected-minutes models,
player expected-points models, betting features, simulations, optimization,
prices, UI work, or Phase 5.

Reference method: Dixon, M. J. and Coles, S. G. (1997), "Modelling Association
Football Scores and Inefficiencies in the Football Betting Market," Journal of
the Royal Statistical Society: Series C, 46(2), 265-280. DOI:
`10.1111/1467-9876.00065`.

## 2. Model Definition

T3 keeps the same attack-defence structure as T2:

```text
eta_home = intercept + home_advantage + attack[home] + defence[away]
eta_away = intercept + attack[away] + defence[home]
lambda_home = exp(eta_home)
lambda_away = exp(eta_away)
```

The Dixon-Coles correction is:

```text
tau(0, 0) = 1 - lambda_home * lambda_away * rho
tau(0, 1) = 1 + lambda_home * rho
tau(1, 0) = 1 + lambda_away * rho
tau(1, 1) = 1 - rho
tau(x, y) = 1 otherwise
```

The joint score probability is:

```text
P_DC(x, y) = Poisson(x | lambda_home) * Poisson(y | lambda_away) * tau(x, y)
```

At `rho = 0`, T3 reproduces the independent Poisson joint model.

## 3. Likelihood And Optimizer

For training fixtures `i`, the fitted objective is:

```text
loss = sum_i w_i * (
           poisson_nll(home_goals_i, lambda_home_i)
         + poisson_nll(away_goals_i, lambda_away_i)
         - log(tau_i)
       )
       + 0.5 * t2_ridge_penalty * (sum_t attack_t^2 + sum_t defence_t^2)
       + 0.5 * t3_rho_penalty * rho^2
```

`attack`, `defence`, and `rho` are penalized. `intercept` and `home_advantage`
are not penalized. The attack and defence identifiability constraints are
unchanged from T2: `sum(attack) = 0` and `sum(defence) = 0`, implemented by
parameterizing the final team effect as the negative sum of the other team
effects.

Recency weights reuse Phase 4's cutoff-relative rule:

```text
w_i = 0.5 ** (age_days_i / 365)
```

T3 initializes `intercept`, `home_advantage`, `attack`, and `defence` from the
converged T2 fit for the same fold. It initializes `rho = 0`. It then uses a
deterministic BFGS line-search optimizer with analytic gradients. The configured
optimizer settings are:

```text
t3_rho_lower_bound = -0.19
t3_rho_upper_bound = 0.035
t3_rho_penalty = 25.0
t3_max_iterations = 200
t3_convergence_tolerance = 1e-4
```

A nonconverged fit, invalid tau, invalid objective, or invalid probability fails
clearly. T3 does not silently fall back to T2 while labeling predictions as T3.

## 4. Rho Validity Handling

`rho` is optimized through a logistic transform into the configured interval
`(-0.19, 0.035)`. The interval permits both negative and positive rho. It is also
conservative for the model's expected-goal clipping range `[0.05, 5.0]`:

- `rho > -1 / max_expected_goals` keeps `tau(0,1)` and `tau(1,0)` positive.
- `rho < 1 / max_expected_goals^2` keeps `tau(0,0)` positive.
- `rho < 1` keeps `tau(1,1)` positive.

The implementation rejects configured bounds that violate those rules.

## 5. Probability Outputs

Frozen T3 rows store the fitted `dixon_coles_rho`, the probability model label,
and corrected low-score probabilities:

```text
joint_prob_0_0
joint_prob_0_1
joint_prob_1_0
joint_prob_1_1
joint_low_score_corner_probability
joint_tail_probability
```

The joint distribution keeps a finite 0-through-8 grid plus a `9+` tail bucket for
probability validation. The Dixon-Coles adjustment changes only the four low-score
cells and preserves Poisson marginals for fixed lambdas, including clean-sheet
marginals. T3 outcome probabilities are computed by applying the low-score
probability deltas to the independent-Poisson win/draw/loss probabilities and then
validating that the three probabilities sum to one.

## 6. Leakage Controls

T3 uses the same Phase 4 fold construction and cutoff rules:

```text
source_available_time < information_cutoff
```

One T3 model is fit per fold from the common pre-deadline information set. Same
season-gameweek rows are excluded from training, so an early completed match in a
deadline block cannot update rho for later fixtures in the same block. Frozen
predictions contain no goals, match outcomes, clean-sheet outcomes, or other
target columns. Targets are joined only after predictions are frozen.

## 7. Tests And Audit Checks

Regression tests cover:

- all five tau cases, `rho = 0`, positive and negative rho, and invalid tau;
- boundary rejection for unsafe rho limits;
- equality with independent Poisson when `rho = 0`;
- only the four low-score cells changing for fixed lambdas;
- joint probability sums, probability bounds, marginals, tail retention, and
  clean-sheet marginal preservation;
- fitted negative rho for excess 0-0 and 1-1 rows;
- rho staying near zero when no low-score correction is informed;
- deterministic repeated fits;
- nonconvergence failure;
- recorded solver diagnostics;
- unchanged T2 expected-goal outputs when T3 is also fit;
- target-free frozen T3 outputs;
- one added model row per eligible fixture;
- same-deadline results not changing T3 rho;
- GW1 excluding target-season results.

Integrated artifact audit:

```text
phase4_1_dixon_coles_rolling
frozen_rows=3040 scored_rows=3040 t3_rows=760
frozen_forbidden_cols=[]
duplicate_scored_keys=0
rho_min_median_max=-0.057071,-0.002852,0.035000
max_recomputed_actual_score_probability_diff=0.0

phase4_1_dixon_coles_gw1
frozen_rows=80 scored_rows=80 t3_rows=20
frozen_forbidden_cols=[]
duplicate_scored_keys=0
rho_min_median_max=0.000991,0.017995,0.035000
max_recomputed_actual_score_probability_diff=0.0
```

Representative hand recomputation from frozen rows:

```text
fold=2023-24_GW01 fixture=2023-24:1 rho=0.035000
tau(0,0)=0.927301
stored joint_prob_0_0=0.04703690
recomputed joint_prob_0_0=0.04703690
joint_sum=1.0
```

Training-fold traces:

```text
ordinary 2023-24_GW09: train=460 test=10 same_block_train=0
GW1 2024-25_GW01: train=760 test=10 same_block_train=0; train seasons=2022-23,2023-24
DGW/rearranged 2024-25_GW24: train=989 test=11 same_block_train=0
```

## 8. Rolling Results

Run ID: `phase4_1_dixon_coles_rolling`.

```text
folds=76
eligible fixtures=760
frozen rows=3040
scored rows=3040
```

Expected-goal guardrails:

| Model | Fixtures | MAE | RMSE | Bias | Poisson NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| T2_REGULARIZED_ATTACK_DEFENCE | 760 | 0.9489 | 1.2060 | -0.0421 | 1.5162 |
| T3_DIXON_COLES | 760 | 0.9490 | 1.2061 | -0.0419 | 1.5163 |

Clean-sheet guardrails:

| Model | Brier | Log loss | Predicted rate | Observed rate |
| --- | ---: | ---: | ---: | ---: |
| T2_REGULARIZED_ATTACK_DEFENCE | 0.1643 | 0.5028 | 0.2454 | 0.2204 |
| T3_DIXON_COLES | 0.1643 | 0.5028 | 0.2454 | 0.2204 |

Joint and outcome metrics:

| Model | Joint NLL | Outcome log loss | Outcome Brier | Draw Brier | Draw predicted | Draw observed | Exact score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2_REGULARIZED_ATTACK_DEFENCE | 3.0324 | 0.9679 | 0.5748 | 0.1760 | 0.2227 | 0.2303 | 0.1237 |
| T3_DIXON_COLES | 3.0332 | 0.9677 | 0.5747 | 0.1759 | 0.2225 | 0.2303 | 0.1211 |

T3 slightly improves rolling outcome log loss, outcome Brier, and draw Brier, but
worsens joint scoreline NLL and exact-score accuracy.

## 9. GW1 Results

Run ID: `phase4_1_dixon_coles_gw1`.

```text
folds=2
eligible fixtures=20
frozen rows=80
scored rows=80
```

Expected-goal guardrails:

| Model | Fixtures | MAE | RMSE | Bias | Poisson NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| T2_REGULARIZED_ATTACK_DEFENCE | 20 | 0.7571 | 0.9986 | 0.2471 | 1.3064 |
| T3_DIXON_COLES | 20 | 0.7576 | 0.9993 | 0.2474 | 1.3069 |

Clean-sheet guardrails:

| Model | Brier | Log loss | Predicted rate | Observed rate |
| --- | ---: | ---: | ---: | ---: |
| T2_REGULARIZED_ATTACK_DEFENCE | 0.1541 | 0.4707 | 0.2543 | 0.2500 |
| T3_DIXON_COLES | 0.1540 | 0.4706 | 0.2542 | 0.2500 |

Joint and outcome metrics:

| Model | Joint NLL | Outcome log loss | Outcome Brier | Draw Brier | Draw predicted | Draw observed | Exact score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2_REGULARIZED_ATTACK_DEFENCE | 2.6129 | 0.8514 | 0.4861 | 0.1809 | 0.2274 | 0.2500 | 0.3000 |
| T3_DIXON_COLES | 2.6117 | 0.8533 | 0.4869 | 0.1816 | 0.2235 | 0.2500 | 0.3000 |

GW1 has only two deadline blocks and is descriptive only. T3 improves GW1 joint
scoreline NLL by a tiny amount but worsens outcome log loss, outcome Brier, and
draw Brier.

## 10. Low-Score Calibration

Rolling low-score calibration:

| Model | Scoreline | Predicted | Observed | Gap |
| --- | --- | ---: | ---: | ---: |
| T2_REGULARIZED_ATTACK_DEFENCE | 0-0 | 0.0526 | 0.0355 | -0.0171 |
| T2_REGULARIZED_ATTACK_DEFENCE | 0-1 | 0.0690 | 0.0605 | -0.0084 |
| T2_REGULARIZED_ATTACK_DEFENCE | 1-0 | 0.0820 | 0.0671 | -0.0149 |
| T2_REGULARIZED_ATTACK_DEFENCE | 1-1 | 0.1025 | 0.1092 | 0.0067 |
| T2_REGULARIZED_ATTACK_DEFENCE | low_score_corner | 0.3060 | 0.2724 | -0.0337 |
| T3_DIXON_COLES | 0-0 | 0.0525 | 0.0355 | -0.0170 |
| T3_DIXON_COLES | 0-1 | 0.0690 | 0.0605 | -0.0085 |
| T3_DIXON_COLES | 1-0 | 0.0820 | 0.0671 | -0.0149 |
| T3_DIXON_COLES | 1-1 | 0.1024 | 0.1092 | 0.0068 |
| T3_DIXON_COLES | low_score_corner | 0.3060 | 0.2724 | -0.0336 |

GW1 low-score calibration:

| Model | Scoreline | Predicted | Observed | Gap |
| --- | --- | ---: | ---: | ---: |
| T2_REGULARIZED_ATTACK_DEFENCE | 0-0 | 0.0572 | 0.0000 | -0.0572 |
| T2_REGULARIZED_ATTACK_DEFENCE | 0-1 | 0.0711 | 0.1000 | 0.0289 |
| T2_REGULARIZED_ATTACK_DEFENCE | 1-0 | 0.0880 | 0.1500 | 0.0620 |
| T2_REGULARIZED_ATTACK_DEFENCE | 1-1 | 0.1050 | 0.2000 | 0.0950 |
| T2_REGULARIZED_ATTACK_DEFENCE | low_score_corner | 0.3214 | 0.4500 | 0.1286 |
| T3_DIXON_COLES | 0-0 | 0.0552 | 0.0000 | -0.0552 |
| T3_DIXON_COLES | 0-1 | 0.0731 | 0.1000 | 0.0269 |
| T3_DIXON_COLES | 1-0 | 0.0900 | 0.1500 | 0.0600 |
| T3_DIXON_COLES | 1-1 | 0.1031 | 0.2000 | 0.0969 |
| T3_DIXON_COLES | low_score_corner | 0.3213 | 0.4500 | 0.1287 |

The low-score changes are very small. T3 does not materially fix the rolling
overprediction of the combined low-score corner.

## 11. Bootstrap T3 Versus T2

Rolling paired block bootstrap, resampling whole season-gameweek blocks:

| Metric | T3 minus T2 | 95% CI |
| --- | ---: | --- |
| joint_score_nll | 0.000879 | [-0.000518, 0.002234] |
| multiclass_log_loss | -0.000134 | [-0.001281, 0.001000] |
| draw_brier | -0.000073 | [-0.000432, 0.000298] |

GW1 paired block bootstrap is descriptive only because there are two blocks:

| Metric | T3 minus T2 | 95% CI |
| --- | ---: | --- |
| joint_score_nll | -0.001186 | [-0.002375, 0.000003] |
| multiclass_log_loss | 0.001864 | [0.000024, 0.003704] |
| draw_brier | 0.000712 | [-0.000005, 0.001429] |

## 12. Rho And Solver Diagnostics

Rolling T3 diagnostics:

```text
rho min=-0.057071 median=-0.002852 max=0.035000
iterations min=14 median=24 max=52
max_abs_gradient max=0.000099
minimum_tau min=0.856726
lambda clipping home=0 away=0
fallback count=0
invalid probability count=0
```

GW1 T3 diagnostics:

```text
rho min=0.000991 median=0.017995 max=0.035000
iterations min=23 max=38
max_abs_gradient max=0.000078
minimum_tau min=0.915988
lambda clipping home=0 away=0
fallback count=0
invalid probability count=0
```

Several early rolling folds hit the positive rho bound, while later folds move
near zero or negative. That is stable enough for a research challenger, but not
strong enough evidence to promote T3 over T2.

## 13. Adoption Decision

`RETAIN AS EXPERIMENTAL CHALLENGER`

The implementation is leakage-safe and probabilistically valid, but the
chronological evidence is mixed and tiny. Rolling T3 improves outcome log loss and
draw Brier by very small amounts, while worsening joint scoreline NLL. GW1 improves
joint scoreline NLL slightly but worsens outcome log loss and draw Brier. T2 remains
the default probability model for MVP work.

## 14. Reproduction Commands

```text
uv sync
uv run pytest -q
uv run ruff check .
uv run fpl validate-data
uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25

uv run fpl backtest-team-model --seasons 2022-23,2023-24,2024-25 --test-seasons 2023-24,2024-25 --mode rolling --run-id phase4_1_dixon_coles_rolling
uv run fpl backtest-team-model --seasons 2022-23,2023-24,2024-25 --test-seasons 2023-24,2024-25 --mode gw1 --run-id phase4_1_dixon_coles_gw1
uv run fpl compare-team-models --run-id phase4_1_dixon_coles_rolling
uv run fpl compare-team-models --run-id phase4_1_dixon_coles_gw1
uv run fpl inspect-team-model --run-id phase4_1_dixon_coles_rolling
uv run fpl inspect-team-model --run-id phase4_1_dixon_coles_gw1
```
