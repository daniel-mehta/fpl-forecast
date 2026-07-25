# Preseason Simulation Semantics and Convergence Audit

## 1. Baseline behavior

This pass was performed before any 2026-27 match outcomes and does not begin Phase 10. The clean
baseline was commit `abf172c` (`Create live-evaluation-protocol.md`). Preflight was:

```text
git status --short --untracked-files=all  # clean
git log --oneline -10                    # HEAD abf172c
git diff --check                         # exit 0
focused baseline tests                   # 22 passed, 22 deselected, exit 0
```

The baseline xPoints configuration was:

| Setting | Baseline |
| --- | --- |
| Simulator | `phase9b13_conditional_sim_v3` |
| Architecture | independent per-player Monte Carlo |
| Draws | 80 |
| Seed | 6060 plus fold/model offsets |
| Default | `X2_TEAM_CONSTRAINED_SIM_M7` |
| Model contract | implicit Phase 9 contract |

A same-input pre-change operational run succeeded as
`preseason_sim_baseline_80_abf172c`. Its public-contract artifacts were hashed before edits. The
frozen public v0.1.0 run remains `official_2026-27_30140778348_1`; its downloaded public artifacts
were read from the public site into `/tmp` only and were not modified.

The 80 player draws are distinct from D2's exact enumeration of `2^15 = 32,768` independent
appearance states for a selected squad. This pass changes the former and leaves the latter
unchanged.

## 2. Complete expectation path

The implemented path before this pass was:

1. `T2_REGULARIZED_ATTACK_DEFENCE` produced home/away Poisson goal lambdas and clean-sheet
   probabilities.
2. `M7_HIERARCHICAL_AVAILABILITY_STATE` produced expected minutes, six state probabilities,
   `p_appearance`, `p_start`, `p_reached_60`, and `p_played_90`.
3. Historical component rates were shrunk, multiplied by expected-minute exposure, and used to
   form attacking weights.
4. Team-constrained goal and assist means were `team lambda * normalized player share`.
5. Clean-sheet and goals-conceded component inputs were multiplied by `p_reached_60`.
6. Saves, cards, own goals, penalties, defensive contributions, and bonus were converted to
   expected event counts from expected-minute exposure.
7. The old simulator independently drew another appearance and 60+ indicator for every player,
   drew player events independently, gated most events by appearance again, and gated clean-sheet
   and goals-conceded draws by 60+ again.
8. Component draw means were summed to total expected points. Tail probabilities and percentiles
   came from the same 80 draws.
9. Conditional-on-appearance xPoints were estimated from the subset of appearance draws and
   empirical-Bayes stabilized.
10. Fixture draws were summed before player-gameweek percentiles were calculated.
11. D2 received full-precision internal `p_appearance`, unconditional expected points, and
    `expected_points_given_appearance`. Public rounded values were not the D2 input.

Values produced at steps 1-6 were analytic expectations. Values at steps 7-9 were Monte Carlo
estimates. The old conditional value was conditional and stabilized; the total was unconditional.
The publication layer displayed sanitized values but did not reconstruct D2 inputs from them.

## 3. Semantic findings

### Confirmed defects

1. **Double appearance discount.** Team-constrained goals/assists were already unconditional
   allocations, while other player event rates already used expected-minute exposure. Multiplying
   sampled events by another appearance draw discounted them again.
2. **Double 60-minute discount.** `clean_sheet_probability` and expected goals-conceded deduction
   events already included `p_reached_60`; the simulator multiplied them by a second sampled 60+
   indicator.
3. **No shared fixture.** Player goals, assists, clean sheets, and goals conceded were simulated
   independently. Player goal draws did not sum to a common team scoreline and opponents could
   receive inconsistent match outcomes.
4. **No row-order-stable seed derivation.** One sequential generator meant row ordering changed
   which random stream a player received.
5. **Noisy conditional contract.** Only 80 draws made conditional values unstable for
   low-appearance players. Stabilization reduced variance but did not make the value a mathematical
   conditional expectation.
6. **Threshold approximations.** Analytic save points used mean saves divided by three rather than
   `E[floor(saves / 3)]`; goals-conceded deductions used lambda divided by two rather than
   `E[floor(goals conceded / 2)]`; defensive-contribution tails were not explicitly conditioned on
   appearance.

### Design limitations, not scoring bugs

- M7 appearance states remain independent between players.
- Bonus remains the documented simplified expected-bonus model; this pass does not implement a
  full fixture BPS ranking simulation.
- Cards, own goals, penalty events, saves, and defensive contributions use shrunk marginal rates.
- When an independently sampled team has a goal but every selectable player is sampled DNP, the
  fixture allocator conditions on one nonzero-appearance scorer. This event is negligible for a
  full team but is recorded as a remaining approximation.
- Official scoring assumptions and T2/M7 parameters were not retuned.

## 4. Corrected expectation definitions

Let `A` be frozen appearance probability, `R60` frozen probability of reaching 60 minutes, `λ_t`
team expected goals, `s_i` player goal share, `a_i` assist share, and `q` assisted-goal rate.

| Component | New unconditional analytic expectation |
| --- | --- |
| Appearance | `A + R60` |
| Goal points | `λ_t * s_i * position_goal_points` |
| Assist points | `λ_t * q * a_i * 3` |
| Clean sheet | `P(opponent goals = 0) * R60 * position_clean_sheet_points` |
| Goals conceded | `-R60 * E[floor(Poisson(λ_opponent) / 2)]` for GKP/DEF |
| Saves | `A * E[floor(Poisson(expected_saves / A) / 3)]` for GKP |
| Penalties | `5 * E[penalty saves] - 2 * E[penalty misses]` |
| Cards | `-E[yellow] - 3 * E[red]` |
| Own goals | `-2 * E[own goals]` |
| Defensive contribution | `A * 2 * P(Poisson(expected contribution / A) >= position threshold)` |
| Bonus | documented simplified `expected_bonus` |

The unconditional total is the sum of these full-precision component means. For `A > 0`, the exact
contract passed to D2 is:

```text
expected_points_given_appearance = expected_points_unconditional / A
```

For `A = 0`, both are zero. Therefore
`expected_points_unconditional - A * expected_points_given_appearance = 0` within floating-point
tolerance. Appearance is applied once. No optimizer or publication code derives this value from a
rounded public xPoints value.

## 5. New architecture

The selected configuration is:

| Setting | Production challenger |
| --- | --- |
| Simulator | `preseason_hybrid_fixture_v1` |
| Architecture | `analytic_means_joint_fixture_monte_carlo` |
| Contract | `xpoints_hybrid_v1` |
| Draws | 10,000 |
| Master seed | 6060 |
| Seed derivation | SHA-256 of master seed, simulator version, model namespace, season/GW, fixture UID, team/player UID, and stream |

Analytic/Rao-Blackwellized means are authoritative for expected points and components. Joint
fixture draws provide scorelines, active-player scorer/assister allocation, shared clean sheets,
shared goals conceded, integer point distributions, tail probabilities, standard deviation, and
intervals. Active-choice weights are iteratively calibrated so unconditional simulated scorer
shares match analytic team shares while DNP players receive no football events.

Fixture and player rows are canonically ordered before streams are derived and restored afterward.
Identical inputs/configuration/seed are exactly reproducible and input row ordering does not change
results. Raw draw matrices remain private and in-memory; public artifacts contain summaries only.
The convergence script permits local diagnostic regeneration without publishing raw draws.

## 6. Unit-test invariants

Hand-calculated coverage now includes:

- certain DNP, 30-minute substitute, 59-, 60-, and 90-minute appearances;
- mixed DNP/substitute/start appearance expectation;
- one expected goal allocated to one certain player and two-player known shares;
- official appearance, clean-sheet, conceded-goal, save, card, own-goal, and defensive-contribution
  boundaries;
- explicit conditional versus unconditional xPoints and no double discount;
- exact analytic component-to-total reconciliation;
- fixture scoreline goal conservation;
- double-gameweek draw aggregation before percentiles;
- deterministic seed behavior and row-order invariance;
- selected 10,000-versus-20,000 slow convergence.

Existing M7 tests verify state probabilities sum to one, expected minutes remain in 0-90, and
unavailable players map to certain DNP/zero minutes. Existing scoring reconstruction tests remain
the authority for official scoring rules.

## 7. Convergence methodology and tolerances

The convergence input was the full-precision M7 player-fixture artifact from the official
challenger input: 554 rows, no 2026-27 outcomes. Counts were 80, 500, 1,000, 5,000, and 10,000.
The initial study used the 10,000 deterministic run as its distribution reference and analytic
component sums as the mean reference. The closure check independently compared 10,000 against a
20,000-draw deterministic reference using the same full-precision inputs and seed policy.

Tolerances were configured before selection:

- median absolute analytic-versus-simulated mean difference `<= 0.02`;
- 95th percentile mean difference `<= 0.05`;
- 95th percentile P(5+) difference `<= 0.01`;
- exact component reconciliation `<= 1e-10`;
- deterministic reruns exactly identical.

The study also measured zero probability, interval endpoints, conditional values, ranks, top-K,
runtime, draw-matrix memory, process peak RSS, and reproducibility. Since analytic expected and
conditional points are authoritative at every count, ranks and D2 mean/conditional inputs are
draw-count invariant by construction.

## 8. Convergence results

| Draws | Runtime, M7 core | Draw matrix | Analytic/sim median | Analytic/sim P95 | P(5+) P95 vs 10k | Zero P95 vs 10k | Interval endpoint P95 | Reproducible |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 80 | 0.44s | 0.08 MiB | 0.1075 | 0.4334 | 0.0641 | 0.0883 | 1.0 | yes |
| 500 | 0.93s | 0.53 MiB | 0.0426 | 0.1776 | 0.0266 | 0.0341 | 1.0 | yes |
| 1,000 | 1.52s | 1.06 MiB | 0.0296 | 0.1253 | 0.0191 | 0.0256 | 1.0 | yes |
| 5,000 | 6.15s | 5.28 MiB | 0.0133 | 0.0585 | 0.0068 | 0.0087 | 0.0 | yes |
| 10,000 | 12.79s | 10.57 MiB | 0.0124 | 0.0473 | reference | reference | reference | yes |

Component reconciliation was exactly zero at every count. Analytic expected points, analytic
conditional points, Spearman rank, and top-15/30/50 were identical across draw counts. Five
thousand passed the P(5+) tolerance but missed the predeclared mean P95 tolerance. Ten thousand is
therefore selected. This is a distribution-stability decision, not a predictive improvement claim.

The peak process RSS reached 437 MiB while the script retained results from all configurations and
their deterministic reruns. A single 10,000-draw M7 matrix was 10.57 MiB. The production operational
run remains practical on a 32 GB Apple Silicon Mac and a standard Actions runner. Fast tests replace
the production count with small deterministic counts; the production convergence check is marked
`slow`.

### 10,000-versus-20,000 closure reference

The non-self-referential closure comparison produced:

| Measurement | 10,000 versus 20,000 |
| --- | ---: |
| P(5+) P95 absolute difference | 0.005250 |
| Zero-point probability P95 absolute difference | 0.006168 |
| Interval-endpoint P95 absolute difference | 0.000000 points |
| Simulated-mean median absolute difference | 0.007850 points |
| Simulated-mean P95 absolute difference | 0.030688 points |
| Simulated-mean Spearman | 0.999863 |
| Top-15 / top-30 / top-50 overlap | 15/15, 29/30, 49/50 |
| 10,000 runtime / draw matrix | 15.39s / 10.57 MiB |
| 20,000 runtime / draw matrix | 27.17s / 21.13 MiB |
| Whole-process peak RSS after 10,000 / 20,000 | 388 MiB / 773 MiB |
| Exact deterministic rerun | yes at both counts |

The 10,000-draw configuration passes the existing, predeclared distribution tolerances against
20,000: P(5+) P95 is below 0.01, and simulated-mean median/P95 differences are below 0.02/0.05.
The zero-probability and interval results are reported diagnostics; no tolerance was added or
changed after observing them. Exact reruns reproduced both summaries and retained draw matrices
byte-for-byte. The 20,000-draw run is a closure reference, not a new production setting.

## 9. Historical comparison

The closure comparison reran exactly the three historical GW1 folds, with identical fold inputs and
training cutoffs for old 80-draw M7 and new hybrid 10,000. Coverage is reported as
`pre_deadline_history_active + cold_start_no_history`; it was identical between simulators.
Zero calibration is `predicted zero rate / actual zero rate (gap)`.

| Fold | Simulator | Rows | Coverage | MAE | RMSE | Bias | Spearman | P(5+) Brier | Zero calibration | Central 80% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-24 GW1 | Old 80 | 658 | 403 + 255 | 1.2281 | 2.0213 | -0.0279 | 0.5030 | 0.06884 | 0.4570 / 0.5669 (-0.1098) | 0.8906 |
| 2023-24 GW1 | Hybrid 10k | 658 | 403 + 255 | 1.3060 | 2.0193 | +0.1703 | 0.5118 | 0.06769 | 0.4625 / 0.5669 (-0.1044) | 0.9043 |
| 2024-25 GW1 | Old 80 | 616 | 471 + 145 | 1.2562 | 1.9901 | -0.0098 | 0.4908 | 0.07432 | 0.4304 / 0.5276 (-0.0972) | 0.9026 |
| 2024-25 GW1 | Hybrid 10k | 616 | 471 + 145 | 1.3135 | 1.9662 | +0.1648 | 0.5026 | 0.07031 | 0.4379 / 0.5276 (-0.0897) | 0.9334 |
| 2025-26 GW1 | Old 80 | 690 | 478 + 212 | 1.4167 | 2.3834 | -0.2482 | 0.4810 | 0.09716 | 0.4735 / 0.5913 (-0.1178) | 0.8551 |
| 2025-26 GW1 | Hybrid 10k | 690 | 478 + 212 | 1.4888 | 2.3599 | -0.0700 | 0.4749 | 0.09342 | 0.4798 / 0.5913 (-0.1115) | 0.8884 |
| Pooled | Old 80 | 1,964 | 1,352 + 612 | 1.3032 | 2.1463 | -0.0996 | 0.4895 | 0.08051 | 0.4545 / 0.5631 (-0.1087) | 0.8819 |
| Pooled | Hybrid 10k | 1,964 | 1,352 + 612 | 1.3726 | 2.1296 | +0.0841 | 0.4937 | 0.07755 | 0.4608 / 0.5631 (-0.1023) | 0.9078 |

Pooled MAE worsens by 0.0694 while RMSE, Spearman, P(5+) Brier, zero-calibration gap, and central
80% coverage improve. Bias changes sign and is 0.0155 closer to zero in absolute terms. The hybrid
has higher MAE in all three folds, lower RMSE and P(5+) Brier in all three, higher central 80%
coverage in all three, and mixed Spearman. Event-level component inputs are unchanged because T2,
M7, and component-rate models were not retuned. This is a regression and stability check, not
evidence of predictive superiority.

## 10. Official GW1 comparison

The final local challenger is `preseason_sim_hybrid_10000_final_v2_abf172c`; it succeeded without
publishing. The clean same-input comparison against `preseason_sim_baseline_80_abf172c` isolates
simulator changes:

| Measurement | Result |
| --- | ---: |
| Rows | 554 versus 554 |
| Expected-minutes maximum absolute difference | 0 |
| xPoints mean difference | +0.2087 |
| xPoints median absolute difference | 0.1961 |
| xPoints P95 absolute difference | 0.6538 |
| xPoints maximum absolute difference | 1.3514 |
| P(5+) P95 absolute difference | 0.1070 |
| P(10+) P95 absolute difference | 0.0214 |
| Rank correlation | 0.9774 |
| Top-15 / 30 / 50 overlap | 11 / 23 / 41 |
| Squad / Starting XI overlap | 9 / 15 and 7 / 11 |
| Captain | Mbeumo -> Saka |
| Vice-captain | B.Fernandes -> B.Fernandes |
| D2 expected-realized value | 50.0837 -> 55.1146 |
| D2 state count | 32,768 -> 32,768 |
| D2 solver status | `heuristic_feasible` -> `heuristic_feasible` |
| D2 optimizer runtime | 132.37s -> 124.69s |

Low-appearance players (`p_appearance < 0.25`, 126 rows) changed by mean +0.0657, median absolute
0.0493, maximum absolute 0.3622. The 71 cold-start/promoted-fallback rows changed by mean +0.1819,
median absolute 0.1973, maximum absolute 0.9427. These are expected areas of sensitivity because
the old double discount was strongest for limited-exposure players.

The published v0.1.0 artifact has 557 rows while the current official snapshot has 554, and
expected minutes differ for some players. Its comparison therefore combines official-source drift
with simulator change: rank correlation 0.9727, top-15/30/50 overlap 13/25/40, squad overlap 7/15,
Starting XI overlap 6/11, and captain/vice B.Fernandes/Watkins versus Saka/B.Fernandes. It is not
presented as a pure simulator comparison. The frozen v0.1.0 files were not overwritten, relabelled,
or deleted.

## 11. Runtime and memory

- M7 core at 10,000 draws: 12.79 seconds for 554 player-fixture rows.
- Single M7 draw matrix: 10.57 MiB (`int16`).
- Convergence process peak RSS while retaining all configurations and reruns: 437 MiB.
- Three-fold historical GW1 closure comparison: approximately 55 seconds.
- Final D2 stage: 124.69 seconds; full-pool search remains the operational bottleneck.
- Raw draws are not written into public artifacts. Diagnostic regeneration is opt-in through
  `scripts/preseason_simulation_convergence.py`.

Full rolling backtests can be memory intensive because the existing runner retains multiple model
draw matrices. This pass deliberately used only the three historical GW1 folds and does not make
production-sized simulation part of fast CI.

## 12. Compatibility with D2 and publication

D2 squad search, fixed-squad lineup refinement, autosub rules, captain/vice behavior, and exact
32,768-state evaluator are unchanged. Its input contract improves: unconditional and
conditional-on-appearance values are analytic, coherent, and passed internally at full precision.
Tail probabilities and intervals do not affect D2 selection.

### Exact trace of unusual D2 choices

The final challenger decision was re-evaluated with the production exact evaluator, using probability
mass 1.0 over all 32,768 independent appearance states.

For Destan versus Igor Jesus:

| Player | Unconditional | Appearance | Conditional | Starting value | Bench/autosub value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Destan | 1.405204 | 0.333000 | 4.219833 | 1.405204 in returned XI | 0.776112 if benched |
| Igor Jesus | 3.547170 | 0.945143 | 3.753051 | 3.547170 if started | 2.987665 in returned order |

Igor is used by any autosub in 0.796063 of states and specifically replaces an absent Destan with
probability 0.298607. The returned Destan-start/Igor-bench decision scores 55.114610057358 exact
expected-realized points. Starting Igor directly and putting Destan in the same bench slot scores
55.045022295876, so the returned ordering improves the exact objective by **0.069587761482**.
The gain comes from preserving Igor's high-appearance conditional value as cover for Destan and
other absences; it is not implied by comparing unconditional xPoints alone.

For Saka versus B.Fernandes captaincy:

| Player | Unconditional | Appearance | Conditional |
| --- | ---: | ---: | ---: |
| Saka | 4.838526 | 0.812972 | 5.951649 |
| B.Fernandes | 5.127670 | 0.946654 | 5.416624 |

Saka captain contributes a 4.838526 captain bonus and Bruno vice contributes a 0.959016 fallback:
`(1 - 0.812972307692308) * 5.127670379378043`. Reversing them contributes a 5.127670 Bruno bonus
and 0.258114 Saka fallback:
`(1 - 0.946654358974359) * 4.838525962818309`. The returned Saka-captain/Bruno-vice decision scores
55.114610057358 versus 54.702852385054 for the reverse, an exact improvement of
**0.411757672304**.

For all four players, full-precision `appearance * conditional` equals the unconditional input.
Appearance is therefore applied exactly once, and no conditional value is reconstructed from
rounded public xPoints. Captain/vice pairs are exhaustively evaluated for the fixed lineup; the
returned fixed-squad lineup is recorded as a `single_change_local_optimum` after 524 exact
evaluations. The two named alternatives were also evaluated directly, so neither choice is caused
by a shortlist omission or an incomplete lineup refinement comparison. This remains a local, not
global, optimality statement for D2 as a whole.

The existing `phase9_frontend_v1` columns remain backward compatible. Additive private columns
record simulator version, architecture, configured draws, master seed, and model contract.
Operational lineage records analytic/simulated components and seed policy. Backtest manifests
record the same configuration and tolerances.

## 13. Remaining limitations

- Appearance states and the D2 evaluator still assume player independence.
- Active-player allocation uses calibrated conditional weights and a negligible all-DNP fallback,
  not a full joint lineup/minutes model.
- Assists use a simplified assisted-goal rate and exclude the scorer; no secondary assists exist.
- Bonus is an analytic simplified expectation, not a joint BPS ranking model.
- Rare-event marginals are not coupled to scoreline/player roles beyond documented gates.
- Clean-sheet and goals-conceded scoring uses the final simulated fixture scoreline with the
  available minutes state rather than exact goal/substitution timing. A player substituted after
  60 minutes may retain a clean sheet despite a later concession, and a player only incurs
  goals-conceded deductions for goals conceded while on the pitch under official scoring. The
  current simulator does not model those event times and therefore approximates both cases using
  the minutes state and final fixture scoreline.
- Exactly three historical GW1 folds were rerun; no broad predictive superiority claim is made.
- Full rolling draw retention should be redesigned or streamed before routine 10,000-draw rolling
  backtests on smaller CI machines.
- No 2026-27 outcome was used.

## 14. Promotion decision

The challenger satisfies the semantic, determinism, reconciliation, convergence, runtime, lineage,
and D2-compatibility gates. Historical MAE regresses in all three folds and is explicitly disclosed;
RMSE, distribution Brier, and coverage improve in all three, while Spearman is mixed. The
10,000-draw hybrid is selected as the production preseason simulator pending review of the material
GW1 squad and ranking changes documented above.

Promotion must create a new model/version freeze. It must not replace the prospective record of
v0.1.0, and no public publication is performed by this pass.
