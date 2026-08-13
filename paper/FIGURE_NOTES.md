# Figure and Table Notes

This file supplies proposed captions and interpretation boundaries for the preseason technical
paper. All paths are repository-relative. Historical and prospective evidence must remain visually
and narratively separate.

## Table 1 — Historical data coverage

- **Caption:** Leakage-safe historical player-fixture panel coverage and integrity checks by season.
- **Exact source:** `data/normalized/phase2/fact_player_fixture.parquet`; GW1 population counts, where
  available, come from the promoted-hybrid three-fold scored artifact.
- **Population and grain:** All normalized football-player and assistant-manager fixture rows;
  player-fixture grain.
- **Supports:** The scale, identity completeness, positional composition and duplicate-key status of
  the historical evidence base.
- **Does not support:** Forecast accuracy or completeness of players absent from the source.
- **Limitation:** Cold-start/history-active counts are available only for evaluated historical GW1
  seasons and are not counts over the full season panel.
- **Suggested placement:** Data section.

## Table 2 — Team-model comparison

- **Caption:** Rolling team-model metrics on the same 76 deadline blocks and 760 fixtures.
- **Exact source:** `reports/team_backtests/phase4_1_dixon_coles_rolling/`.
- **Population and grain:** Historical Premier League fixtures; fixture and fixture-team-side
  metrics as labelled.
- **Supports:** Selection of T2 over T0/T1 and the negligible, mixed T3 challenger changes.
- **Does not support:** Player-level or prospective accuracy.
- **Limitation:** T3's joint-scoreline NLL is its distinctive dependence diagnostic; do not imply
  that it is a separate population or that tiny differences establish superiority.
- **Suggested placement:** Team-model methods/results section.

## Table 3 — Minutes-model comparison

- **Caption:** M3, M5 and M7 minutes and availability performance, with rolling and historical GW1
  evaluation shown as separate blocks.
- **Exact source:** `reports/minutes_backtests/phase9b12_minutes_rolling/` and
  `reports/minutes_backtests/phase9b12_minutes_gw1/`.
- **Population and grain:** All observed player-fixture rows.
- **Supports:** Different error/calibration trade-offs and the operational role of M7.
- **Does not support:** A claim that M7 wins every historical metric.
- **Limitation:** Aggregate all-observed metrics include many non-participants; GW1 has only three
  folds.
- **Suggested placement:** Availability-model results section.

## Table 4 — xPoints comparison

- **Caption:** Historical xPoints metrics in explicitly separated rolling, legacy GW1 and promoted
  hybrid GW1 blocks.
- **Exact source:** The three season-aware goalkeeper-scoring xPoints run directories listed in the
  table. Their immutable predecessors are classified in `paper/evidence_supersession.csv`.
- **Population and grain:** All observed player-gameweeks.
- **Supports:** Historical model trade-offs and the scope of available hybrid evidence.
- **Does not support:** Direct comparison of rolling legacy metrics with a hybrid simulator that was
  rerun only on GW1 folds.
- **Limitation:** No promoted-hybrid rolling backtest was run for this evidence pack.
- **Suggested placement:** xPoints results section.

## Table 5 — Simulation convergence

- **Caption:** Deterministic hybrid-simulation convergence, runtime and memory evidence.
- **Exact source:** `outputs/operational/validation_runs/
  preseason_sim_hybrid_10000_goalkeeper_corrected_validation_clean_034830b041c1/
  preseason_simulation_convergence.json` and `preseason_simulation_closure.json`.
- **Population and grain:** Current official challenger input, 554 player-fixture rows.
- **Supports:** The declared-tolerance justification for 10,000 production draws.
- **Does not support:** Predictive superiority from increasing draw count.
- **Limitation:** Early distribution differences use 10,000 as reference; the closure row alone uses
  the non-self-referential 20,000 reference. The corrected closure artifact did not record closure
  wall-clock time or process peak RSS, so those cells are intentionally blank rather than copied
  from superseded evidence.
- **Suggested placement:** Simulation methods/validation section.

## Table 6 — Old versus hybrid GW1 folds

- **Caption:** Regression and stability comparison of the old 80-draw and promoted hybrid simulator
  across three historical GW1 folds.
- **Exact source:** Full-precision scored parquets for
  `phase9b12_xpoints_gw1_goalkeeper_corrected_exact` and
  `preseason_sim_hybrid_10000_gw1_three_fold_goalkeeper_corrected_clean_034830b041c1`.
- **Population and grain:** All observed player-gameweeks, separately by fold and pooled.
- **Supports:** Honest disclosure of improved distribution stability and worsened MAE.
- **Does not support:** Broad predictive superiority.
- **Limitation:** Three GW1 folds are a small development sample; 2025-26 informed hardening.
- **Suggested placement:** Simulator validation section.

## Table 7 — Decision-system evidence

- **Caption:** D1 and D2 historical weekly-reset GW1 decision evidence.
- **Exact source:**
  `reports/decision_backtests/phase9b13_goalkeeper_scoring_corrected_exact_decisions_gw1_clean_034830b041c1/`.
- **Population and grain:** One frozen squad decision per historical GW1; three decisions per method.
- **Supports:** Exact paired arithmetic and limited acceptance evidence for D2.
- **Does not support:** Strong statistical or season-long decision superiority.
- **Limitation:** The three-block bootstrap interval is highly unstable; D2 is heuristic rather than
  globally optimal.
- **Supersession:** Pre-fix and earlier corrected runs remain immutable historical records; the
  authoritative mapping and reasons are in `paper/evidence_supersession.csv`.
- **Suggested placement:** Decision-layer validation section.

## Table 8 — Current official GW1 prospective snapshot

- **Caption:** Prospective example, not an accuracy result: frozen 2026-27 GW1 model and decision
  snapshot.
- **Exact source:** `outputs/operational/validation_runs/
  preseason_sim_hybrid_10000_goalkeeper_corrected_validation_clean_034830b041c1/`.
- **Population and grain:** Officially selectable current players; player-gameweek projections and
  one squad decision.
- **Supports:** A concrete example of forecast outputs, lineage and the recommended decision.
- **Does not support:** Any accuracy, calibration or realized-points claim.
- **Limitation:** This is a non-published validation successor. The frozen published predecessor
  remains immutable; the corrected run changes two bench squad members and the bench order while
  retaining the starting XI, formation, captain and vice-captain.
- **Suggested placement:** Prospective example box near the conclusion.

## Figure 1 — System architecture

- **Caption:** Leakage-safe data flow from official and historical inputs through T2 team state, M7
  availability, X2 hybrid player outcomes, D2 decisions and validated publication.
- **Exact source:** Model contracts and the season-aware validation successor's `model_lineage.json`.
- **Population and grain:** Conceptual system diagram.
- **Supports:** Separation of team, player-availability, player-outcome and decision layers.
- **Does not support:** Accuracy or causal attribution.
- **Limitation:** It omits lower-level feature engineering and fallback branches for readability.
- **Suggested placement:** Methods overview.

## Figure 2 — Chronological evaluation design

- **Caption:** Historical training/evaluation chronology and the transition to the frozen prospective
  2026-27 GW1 forecast.
- **Exact source:** Corrected xPoints manifests and corrected convergence/closure evidence listed in
  `paper/evidence_manifest.csv`.
- **Population and grain:** Historical season/gameweek folds plus one prospective forecast.
- **Supports:** Leakage-safe cutoff design and the distinction between development and prospective
  evidence.
- **Does not support:** Treating 2025-26 as an untouched final holdout.
- **Limitation:** The timeline is schematic rather than proportional to fixture counts.
- **Suggested placement:** Evaluation-design section.

## Figure 3 — Team-model comparison

- **Caption:** Comparable rolling goal, clean-sheet and outcome metrics for T0, T1 and selected T2;
  T3 is separately labelled as a low-score-dependence challenger.
- **Exact source:** `reports/team_backtests/phase4_1_dixon_coles_rolling/`.
- **Population and grain:** Same 76 folds; fixture sides for goal/clean-sheet metrics and fixtures for
  outcome metrics.
- **Supports:** T2 selection and the absence of material T3 improvement.
- **Does not support:** Cross-league generalization.
- **Limitation:** Axes differ by metric; compare models within a panel only.
- **Suggested placement:** Team-model results.

## Figure 4 — Historical xPoints comparison

- **Caption:** xPoints MAE and rank correlation in separate rolling and historical GW1 panels.
- **Visible scope:** The rolling panels contain 114 folds across `2023-24` to `2025-26`; the GW1
  panels contain three folds over the same seasons.
- **Exact source:** `phase9b12_xpoints_rolling_goalkeeper_corrected_exact`,
  `phase9b12_xpoints_gw1_goalkeeper_corrected_exact`, and
  `preseason_sim_hybrid_10000_gw1_three_fold_goalkeeper_corrected_clean_034830b041c1`.
- **Population and grain:** All observed player-gameweeks.
- **Supports:** Model trade-offs under each registered evaluation mode.
- **Does not support:** Comparing the hybrid GW1 annotation as if it were a rolling result.
- **Limitation:** No uncertainty interval was available consistently across all plotted series.
- **Suggested placement:** xPoints results.

## Figure 5 — Simulation convergence

- **Caption:** Error stabilization and runtime as joint fixture draws increase; 10,000 is selected and
  20,000 is the closure reference.
- **Exact source:** The corrected convergence and 10,000-versus-20,000 closure JSON artifacts listed
  in `paper/evidence_manifest.csv`.
- **Population and grain:** 554 current player-fixture rows.
- **Supports:** Production draw-count selection against predeclared tolerances.
- **Does not support:** Improved football forecasting skill.
- **Limitation:** Initial and closure points have explicitly different distribution references; the
  corrected closure artifact did not retain closure runtime or process peak RSS.
- **Suggested placement:** Simulation validation.

## Figure 6 — Old versus hybrid trade-offs

- **Caption:** Pooled three-fold GW1 comparison of old 80-draw and promoted 10,000-draw hybrid
  simulation.
- **Exact source:** Full-precision scored xPoints parquets for the two corrected registered runs.
- **Population and grain:** 1,964 all-observed player-gameweeks.
- **Supports:** Distribution gains alongside the visible MAE regression.
- **Does not support:** Statistical superiority.
- **Limitation:** Metrics have different units and are therefore shown as separate small multiples.
- **Suggested placement:** Simulator results.

## Figure 7 — Historical P(5+) calibration

- **Caption:** Fixed-bin P(5+) calibration for old and hybrid M7 simulations on identical historical
  GW1 folds; point area represents bin size.
- **Exact source:** Full-precision `prob_points_ge_5` and official outcomes in the two corrected
  scored parquet artifacts.
- **Population and grain:** All observed player-gameweeks across three GW1 folds.
- **Supports:** Distribution calibration comparison without rounded reconstruction.
- **Does not support:** Prospective 2026-27 calibration.
- **Limitation:** High-probability bins contain few rows and should not be overinterpreted.
- **Suggested placement:** Simulator calibration subsection.

## Figure 8 — Prospective GW1 price versus xPoints

- **Caption:** Prospective 2026-27 GW1 snapshot, not evaluated against outcomes: official price versus
  frozen expected points, coloured by position with appearance probability encoded by opacity.
- **Exact source:** Corrected non-published validation-successor
  `player_gameweek_projections.csv`.
- **Population and grain:** 554 current player-gameweek projections.
- **Supports:** Descriptive structure of the released forecast and price/value landscape.
- **Does not support:** Accuracy, value realization or superiority to external forecasts.
- **Limitation:** Labels identify only a small set of top-ranked players to preserve readability.
- **Suggested placement:** Prospective example or appendix.
