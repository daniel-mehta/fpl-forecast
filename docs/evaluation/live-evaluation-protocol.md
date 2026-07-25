# Prospective 2026-27 Live Evaluation Protocol

Status: **pre-outcome protocol, to be frozen before any 2026-27 match result is observed**  
Release under evaluation: `v0.1.0` (annotated tag target `1c83619b4396474298b8d950f60a2aa39435e5f3`)  
First public forecast: 2026-27 Gameweek 1  
Public site: <https://daniel-mehta.github.io/fpl-forecast/>

## 1. Purpose and scope

This protocol pre-registers how the project's frozen, pre-deadline forecasts will be scored against genuine official 2026-27 outcomes. It is intended to prevent retrospective population selection, metric selection, model substitution, forecast regeneration, and silent replacement of a failed forecast by a later forecast.

This is a documentation and audit protocol. It does not authorize forecast regeneration, model or production-code changes, publication, scheduling, or outcome ingestion before fixtures are complete. Prospective results, outcome-conditioned diagnostics, retrospective experiments, and explanatory error labels must always be reported separately.

The first verified public run was `official_2026-27_30140778348_1`: 557 player-gameweek rows, generated at `2026-07-25T02:44:22.547218Z`. Its official inputs were retrieved at `2026-07-25T02:38:58.972668Z` (`bootstrap-static`, SHA-256 `cc52d36a8967f9df898738663ea324aaeeb1fffea828612c409d666fc0c2b8ae`) and `2026-07-25T02:38:58.974287Z` (`fixtures`, SHA-256 `9e7484118381f8202830906ba993c176475d8ca1796571f5dd78cbfc2d73bd3e`). The private run and publication manifest are ignored operational artifacts and are not present in this checkout. They must be obtained from the original publication/audit retention location and frozen; they must not be reconstructed from results or substituted with the locally available rehearsal run.

## 2. Implemented system being evaluated

| Layer | Implemented forecast, target, and grain |
| --- | --- |
| Team | `T2_REGULARIZED_ATTACK_DEFENCE`; home and away goals at fixture-team goal-side grain. Outputs include expected goals, Poisson goal probabilities 0 through 8 and `9+`, clean-sheet probabilities, and home/draw/away probabilities. Joint scoreline output exists in later backtests where preserved. |
| Minutes | `M7_HIERARCHICAL_AVAILABILITY_STATE`; player-fixture expected minutes and six-state probabilities (`DNP`, `SUB_LT60`, `SUB_60_PLUS`, `START_LT60`, `START_60_89`, `START_90`). Operational fields include `expected_minutes`, `p_appearance`, `p_start`, `p_reached_60`, and `p_played_90` in private artifacts; the public contract exposes only `expected_minutes`, `p_appearance`, and `p_start`. |
| xPoints | `X2_TEAM_CONSTRAINED_SIM_M7`; player-fixture simulation followed by player-gameweek aggregation. Public distribution fields are `expected_points`, `points_std`, `points_p10`, `points_p50`, `points_p90`, `prob_points_ge_5`, and `prob_points_ge_10`; private model artifacts also support `points_p25`, `points_p75`, `prob_points_eq_0`, and `prob_points_ge_1`. |
| Decision | `D2_EXPECTED_REALIZED_POINTS`; one 15-player squad, 11-player Starting XI, ordered four-player bench, captain, and vice-captain per gameweek. D2 starts from the exact full-candidate `D1_MEAN_ONLY_MILP` solution and applies deterministic bounded one-swap search plus fixed-squad lineup refinement under exact enumeration of 32,768 independent appearance states. D2 is heuristic-feasible, not globally optimal. |

The public artifact contract is exactly `operational_status.json`, `player_gameweek_projections.csv`, `optimized_squad.csv`, `optimized_lineup.csv`, `model_comparison.csv`, `data_freshness.json`, and `run_manifest.json`. Private model-chain outputs are `team_predictions.parquet`, `minutes_predictions.parquet`, `xpoints_predictions.parquet`, `gameweek_predictions.parquet`, `decision_candidates.parquet`, `optimized_squad.parquet`, `optimized_lineup.parquet`, `comparison.parquet`, and `lineage.json` under `model_chain/`. Public rows use `stable_player_id`; private and historical rows use `player_uid`. Fixture forecasts use official `fixture_id` and/or `stable_fixture_uid` as implemented.

The operational manifest records `run_id`, target and inferred season, gameweek, timestamps, launch state, input fingerprint, code revision and dirty state, model defaults and `model_lineage`, frontend schema, warnings, fallbacks, and completed stages. Lineage run IDs use `<run_id>_team_current`, `<run_id>_minutes_current`, `<run_id>_xpoints_current`, and `<run_id>_decision_current`. Historical reconstruction pins Vaastav revision `f2090d378ebd1b0c3d14884770dde95f38c50a0d`; official source mode is `official_current_season`.

Implemented pre-deadline flags include `pre_deadline_population`, `is_pre_deadline_history_active`, `cold_start_no_history`/`is_cold_start_no_history`, `transferred_player`, `position_change`, `fallback_flag`, status/news availability fields, and `lineage_note` (for example `returning_player`). Not every special-case flag is currently exported by the sanitized public contract; Section 17 records the required gap closures.

## 3. Prospective research questions

The live evaluation will answer:

1. How accurately does the system forecast player FPL points?
2. How accurately does it forecast appearances, starts, and minutes?
3. Are probability and interval forecasts calibrated?
4. Does opponent and team context improve forecasts relative to honest baselines?
5. Does the model rank useful FPL candidates correctly?
6. Does D2 improve expected and realized decisions relative to D1 and simpler benchmarks?
7. Which failures arise from model assumptions, new information, football variance, or software/data problems?
8. How does prospective 2026-27 performance compare with historical rolling and GW1 backtests?

No claim may be made where the required frozen prediction, comparator, or official outcome was not preserved.

## 4. Forecast freeze and authority

A forecast is prospectively eligible only if it was generated before the official FPL deadline. For each official forecast, preserve an immutable directory or content-addressed archive containing:

- season, gameweek, official deadline, forecast generation and publication timestamps;
- official-data retrieval timestamp, source availability cutoff, and source availability method;
- Git commit SHA, release/model version, dirty-state declaration, configuration-file SHA-256 hashes, source revision identifiers, and raw-source SHA-256 hashes;
- publication `run_id`, input fingerprint, all model-lineage names and run IDs;
- fixture-team predictions, player-fixture minutes-state predictions, player-fixture xPoints projections, and player-gameweek projections;
- every supported probability, percentile, distribution, and expected component output;
- recommended squad, Starting XI, ordered bench, captain, vice-captain, D1 seed/comparator, D2 result, and optimizer diagnostics;
- the seven sanitized public artifacts, publication audit, publication manifest, public URL, release identifier, and a SHA-256 inventory of every frozen artifact.

The freeze process must verify that generation and successful validation/publication precede the deadline, keys are unique, the manifest agrees with every artifact, and all recorded hashes reproduce. The immutable inventory itself must be copied to durable storage before the deadline. Public files are not a substitute for private component and probability artifacts.

Frozen forecasts are immutable after the deadline. Official outcomes must never be added to or merged into frozen prediction artifacts. Outcomes and scoring products belong in separate, versioned directories and must reference the frozen forecast by run ID and hashes.

Retrospective reruns require a different run ID and the explicit label `retrospective`; they cannot replace or count as the original prospective forecast. Failed and missing forecasts must be registered with failure time, stage, and reason. The system must never silently choose a later or better-looking forecast.

If multiple pre-deadline forecasts exist, the authoritative forecast is **the final successfully validated official forecast published before the deadline**. Earlier forecasts remain archived, may be compared only as declared secondary analyses, and cannot be substituted after outcomes are known. A forecast generated but not successfully validated and published is not authoritative.

## 5. Model-change policy

Model development between gameweeks is allowed only before the next applicable deadline. Each change requires a new code SHA and explicit model/configuration version (including hashes); earlier forecasts remain frozen. Report prospective evaluation both overall and stratified by model version.

The change log must classify each change as a bug fix, modelling improvement, data/source change, scoring-rule change, or operational change. It must disclose whether observed live results motivated the change and identify the affected assumptions. Bug-fixed retrospective reruns may diagnose impact but remain retrospective. No historical forecast may be regenerated and presented as prospective.

## 6. Predefined populations

Population membership is computed separately for each deadline using only the frozen deadline snapshot, except for the two explicitly outcome-conditioned populations. Players may belong to multiple special-case groups; group indicators are non-exclusive and results report each indicator independently, with intersections only if declared before inspection or clearly labelled exploratory.

| Population | Fixed definition and use |
| --- | --- |
| All selectable | Every officially selectable football player (`element_type` 1-4) at the forecast deadline. Assistant managers (`element_type == 5`, position `AM`) are not football-player rows. This is the primary minutes population and a required xPoints result. |
| Pre-deadline candidate | All-selectable players with frozen `p_appearance >= 0.25`. The threshold is fixed now and cannot change after results. This is the primary xPoints population. |
| Pre-deadline history-active | Repository definition `pre_deadline_history_active`: at GW1, `prior_season_minutes > 0` or `prior_season_appearances > 0`; after GW1, `prev5_minutes_sum > 0` or `season_to_date_appearances > 0`. All inputs must predate the deadline. |
| Actual participants | Players with official minutes greater than zero. Outcome-conditioned diagnostic only, never primary. |
| Actual starters | Players officially recorded as starters. Outcome-conditioned diagnostic only. If the official feed does not preserve reliable starts, report unavailable rather than infer them from minutes. |
| Model top-K | Top 15, 30, and 50 all-selectable players ranked by frozen `expected_points`, with ties resolved by ascending stable player UID. |
| Recommended squad | The 15 players in the authoritative frozen D2 squad. |
| Recommended Starting XI | The authoritative frozen Starting XI, with its frozen captain and vice-captain. |

Pre-declared non-exclusive diagnostic groups are: `cold_start_no_history` (new-player cold start), promoted-team fallback, `transferred_player`, returning player, `position_change`, current injury/availability flag, double-gameweek player, and blank-gameweek player. Use the frozen implemented fields where present. “Promoted-team fallback” requires the frozen team-model fallback marker, “returning player” requires the frozen lineage/availability classification, injury/availability uses frozen official `status` and `news`, and double/blank uses the frozen official fixture list. No group may be inferred after results merely because a forecast was poor.

## 7. Primary measurements

| Layer | Primary metric | Primary population and grain |
| --- | --- | --- |
| Team scoring | Poisson negative log-likelihood | Fixture-team goal side |
| Minutes appearance | Appearance Brier score | All selectable player-gameweeks, with appearance = official minutes > 0 |
| Minutes amount | Minutes MAE | All selectable player-gameweeks; sum fixture minutes in doubles |
| xPoints mean | Player-gameweek MAE | Pre-deadline candidate population |
| xPoints ranking | Spearman rank correlation | Pre-deadline candidate population, calculated per gameweek and summarized across gameweeks |
| xPoints distribution | Brier score for scoring at least 5 points | Pre-deadline candidate population using frozen `prob_points_ge_5` |
| Decisions | Realized Starting XI points including captaincy | One authoritative frozen decision per gameweek, after official autosubs and vice-captain fallback rules are applied |

All-selectable xPoints MAE, RMSE, bias, and ranking must also be reported, but they are not the only primary result: many obvious non-participants produce zero minutes and zero points, which can make aggregate accuracy look misleadingly strong. No unrelated measures will be combined into a composite score.

## 8. Secondary and diagnostic measurements

All metrics below are secondary unless explicitly described as outcome-conditioned diagnostics.

| Layer | Measurements |
| --- | --- |
| Team | Goal MAE, RMSE, and bias; clean-sheet Brier score; three-way outcome log loss; joint-scoreline negative log-likelihood where the frozen joint distribution exists; fixed-bin calibration. |
| Minutes | RMSE and bias; start, reached-60, and played-90 Brier scores; fixed-bin probability calibration; results by position and each special-case group. |
| xPoints | RMSE and bias; Spearman by position; `prob_points_eq_0` calibration; Brier/calibration for at least 2, 5, 8, and 10 points only where frozen distributions support the threshold; central 50% (`points_p25`-`points_p75`) and 80% (`points_p10`-`points_p90`) coverage and width; realized points of frozen top 15/30/50; selected-player calibration; results by position, predeclared price bands, and special-case group. |
| Decisions | Recommended squad points; Starting XI points; captain points; vice-captain fallback; autosub points; unreplaced-starter events; bench points; cost and bank; legality; D2 versus D1 and `D0_PRICE_VALUE_BASELINE`; captain agreement; squad/lineup/bench overlap; paired realized differences. |

Price bands must be fixed before the first scored report as official tenths: `<=45`, `46-55`, `56-70`, `71-90`, and `>=91`; these boundaries must not be tuned to results.

For xPoints components, report event occurrence/count, expected component points, and contribution to actual/predicted total separately where artifacts support them. Components are appearance/minutes points, goals, assists, clean sheets, goals-conceded deductions, saves, penalty saves/misses, yellow/red cards, own goals, bonus, and defensive contributions. The implemented private names include `expected_points_appearance`, `_goals`, `_assists`, `_clean_sheets`, `_goals_conceded`, `_saves`, `_penalties`, `_cards`, `_own_goals`, `_bonus`, and `_defensive_contribution`, with corresponding expected event fields. Do not claim an event-probability analysis when only an expectation was frozen.

One gameweek is one decision observation. The 15 selected players are not 15 independent squad decisions. “Realized Starting XI points” means official points after legal automatic substitutions, captain doubling, and vice-captain takeover when applicable; also report the nominal no-autosub XI as a secondary diagnostic.

## 9. External and internal benchmarks

Freeze before each deadline, with retrieval timestamp, source availability time, provenance, stable IDs, and hashes:

- `B0_ZERO`/`B0_ZERO_MINUTES` zero-points or zero-minutes baseline;
- recent-points baseline (`B3_RECENT_POINTS_P3` and/or `B3_RECENT_POINTS_P5`, declared at freeze);
- Phase 3 `B5_EB_POINTS_PER90`;
- `D1_MEAN_ONLY_MILP`, evaluated on the same frozen player projections;
- player-price ranking and `D0_PRICE_VALUE_BASELINE`;
- official selected-by percentage/ownership ranking;
- official FPL pre-deadline expected-points field, if available;
- market-derived team expected-goals or clean-sheet probabilities, if legally and practically obtainable.

Official FPL expected points must never be a predictive feature. Restricted external data must not be redistributed. A benchmark unavailable before a deadline is recorded as missing and cannot be reconstructed afterward as prospective. Any later reconstruction is labelled retrospective.

Opponent/team-context value is assessed with paired frozen predictions from the implemented T2/X2 system and an honest context-free comparator (normally `X1_INDEPENDENT_COMPONENT_RATES_M7` if actually generated and frozen, otherwise the predeclared Phase 3/recent-points baseline). Absence of a frozen comparator means the question is unanswered for that gameweek, not that a post-outcome comparator may be generated.

## 10. Official outcome ingestion and revision policy

Ingest outcomes only after all relevant fixtures are complete and official FPL scoring is stable. Preserve one row per `season, fixture_id, player_uid`, official `fixture_id`, official player ID, stable `player_uid`, official minutes, starts where available, FPL event components, and official total points. Record retrieval time, source availability time/method, raw snapshot path/hash, source and source version.

Aggregate separately to `season, gameweek, player_uid`; preserve every fixture row in double gameweeks and sum official minutes, components, and points only at the gameweek layer. Validate unique keys, fixture completeness, player/fixture identity coverage, double-gameweek non-duplication, and reconstruction of official scoring. The existing event-live audit distinguishes `exact_match`, `incomplete_fixture`, `unresolved_source_limitation`, and `genuine_reconstruction_error`; unresolved or genuine mismatches block finalized scoring.

Official revisions are append-only snapshots. Retain every raw and normalized snapshot, identify changed rows/fields and retrieval times, and declare one finalized outcome snapshot in the registry. Re-score against a later official revision only as a versioned correction with the prior report retained. Never mutate the frozen prediction.

## 11. Exclusions and missing-data rules

| Case | Rule |
| --- | --- |
| Assistant managers | Exclude `entity_type == assistant_manager`, `element_type == 5`, or `AM` from football-player populations. Report separately only if a future implemented forecast contract supports them. |
| Officially unselectable | Not in all-selectable if unselectable at the deadline; retain if transferred/unselectable only after the deadline. |
| Postponed fixture | Keep player in deadline population. Move fixture outcome to the gameweek in which official FPL ultimately scores it; report the original blank and later double according to official fixture/event assignment. |
| Abandoned fixture | Do not score until official FPL finalizes or voids it. Record pending status and eventual rule. |
| Blank gameweek | Zero fixtures and official points are valid forecast outcomes for selectable players; retain and label blank. |
| Double gameweek | Retain all player-fixture rows and aggregate once to player-gameweek. Never repeat event totals on each fixture. |
| Transfer after deadline | Retain under frozen identity/team context; label the later transfer, do not exclude. |
| Missing/revised result | Mark pending/missing; do not impute. Apply the snapshot/revision policy above. |
| Identity failure | Quarantine unresolved rows, report counts and coverage, issue an incident, and do not opportunistically drop them from headline denominators. Correct mappings are versioned. |
| Publication failure | Register failure. With no successful pre-deadline publication there is no authoritative prospective decision forecast. |
| Partial gameweek | No finalized gameweek headline score; fixture-level provisional diagnostics must be labelled partial and later retained or superseded visibly. |
| API/source failure | Record missing forecast, benchmark, or outcome as applicable; no after-the-fact recreation as prospective. |
| No historical data | Retain; classify `cold_start_no_history` and any applicable fallback. |

Injuries, rotation, suspensions, transfers, registration changes, and tactical changes are generally forecast outcomes or explanatory labels, not exclusions.

## 12. Non-exclusive error taxonomy

Permitted labels are: team attack-strength error; team defence-strength error; expected-minutes error; appearance error; start error; role or position error; attacking-share error; finishing variance; assist-allocation variance; clean-sheet variance; saves variance; bonus variance; disciplinary variance; injury announced after deadline; unexpected rotation; suspension; transfer or registration change; tactical-role change; official-source revision; source-data defect; identity-resolution defect; scoring defect; simulation defect; optimizer defect; and operational publication failure.

Labels are non-exclusive and never remove an observation from primary metrics. Every label requires contemporaneous or auditable evidence: frozen inputs/outputs, official team sheet or injury/disciplinary notice, source snapshot diff, deterministic scoring reconstruction, code-level reproduction, or optimizer diagnostic. A wrong prediction alone is not evidence for a label. Reports must identify label author, timestamp, evidence link/hash, and confidence or unresolved status.

## 13. Reporting checkpoints

- **GW1 diagnostic report:** descriptive only; no superiority claim.
- **GW4 or GW6 interim report:** choose and record the checkpoint before GW4 outcomes (default GW6 if no earlier registry decision); first limited multi-gameweek assessment.
- **Mid-season report:** after the official halfway checkpoint and finalized outcomes.
- **Final season report:** after the final official outcome snapshot is stable.
- **Incident report:** whenever a data, identity, scoring, simulation, optimizer, or publication defect affects a frozen forecast or its evaluation.

Early reports must avoid strong superiority, calibration, or generalization claims. Interim checkpoint choice cannot be changed in response to results without being disclosed as a protocol deviation.

## 14. Statistical reporting

Every report gives numbers of gameweeks, fixtures, fixture-team sides, players, player-fixtures, and player-gameweeks; population eligibility and coverage; missingness/exclusions; point estimates; and uncertainty intervals. Calibration uses fixed bins `[0,.1)`, `[.1,.2)`, ..., `[.9,1]`, merging bins only for display while retaining the fixed-bin table.

Use paired differences when forecasts/decisions apply to the same gameweeks. For multi-gameweek comparisons, use a gameweek-block bootstrap (resample whole gameweeks, at least 2,000 deterministic-seed replicates) where sample size permits; fixture blocks may supplement team-only analyses. Report the seed, estimator, percentile interval, and number of unique blocks. With too few blocks, report descriptive ranges and explicitly withhold inferential interpretation.

Player rows within a fixture or gameweek are not independent. Do not significance-hunt across metrics, populations, thresholds, or subgroups; report the registered family in full. Formal null-hypothesis tests are not required for every metric.

Historical comparisons must use the repository's frozen rolling and GW1 backtests and name the exact run/model/population. Historical rolling figures and three-fold GW1 decisions are context, not prospective observations. Retrospective experiments are visually and tabularly separated from prospective results.

## 15. Prospective registry

Keep this table append-only. A blank field means not yet available, never an invitation to invent it.

| Gameweek | Deadline (UTC) | Authoritative run ID | Code SHA | Model version | Publication timestamp (UTC) | Manifest SHA-256 | Public URL | Outcome snapshot timestamp (UTC) | Scoring status | Notes |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |

Failures and non-authoritative pre-deadline runs require a companion append-only register with run ID, timestamps, validation/publication status, artifact inventory hash, and reason.

## 16. Protocol deviations and audit trail

Freeze this document by recording its Git blob/commit and SHA-256 before outcomes. Any later edit requires a dated amendment describing the changed text, reason, whether outcomes had been observed, and affected analyses. The original remains available. Changes after outcome observation cannot redefine prospective primary populations, metrics, authority, or exclusions; they are deviations or retrospective analyses.

Each scoring report must join predictions to outcomes using frozen stable IDs, record the exact forecast and outcome hashes, produce unmatched-key tables, and include a machine-readable metric specification/version. Another analyst must be able to reproduce all denominators and results without discretionary choices from the project author.

## 17. Required pre-GW1 retention and implementation gaps

Before the GW1 deadline, complete these documentation/retention items without regenerating the forecast:

1. Copy the original private directory, audit, and manifest for `official_2026-27_30140778348_1` from CI/publication retention into immutable storage; compute a complete SHA-256 inventory and record the manifest digest and authoritative publication timestamp in the registry.
2. Preserve private T2 fixture-team predictions, M7 six-state probabilities (`p_reached_60`, `p_played_90` included), X2-M7 player-fixture component outputs and full distribution fields (`points_p25`, `points_p75`, `prob_points_eq_0`, `prob_points_ge_1`), D1 seed, D2 decision, and optimizer diagnostics. The seven sanitized public files alone do not support all registered metrics.
3. Preserve the all-selectable deadline universe and all population/special-case flags. The public projection contract has `cold_start_no_history`, `fallback_flag`, `status`, `news`, and `lineage_note`, but does not expose every implemented/private history, transfer, position-change, promoted-team, returning-player, double, and blank indicator.
4. Freeze each desired external benchmark before the deadline. No repository artifact currently proves prospective capture of ownership, official FPL expected points, or market probabilities.
5. Record config hashes for T2, M7, X2-M7, D1/D2 and the raw-source hashes/retrieval metadata in the authoritative manifest inventory. The public contract does not expose every config hash.
6. Choose GW4 or GW6 for the interim checkpoint before GW4 outcomes; absent a recorded choice, GW6 is authoritative.
7. Implement/test the separate outcome-scoring workflow before scoring, including M7 live metrics, revised-snapshot retention, exact-start availability handling, and prospective D1/D2 paired outputs. Existing `frozen_evaluation.py` evaluates 2025-26 with M3/X2-M3 and is historical infrastructure, not a scorer for the released M7/X2-M7 forecast.

If any item cannot be recovered before the deadline, mark its metric or subgroup unsupported/missing. Do not reconstruct the prediction-side data after outcomes.

## 18. Known limitations

- The current public release covers GW1, and there is no live accuracy record yet.
- Historical GW1 decision evidence contains only three folds.
- M7 was selected for operational coherence, not because it won every historical metric.
- D2 is bounded and heuristic, not globally optimal; its appearance states assume player independence.
- GW2+ clean-runner event-live reconstruction/publication is not implemented, and scheduling is disabled.
- External official, historical, ownership, and market sources may be unavailable or revised.
- The private official publication manifest and model-chain artifacts are ignored/unavailable in this checkout; only the report's verified identifiers, source hashes, and public contract are locally auditable here.
- Simulation draws support only the thresholds and intervals actually frozen. Unsupported 2+/8+ thresholds must be reported unavailable, not approximated from post-deadline reruns.
- Official event-live data may not provide a reliable exact-start field; start scoring remains unavailable until an authoritative field is preserved.
- This is an unofficial project unaffiliated with the Premier League or Fantasy Premier League.

## 19. Interpretation boundaries

Prospective primary tables answer the pre-registered performance questions. Secondary tables add resolution but do not replace a poor primary result. Actual-participant and actual-starter results are outcome-conditioned diagnostics and cannot be presented as the headline population. Retrospective reruns answer counterfactual development questions only. Error labels explain observations but neither excuse nor exclude them.

No report may describe a model as superior merely because one selected metric, subgroup, or gameweek is favorable. Claims must match the available paired gameweeks, uncertainty, version stratification, benchmark availability, and registered population.
