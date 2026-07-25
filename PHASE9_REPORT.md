# Phase 9 Report

## 1. Scope And Objectives

Phase 9 turned the existing forecasting pipeline into a reviewable public GW1 forecast while
preserving time lineage, fail-closed publication, and explicit modelling limitations. It covered
the static dashboard, official-current integration, forecast hardening, expected-realized decision
support, clean-runner manual publication, public UI clarity, and launch verification.

This report distinguishes implementation evidence from predictive evidence. The first public
forecast is operationally verified; no sustained live-season accuracy evidence exists yet.

## 2. Phase 9A: Frontend And Pages Foundation

Phase 9A added the Vite, React, and TypeScript dashboard, waiting/demo/official states, GitHub Pages
support, public-repository notices, AGPL licensing, data attribution, and CI. The tracked frontend
contains no generated official forecast data. This separation initially caused a clean Pages build
to show the waiting state, which was correct until clean-runner forecast reconstruction existed.

## 3. Phase 9B1: Official Current-Season Integration

Phase 9B1 connected official 2026-27 players, teams, positions, prices, eligibility, fixtures, and
deadlines to the operational chain. It preserved mock mode for offline tests but prevented mock
inputs from entering official publication. Official season inference comes from payload event
dates; an old payload cannot be relabelled as a new season.

## 4. Phase 9B1.1: 2025/26 And Scoring Compatibility

Phase 9B1.1 added complete 2025/26 historical ingestion, dynamic training-season resolution,
defensive-contribution compatibility, official eligibility preservation, opponent display, and a
frozen 2025/26 evaluation. It reduced target-season cold starts and corrected cross-season team
identity handling. Once used for model hardening, 2025/26 ceased to be an untouched holdout.

## 5. Phase 9B1.2: Minutes And Allocation Hardening

The first official projection exposed two structural problems: M3 converted expected minutes into
identical start and appearance probabilities, while cold-start teammates could disappear from team
attacking allocations.

Phase 9B1.2 introduced `M7_HIERARCHICAL_AVAILABILITY_STATE`, explicit appearance states,
team-fixture participation coherence, cold-start priors, exposure-aware attacking-rate shrinkage,
current-squad share allocation, promoted-team priors, and draw-level component reconciliation.
M7 was selected for operational GW1 coherence. It did not win every historical metric and must not
be described as a universally superior minutes model.

## 6. Phase 9B1.3: Expected-Realized Decisions

`D1_MEAN_ONLY_MILP` remains the exact full-candidate benchmark for its mean-only objective.
`D2_EXPECTED_REALIZED_POINTS` starts from D1 and uses deterministic bounded one-swap squad search.
It evaluates all 32,768 appearance states for a selected 15-player squad under an explicit
independent-appearance assumption, applies ordinary automatic substitutions and captain fallback,
and performs fixed-squad lineup refinement.

Conditional-on-appearance xPoints are generated directly and shrunk when supported by few
appearance draws. The final squad search is heuristic and is not globally optimal. The
fixed-squad lineup is refined to a documented single-change local optimum.

Three historical GW1 weekly-reset decisions provided acceptance evidence only: D2 improved the
realized score in two folds, tied one, and was worse in none, for a mean difference of `+1.67`
points in `phase9b13_lineup_refined_decisions_gw1`. This is not a transfer-aware season simulation
and is too small a sample for a performance claim.

## 7. Phase 9B2A: Manual Clean-Runner Publication

The manual **Publish official FPL forecast** workflow:

1. checks out a clean commit;
2. installs locked Python and frontend dependencies;
3. reconstructs 2022-23 through 2025-26 from pinned Vaastav revision
   `f2090d378ebd1b0c3d14884770dde95f38c50a0d`;
4. downloads fresh official FPL bootstrap and fixture data;
5. validates season, rules, identities, cutoffs, and GW1;
6. runs T2, M7, X2, and D2;
7. applies publication gates before and after sanitization;
8. builds Vite;
9. deploys only after the build job succeeds.

Publication is manual. The workflow has no mock input and no schedule. Frontend-only CI does not
deploy because its clean runner has no last-successful ignored forecast.

## 8. Phase 9C1: Public UI Clarity

Phase 9C1 retained the approved design while adding inclusive minimum/maximum price filters,
accessible information tooltips, explicit UTC timestamps, a consistent optimizer metric grid, and
separate Starting XI and ordered bench tables. It exposes formation, squad cost, bank, captain,
vice-captain, heuristic search status, termination status, and lineup-refinement status without
claiming global optimality.

## 9. Live Public Verification Evidence

Verified URL: `https://daniel-mehta.github.io/fpl-forecast/`

Verified Git commit: `1c83619b4396474298b8d950f60a2aa39435e5f3`

Verified publication:

- workflow run `30140778348`;
- run ID `official_2026-27_30140778348_1`;
- build job and deploy job both succeeded;
- official season `2026-27`, Gameweek 1;
- 557 projection rows;
- 15-player squad with 2 GKP, 5 DEF, 5 MID, and 3 FWD;
- squad cost `£100.0m`, bank `£0.0m`;
- legal 3-5-2 Starting XI;
- distinct captain and vice-captain in the Starting XI.

A fresh public-browser check at 1280px and 390px verified HTTPS loading, official content, UTC
timestamps, search, position and inclusive price filtering, combined filters, reset and matching
count, tooltip click/keyboard/Escape/outside-click behavior, mobile tooltip containment, internal
table scrolling, and no page-level horizontal overflow. Hover support was verified in the deployed
commit's `onMouseEnter` implementation. No site console errors or failed first-party artifact
loads were recorded. No mock, demo, waiting, localhost, local-path, placeholder, or ambiguous
`Local time` content appeared.

## 10. Official Provenance And Lineage

The successful private audit and the seven live public artifacts agree on season, gameweek, run ID,
projection count, squad, and model lineage. Every live public artifact matched its private audit
SHA-256 digest.

Official inputs:

- bootstrap retrieved `2026-07-25T02:38:58.972668Z`, SHA-256
  `cc52d36a8967f9df898738663ea324aaeeb1fffea828612c409d666fc0c2b8ae`;
- fixtures retrieved `2026-07-25T02:38:58.974287Z`, SHA-256
  `9e7484118381f8202830906ba993c176475d8ca1796571f5dd78cbfc2d73bd3e`;
- forecast generated `2026-07-25T02:44:22.547218Z`;
- source mode `official_current_season`;
- team model `T2_REGULARIZED_ATTACK_DEFENCE`;
- minutes model `M7_HIERARCHICAL_AVAILABILITY_STATE`;
- xPoints model `X2_TEAM_CONSTRAINED_SIM_M7`;
- decision optimizer `D2_EXPECTED_REALIZED_POINTS`.

The audit records complete player and team identity coverage and the pinned historical source
revision. Public artifacts expose sanitized lineage, not raw datasets or credentials.

## 11. Publication Safety Gates

The verified audit records all gates as passed, including official source mode, season/gameweek
agreement, freshness, source hashes, identity and opponent coverage, projection uniqueness,
prices/minutes/probabilities, finite stabilized conditional xPoints, nonnegative supported xPoints,
legal squad and lineup, captaincy, allowed heuristic status, completed lineup refinement, frontend
schema, disclaimer, allowlisted inventory, and rejection of mocks, placeholders, local paths, and
secret-like values.

## 12. Failure And Last-Known-Good Evidence

Local deterministic failure tests verified that mock publication candidates are rejected, a failed
publication does not update `latest_successful`, and the deploy job requires a successful validated
build. The test did not trigger a destructive hosted deployment. After local failure injection, a
no-cache public status request retained SHA-256
`707b21c3f08f24875921614c1e31dc9e534464058052f22dd370faaeebca44c0`,
matching the successful audit. The public site remained available and unchanged.

## 13. Repository And Data Hygiene

The tracked tree and Git object history contain no raw FPL snapshots, raw Vaastav data, normalized
Parquet/database files, operational runs, backtests, publication audits, synchronized official
frontend data, Vite output, dependencies, environment files, credentials, or private audit
artifacts.

Intentional tracked exceptions are:

- `.gitkeep` files under raw and normalized data directories;
- `frontend/public/data/README.md`;
- clearly labelled synthetic demo CSV/Markdown files under `outputs/synthetic_demo/`.

The only tracked absolute user path is a deliberate `/Users/example/...` sanitization test fixture.
Representative ignore checks cover raw, normalized, operational, audit, frontend-data, build,
dependency, and environment paths.

## 14. Current Limitations

- The first public forecast covers GW1 only.
- No sustained live-season accuracy evidence exists.
- Historical GW1 decision evidence contains only three weekly-reset decisions.
- M7 was selected for operational coherence, not because it won every historical metric.
- D2 squad search is bounded and heuristic, not globally optimal.
- D2 appearance states assume player independence.
- Clean-runner GW2+ event-live reconstruction is not implemented.
- Forecast scheduling remains disabled.
- External official and historical sources can block reconstruction.
- This is an unofficial project and is not associated with the Premier League or Fantasy Premier
  League.

## 15. Recovery And Manual Refresh

The operator runbook is
[`docs/operations/manual-publication-and-recovery.md`](docs/operations/manual-publication-and-recovery.md).
It documents the manual workflow, audit review, failure-specific recovery, suspicious-publication
review, last-known-good preservation, and the GW2+ block. Recovery never recommends bypassing a
gate.

## 16. Phase 9 Acceptance Decision

Phase 9 meets its operational acceptance criteria for a first experimental public GW1 forecast:
the live page is verified, provenance is cryptographically linked to the successful clean-runner
audit, publication fails closed, generated data remain outside Git, and recovery is documented.

Acceptance does not mean production readiness or proven predictive superiority. It accepts a
credible, reproducible, manually published research release with explicit constraints.

## 17. Recommended Next Work

1. Freeze every pre-deadline public prediction and evaluate it after official results are complete.
2. Implement and test clean-runner current-season event-live reconstruction before enabling GW2+.
3. Keep publication manual until at least one later-gameweek reconstruction is reviewed.
4. Begin Phase 10 only with genuine live outcomes and preserve failed assumptions in the
   post-mortem.

## Release Tag Recommendation

- Tag: `v0.1.0`
- Target: `1c83619b4396474298b8d950f60a2aa39435e5f3`
- Title: `First credible public GW1 forecast`

Draft release notes:

> Publishes the first official-data 2026-27 GW1 forecast from a clean GitHub Actions runner. The
> release reconstructs leakage-safe historical inputs, uses the T2 team model, M7 minutes model,
> X2 team-constrained xPoints simulation, and D2 expected-realized heuristic optimization, then
> validates and deploys allowlisted static artifacts manually. Known limits: GW1 only, limited
> historical GW1 evidence, independent appearance states, heuristic squad search, no scheduled or
> GW2+ publication, and no claim of proven predictive superiority.

The tag should be created only after this documentation pass is reviewed and the working tree is
clean. The recommended target is the exact commit used by the verified successful workflow and
deployed site; no tag or GitHub Release was created during Phase 9D.
