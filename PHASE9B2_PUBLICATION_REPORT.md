# Phase 9B2A Manual Official Publication

## Scope

Phase 9B2A adds manual official forecast publication and separates fast, full, and frontend CI. It
does not schedule forecast publication, change forecasting models, alter optimizer behavior, add
multi-gameweek planning, or redesign the dashboard.

## Why Pages Showed A Waiting State

Operational forecasts and synchronized frontend data are deliberately ignored by Git. The previous
Pages workflow checked out a clean repository and built Vite without `npm run sync-data`, so it
correctly deployed the tracked waiting state rather than a local or mocked forecast.

## Workflow Architecture

| Workflow | Trigger | Purpose |
|---|---|---|
| `CI` | Push to `main`, pull request | Locked install, representative `not slow` Python tests, Ruff |
| `Full Python suite` | Manual, nightly at `06:17 UTC` | Complete Python suite and Ruff |
| `Frontend CI` | Push or PR affecting `frontend/**` or its workflow | `npm ci`, lint, build; never deploys |
| `Publish official FPL forecast` | Manual only | Reconstruct, forecast, gate, synchronize, build, deploy |

The publication workflow uses only `contents: read`, `pages: write`, and `id-token: write`. It has
no `contents: write`, no mock input, and no schedule. Deployment uses the `github-pages`
environment and cannot run unless the complete build job succeeds.

## Clean-Runner Reconstruction

`fpl prepare-publication-data` starts from empty raw and normalized directories and:

1. Downloads `2022-23` through `2025-26` from Vaastav revision
   `f2090d378ebd1b0c3d14884770dde95f38c50a0d`.
2. Normalizes player-fixture data and rejects normalized raw `xP`.
3. Builds identities, fixtures, facts, panels, and leakage-safe features.
4. Runs data-quality and leakage audits.
5. Downloads fresh official bootstrap and fixture payloads.
6. Infers and validates the requested season and official rule schema.
7. Resolves a deterministic pre-deadline target gameweek.

The command is idempotent and fails rather than fabricating missing inputs.

## Target Gameweek

An explicit target must exist, be unfinished, have a future deadline, and have official fixtures.
When omitted, the resolver chooses the earliest unfinished event with a future deadline. Every prior
event must be finished and data-checked, and every prior fixture must be finished and provisionally
final. Contradictory or incomplete metadata fails publication. Phase 9B2A then deliberately rejects
gameweeks after GW1 because clean-runner event-live reconstruction of completed current-season
results is not yet part of this narrow first-publication pass.

## Publication Gates

Before Pages configuration, the workflow requires official-current lineage, matching requested and
inferred seasons, the resolved gameweek, current source hashes and timestamps, current freshness,
unique projection keys, valid identity and opponent coverage, valid prices/minutes/probabilities,
finite conditional xPoints, nonnegative supported xPoints, a legal squad and lineup, valid
captaincy, allowed optimizer status, completed fixed-squad refinement, the expected frontend schema,
the public disclaimer, and no mock markers, placeholders, local paths, or token-shaped secrets.

Only seven allowlisted frontend artifacts are synchronized. The synchronized directory is validated
again before Vite is built.

## Failure Preservation

Operational publication remains atomic: failed model or gate runs do not change the
`latest_successful` pointer. At workflow level, the deploy job has an explicit successful-build
dependency. No waiting-state or partial artifact is deployed after failure.

## CI Split

Slow markers are limited to genuine full-chain, full-pool optimization, transition, and dashboard
integration tests. The fast suite retains ingestion, normalization, identities, leakage, scoring,
team/minutes/xPoints behavior, decision legality, autosubs, official-current failure guards,
publication failure preservation, and frontend-contract checks.

Frontend-only pushes do not deploy because their clean runners do not possess ignored
last-successful forecast artifacts.

## Manual Publication Review

After these changes are committed and pushed with CI passing:

1. Open **Actions > Publish official FPL forecast**.
2. Use season `2026-27`.
3. Leave gameweek empty for deterministic resolution, or supply the reviewed official event.
4. Leave run ID empty unless a review-specific stable ID is needed.
5. Set the official-publication confirmation to `yes`.
6. Review reconstruction, gate output, the 14-day audit artifact, and the generated Pages preview.
7. Verify season, gameweek, timestamp, source mode, squad, opponent labels, disclaimer, and absence
   of demo markers on the deployed site.

## Verification

Final focused publication tests passed `16/16`. Normal push CI selects `169` tests and completed
locally in `58.89s`; `13` genuinely expensive tests are reserved for the full workflow. The one
complete-suite run passed its then-current `181/181` tests in `509.50s`; the final added GW1-only
guard subsequently passed in both the focused and fast suites, bringing final collection to `182`.
Ruff, YAML parsing, locked dependency installation, frontend lint, and frontend build all passed.

The real clean-directory rehearsal used no existing normalized or operational inputs:

| Stage | Result |
|---|---|
| Historical ingest and normalization | `4.355s`; rows `26,505 / 29,725 / 27,605 / 29,747` |
| Identities, panel, validation, leakage | `23.596s` |
| Fresh official snapshot and normalization | `0.330s` |
| Official forecast chain | succeeded in approximately `117s` |
| Pre-sync publication validation | `59` gates passed |
| Post-sync publication validation | `61` gates passed |
| Vite production build | `34` modules; `227.22kB` JavaScript before gzip |

The official payload inferred `2026-27`, launch state `READY_TO_REFRESH`, and GW1 deadline
`2026-08-21T17:30:00Z`. The rehearsed optimizer remained `heuristic_feasible`, terminated its
bounded squad search at the configured iteration limit, completed fixed-squad lineup refinement as
`single_change_local_optimum`, and reported expected realized points of `50.7602`.

Clean reconstructed data validation completed with zero errors and the expected warning that raw
Vaastav contains `xP`; normalized data excluded it. Leakage audit passed.

Official action tags referenced by the workflows were verified against their Git repositories:
checkout `v7`, setup-node `v6`, configure-pages `v6`, upload-pages-artifact `v5`, deploy-pages `v5`,
and upload-artifact `v6`. The setup-uv action remains pinned to the verified `v8.1.0` commit.

## Changed Files

- `.github/workflows/ci.yml`
- `.github/workflows/full-python.yml`
- `.github/workflows/frontend-ci.yml`
- `.github/workflows/deploy-pages.yml`
- `src/fpl_forecast/operations/publication_pipeline.py`
- `src/fpl_forecast/operations/orchestrator.py`
- `src/fpl_forecast/cli.py`
- `tests/test_phase9b2_publication.py`
- Slow-marker annotations in the Phase 7, Phase 8, and Phase 9B official-current tests
- `pyproject.toml`
- `README.md`, `frontend/README.md`, and `docs/deployment/github-pages.md`

Generated rehearsal data, operational outputs, audit files, synchronized frontend artifacts,
dependencies, and Vite output remain ignored.

## Remaining Limitations

- Forecast publication is manual; no forecast schedule exists.
- Appearance outcomes remain conditionally independent in D2 evaluation.
- D2 squad search remains bounded and heuristic.
- The gameweek transition gate is intentionally strict and will fail rather than publish through
  incomplete prior results.
- This workflow can publish the first official GW1 forecast only. A later manual-publication phase
  must reconstruct and validate completed current-season event-live inputs before GW2+ publication.
- External source availability can block reconstruction.

## Git State And Readiness

The task began from clean `main` at `ff92e43`, aligned with `origin/main`. This pass leaves only
tracked source, workflow, test, and documentation changes in the working tree. Raw and normalized
rehearsal data, operational runs, publication audits, synchronized frontend artifacts,
`node_modules`, and `frontend/dist` remain ignored.

The implementation is safe to commit and push for CI review. After fast Python CI, frontend CI, and
the manually triggered full Python workflow pass on GitHub, it is safe to trigger the first manual
GW1 publication with explicit official-publication confirmation. No publication should be triggered
for GW2 or later in Phase 9B2A.
