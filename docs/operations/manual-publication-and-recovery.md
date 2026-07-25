# Manual Publication And Recovery

This runbook covers the manually triggered official GW1 publication workflow. It does not bypass
validation gates, support scheduled forecast publication, or support clean-runner publication after
GW1.

## Publish An Official Forecast

1. Confirm `main` is clean, pushed, and passing the normal Python, frontend, and full Python
   workflows.
2. Open **Actions > Publish official FPL forecast**.
3. Select **Run workflow** on the reviewed `main` commit.
4. Set `season` to the official season, currently `2026-27`.
5. Leave `target_gameweek` empty for deterministic resolution, or enter the reviewed official
   gameweek. Phase 9B2A rejects values after GW1.
6. Leave `run_id` empty for the workflow-generated identifier unless a stable review identifier is
   needed.
7. Set `confirm_official_publication` to `yes`.
8. Start the workflow and review every reconstruction, forecast, validation, synchronization, and
   build step before reviewing the deployment job.

The workflow reconstructs the four pinned historical seasons, downloads fresh official FPL inputs,
resolves the target gameweek, runs the reviewed T2/M7/X2/D2 chain, applies the publication gates,
synchronizes seven allowlisted public artifacts, builds Vite, and deploys only after the build job
succeeds.

## Review The Publication

Download the private `publication-audit-<run-id>` Actions artifact before its 14-day retention
expires. Confirm:

- the workflow commit SHA is the reviewed commit;
- requested and inferred seasons match;
- target gameweek and deadline match official metadata;
- source mode is `official_current_season`;
- bootstrap and fixture retrieval times and SHA-256 hashes are recorded;
- the pinned historical source revision is recorded;
- all publication gates passed;
- model lineage identifies T2, M7, X2, and D2;
- D2 is labelled heuristic and fixed-squad lineup refinement completed;
- the sanitized inventory contains only the seven expected frontend files.

Then open `https://daniel-mehta.github.io/fpl-forecast/` in a fresh browser context. Check season,
gameweek, UTC timestamps, projection count, squad, opponents, prices, optimizer limitations,
disclaimer, and the absence of mock, demo, waiting, placeholder, or local-path content.

## Recovery Principles

- Do not bypass, edit around, or disable a failing gate.
- Do not deploy a waiting page or partial output after a forecast failure.
- Do not copy local ignored forecast artifacts into Git.
- Investigate the failed build job while the previous successful Pages deployment remains live.
- Rerun manually only after the underlying input, mapping, configuration, or source problem is
  understood and reviewed.
- If a publication succeeds but looks suspicious, do not immediately publish another guess. Save
  the audit artifact, compare public hashes and lineage, inspect the forecast locally, and leave the
  last known good deployment in place until the issue is resolved.

## Failure Procedures

### Official API Retrieval Fails

Leave the deployment untouched. Confirm the official endpoints are reachable and return valid JSON,
then rerun the same reviewed commit. Do not substitute cached, mock, or hand-labelled payloads.

### Season Identity Does Not Match

Stop. Verify official event dates and the requested season. Never relabel an old payload as the
requested season. Resume only after the official payload genuinely identifies the requested season.

### Team Or Player Identity Validation Fails

Inspect the generated identity review locally, update tracked manual identity mappings only after
evidence-based review, run the relevant identity and publication tests, and publish from a newly
reviewed commit. Do not silently fuzzy-match unresolved identities.

### Historical Reconstruction Fails

Check the pinned Vaastav revision and source availability. Confirm all four required seasons
reconstruct from an empty data directory. Do not use an existing ignored normalized directory as a
fallback.

### Scoring Or Leakage Validation Fails

Treat the failure as a release blocker. Determine whether source fields, timestamps, scoring rules,
or feature lineage changed. Fix and revalidate in tracked code; never suppress the warning or allow
retrospective fields through publication.

### Forecast Generation Fails

Retain the previous deployment. Reproduce the same run locally from clean inputs, inspect the model
lineage and failing stage, and correct the general implementation or input problem. Do not add
player-specific exceptions.

### Frontend Sanitization Fails

Inspect only the staged public inventory. Remove the private path, secret-like value, raw metadata,
or schema mismatch at its source. Do not weaken the local-path, secret, mock-marker, or allowlist
gates.

### Frontend Build Fails

Run `npm ci`, `npm test`, `npm run lint`, and `npm run build` from `frontend/`. Fix the tracked
frontend source or lockfile, then publish from a reviewed commit. A frontend-only push does not
deploy because it cannot reconstruct the last successful ignored forecast.

### Pages Deployment Fails

If the validated build and Pages artifact succeeded but deployment failed, inspect the deployment
job and Pages environment. Rerun the same workflow commit only after confirming that the validated
artifact is intact. Do not rebuild from unreviewed local files.

### The Public Page Unexpectedly Shows Waiting

Confirm the latest workflow used **Publish official FPL forecast**, not a frontend-only build.
Inspect the deployed Pages run, the seven synchronized files, and the public asset responses. Do not
deploy the tracked waiting shell as a recovery action.

### A Successful Publication Looks Suspicious

Save its audit artifact, record the workflow run and public hashes, and compare projections and the
decision artifact with a clean local reconstruction. If the forecast is not credible, leave the
site unchanged until a reviewed correction is ready; do not bypass gates or manually edit public
CSV files.

## GW2 And Later

Phase 9B2A supports the first official GW1 publication only. Clean-runner reconstruction of official
current-season `event-live` outcomes is not implemented or verified. The workflow therefore fails
closed for GW2 and later. Scheduling remains disabled. Implement and verify event-live
reconstruction before claiming or enabling later-gameweek publication.
