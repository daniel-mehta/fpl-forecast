# Manual official GitHub Pages publication

The intended GitHub Free setup requires this repository to be public. Making a repository public
does not make it open source; licence selection is a separate decision.

## Configure Pages

1. Commit and push the Pages preparation changes to the default branch.
2. Make the repository public when the public-release review is complete.
3. Open **Settings > Pages**.
4. Under **Build and deployment**, set **Source** to **GitHub Actions**.
5. Open **Actions > Publish official FPL forecast**.
6. Select **Run workflow** on the default branch.
7. Confirm the requested official season, optionally supply the target gameweek or run ID, and set
   `confirm_official_publication` to `yes`.

Phase 9B2A is intentionally limited to the first official GW1 publication. A later gameweek fails
closed until clean-runner current-season event-live reconstruction is implemented and reviewed.

The project site should use:

```text
https://<owner>.github.io/fpl-forecast/
```

For this repository's current remote, the expected URL is
`https://daniel-mehta.github.io/fpl-forecast/`.

The public page previously showed the waiting state because generated forecast files are ignored by
Git. A Pages clean runner therefore had only the tracked frontend shell. The manual publication
workflow now reconstructs all four pinned historical seasons, downloads fresh official inputs,
resolves the pre-deadline gameweek, runs the official forecast chain, validates its public contract,
and runs `npm run sync-data` before building Vite.

The workflow has no mock-launch input and no forecast schedule. The deployment job depends on the
entire reconstruction, forecasting, validation, synchronization, lint, and build job succeeding.
Any earlier failure leaves the previous successful Pages deployment untouched. A small audit
artifact is retained for 14 days; it contains hashes, validation results, model lineage, and a
sanitized inventory, not raw or normalized datasets.

Frontend-only pushes run lint and build but do not deploy. Such a runner cannot reconstruct the last
successful ignored forecast and could otherwise replace a live forecast with the waiting state.

The full Python test suite is separate from publication. It can be started manually and runs
nightly. Ordinary push and pull-request CI runs Ruff and `pytest -m "not slow"`.
