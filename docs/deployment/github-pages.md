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

GW2 and later use the same workflow after reconstructing all completed prior current-season events
from official event-live snapshots. Enter `target_gameweek` as a plain integer such as `2`, never a
presentation label such as `Week 2`. Leaving it empty retains deterministic earliest-future-event
resolution.

The project site should use:

```text
https://<owner>.github.io/fpl-forecast/
```

For this repository's current remote, the expected URL is
`https://daniel-mehta.github.io/fpl-forecast/`.

The public page previously showed the waiting state because generated forecast files are ignored by
Git. A Pages clean runner therefore had only the tracked frontend shell. The manual publication
workflow now reconstructs all four pinned historical seasons, downloads fresh official
`bootstrap-static/` and `fixtures/` inputs, resolves the pre-deadline gameweek, downloads each
required prior `event/{gameweek}/live/` payload, reconstructs completed player and team results,
runs the official forecast chain, validates its public contract, and runs `npm run sync-data` before
building Vite.

The workflow has no mock-launch input and no forecast schedule. The deployment job depends on the
entire reconstruction, forecasting, validation, synchronization, lint, and build job succeeding.
Any earlier failure leaves the previous successful Pages deployment untouched. A non-public audit
artifact is retained for 14 days even when a later build step fails. It includes preparation or
failure records, official raw snapshots and metadata, the current-season reconstruction manifest,
validation results, model lineage, and the sanitized inventory. None enters the public bundle.

## Frontend-only deployments

`Frontend CI` lints and builds frontend changes. `Deploy frontend to GitHub Pages` deploys UI-only
changes after retrieving `official-forecast-data`, a dedicated branch containing only the frozen,
sanitized public forecast files and their checksum manifest. It runs automatically for `main` pushes
that change `frontend/**` or its own workflow, and can be run manually by selecting **Deploy frontend
to GitHub Pages** in Actions. It does not run Python, generate a forecast, call the optimizer, or
fall back to empty or sample data. It fails closed if the bundle is unavailable, malformed, has a
different identity, or fails checksum validation.

The first use requires one manual migration only: select **Seed frozen official forecast bundle** in
Actions. It downloads the already-public direct `/data/` files, validates their existing official
identity, writes their checksummed bundle to `official-forecast-data`, and does not deploy or
regenerate anything. It refuses to overwrite a bundle that differs. After that, successful `Publish
official FPL forecast` runs replace this branch only after the official forecast and public-contract
validation complete, before building and deploying the matching dashboard.

Forecasting, simulator, optimizer, or public-forecast changes must use **Publish official FPL
forecast**; frontend-only deployment is intentionally not a publication mechanism.

The full Python test suite is separate from publication. It can be started manually and runs
nightly. Ordinary push and pull-request CI runs Ruff and `pytest -m "not slow"`.
