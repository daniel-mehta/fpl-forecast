# GitHub Pages deployment

The intended GitHub Free setup requires this repository to be public. Making a repository public
does not make it open source; licence selection is a separate decision.

## Configure Pages

1. Commit and push the Pages preparation changes to the default branch.
2. Make the repository public when the public-release review is complete.
3. Open **Settings > Pages**.
4. Under **Build and deployment**, set **Source** to **GitHub Actions**.
5. Open **Actions > Deploy frontend to GitHub Pages**.
6. Select **Run workflow**, choose the default branch, and confirm the run.

The project site should use:

```text
https://<owner>.github.io/fpl-forecast/
```

For this repository's current remote, the expected URL is
`https://daniel-mehta.github.io/fpl-forecast/`.

The first deployment intentionally contains no generated forecast artifacts and therefore displays
the safe upcoming-season waiting state. The workflow never runs the Python pipeline or
`npm run sync-data`; mocked local recommendations must not be deployed as real forecasts.

If deployment fails, open the workflow run in **Actions** and inspect the first failed lint, build,
artifact, or deployment step. Fix that failure before rerunning the manual workflow.

Scheduled Python refreshes and genuine production-data publication are not implemented. Main README
restructuring, Phase-document presentation, and licence selection are deferred to Prompt 3.
