# Data Notice

Daniel Mehta's original source code in this repository is licensed under `AGPL-3.0-only`. That
source-code licence does not relicense third-party datasets, official Fantasy Premier League
content, Premier League material, player or team identities, trademarks, logos, fixtures, scores,
statistics, or third-party database rights.

Unofficial project. Not affiliated with, endorsed by, or associated with the Premier League or
Fantasy Premier League.

## Third-Party Data Boundaries

Raw and normalized third-party data are intentionally excluded from the `main` research bundle. Git
ignores retrieved official API payloads, Vaastav historical CSVs, normalized Parquet tables,
operational outputs, generated frontend data, reports, and logs. The tracked `paper/` assets are
aggregate research summaries, provenance metadata, a data dictionary, and the conceptual image
embedded in manuscript v7. They are not a release of the underlying player-fixture research panel.

The separate `official-forecast-data` branch and public dashboard contain sanitized operational
forecasts. Their existence does not establish permission to redistribute raw, normalized, or
row-level research inputs in a Sloan submission. The previously tracked player-level prospective
Table 8 and Figure 8 are excluded from the current `main` submission bundle pending rights review;
their earlier public Git history is a separate review item and is not erased by this working-tree
change.

The public repository does not grant downstream users rights to third-party data. Users are
responsible for complying with the terms, licences, acceptable-use policies, and database-rights
requirements of each source they choose to access.

## Sloan Research Data Boundary

The SSAC research-data requirement still needs a written interpretation from Sloan. This project
provides pinned source URLs and revisions, retrieval timestamps and hashes, a schema, aggregate
results, code/configuration, and conditional reproduction guidance in [`README_SLOAN.md`](README_SLOAN.md).
Those materials do not make the excluded row-level data publicly available. Whether links plus
retrieval scripts and hashes satisfy Sloan, and whether any derived row-level data can lawfully be
shared, remain external decisions. Work on the legally safe public bundle proceeds while those
answers are pending.

Commercial deployment may require separate permission, an appropriately licensed data provider, or
other legal review. This notice is not legal advice.

## Sources Used By The Project

The ingestion code supports official Fantasy Premier League endpoints including:

- `https://fantasy.premierleague.com/api/bootstrap-static/`
- `https://fantasy.premierleague.com/api/fixtures/`
- `https://fantasy.premierleague.com/api/event/{gameweek}/live/`

Historical CSV ingestion uses Vaastav's Fantasy Premier League historical dataset repository:

- https://github.com/vaastav/Fantasy-Premier-League

Vaastav's repository code is MIT licensed, but its licence states that the underlying data belongs
to Fantasy Premier League and Understat. This project uses Vaastav as a historical FPL data source
and attributes it accordingly. Understat-origin rights are mentioned here because Vaastav's licence
identifies Understat as an underlying data owner; the current project should only treat specific
fields as Understat-derived when the ingested files or source documentation support that.

## What Is Not Included

The tracked `main` research bundle does not include Premier League, Fantasy Premier League, or club
logos, crests, copied visual identity, raw official payloads, normalized historical data, or
row-level research predictions. The separate frozen operational publication branch is described
above.
