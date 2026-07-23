# Data Notice

Daniel Mehta's original source code in this repository is licensed under `AGPL-3.0-only`. That
source-code licence does not relicense third-party datasets, official Fantasy Premier League
content, Premier League material, player or team identities, trademarks, logos, fixtures, scores,
statistics, or third-party database rights.

Unofficial project. Not affiliated with, endorsed by, or associated with the Premier League or
Fantasy Premier League.

## Third-Party Data Boundaries

Raw and normalized third-party data are intentionally excluded from Git. The repository ignores
retrieved official API payloads, Vaastav historical CSVs, normalized Parquet tables, operational
outputs, generated frontend data, reports, and logs.

The public repository does not grant downstream users rights to third-party data. Users are
responsible for complying with the terms, licences, acceptable-use policies, and database-rights
requirements of each source they choose to access.

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

The tracked repository does not include Premier League, Fantasy Premier League, or club logos,
crests, copied visual identity, raw official payloads, normalized historical data, generated
forecasts, or generated operational artifacts.
