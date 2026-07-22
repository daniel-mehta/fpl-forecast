# fpl-forecast

Real-data foundation for a Fantasy Premier League forecasting system.

This repository currently implements Phase 1 and Phase 2. Phase 1 covers ingestion, immutable raw
snapshots, normalization to Parquet, validation, a command-line interface, tests, and documentation.
Phase 2 adds multi-season historical coverage, cross-season team and player identities, a canonical
player-fixture panel, deadline-safe feature primitives, feature lineage, and leakage auditing.

It does not implement forecasting models, expected-minutes models, team-strength models,
simulations, backtests, expected-points projections, squad optimization, transfer planning, or
predictive performance reporting.

## Data Boundaries

Generated real-data artifacts are ignored by Git:

```text
data/
├── raw/
│   ├── fpl_api/
│   └── vaastav/
├── normalized/
└── manual/

outputs/
└── synthetic_demo/
```

Only externally retrieved source bytes belong in `data/raw`. Only tables derived from those real
raw inputs belong in `data/normalized`.

`outputs/synthetic_demo/` is a quarantine for four pre-existing synthetic demo artifacts. Those
files are not real FPL data, are not evidence of predictive accuracy, and must not be consumed by
application code or tests.

## Setup

Target environment:

- Apple Silicon macOS
- Python 3.12 or newer
- `uv`
- CPU only

Install dependencies:

```bash
uv sync
```

## Canonical Phase 1 Workflow

Use the season label that matches the live FPL API payload. The command validates that label by
parsing event deadlines and fixture kickoff times before writing a current snapshot. In the cached
audit run on July 22, 2026, the API payload was inferred as `2025-26`, with deadlines and fixtures
from August 2025 through May 2026.

```bash
uv run fpl snapshot-current --season 2025-26
uv run fpl normalize-current --season 2025-26
uv run fpl ingest-historical --season 2024-25 --refresh
uv run fpl normalize-historical --season 2024-25
uv run fpl validate-data
```

Validation exits with a nonzero status when serious data-quality errors are present. Warnings are
reported separately. Vaastav historical `merged_gw.csv` often includes `xP`; the pipeline warns
when it is present and excludes it from normalized historical data.

## Commands

```bash
uv run fpl snapshot-current --season 2025-26
uv run fpl snapshot-current --season 2025-26 --refresh
uv run fpl snapshot-current --season 2025-26 --offline

uv run fpl ingest-historical --season 2024-25
uv run fpl ingest-historical --season 2024-25 --refresh
uv run fpl ingest-historical --season 2024-25 --revision <git-sha>

uv run fpl normalize-current --season 2025-26
uv run fpl normalize-historical --season 2024-25
uv run fpl validate-data
```

`snapshot-current` archives these FPL API endpoints:

- `https://fantasy.premierleague.com/api/bootstrap-static/`
- `https://fantasy.premierleague.com/api/fixtures/`

For current snapshots, the requested `YYYY-YY` label is checked against inferred season identity:

- event `deadline_time` values
- fixture `kickoff_time` values
- standard Premier League structure of 20 teams, 38 events, and 380 fixtures

If the inferred payload season conflicts with the requested label, the command fails rather than
archiving prior-season data under a current-season name. The same check runs before current
normalization, so a stale mislabeled cache cannot silently produce normalized tables.

The client also has reusable support for:

- `element-summary/{player_id}/`
- `event/{gameweek}/live/`

`ingest-historical` archives these Vaastav CSVs for the requested season:

- `data/<season>/gws/merged_gw.csv`
- `data/<season>/players_raw.csv`

When possible, the Vaastav source is fetched by exact Git revision rather than a moving branch URL.

## Normalized Tables

Current FPL API normalization writes:

- `data/normalized/<season>/current_players.parquet`
- `data/normalized/<season>/current_teams.parquet`
- `data/normalized/<season>/current_fixtures.parquet`

Historical Vaastav normalization writes:

- `data/normalized/<season>/historical_player_fixtures.parquet`

The historical table is player-fixture grain, not player-gameweek grain. Double gameweeks can
therefore contain more than one row for the same `season, gameweek, player_id`; the intended key is
`season, gameweek, player_id, fixture_id`.

Each table includes provenance columns:

- `source`
- `source_version`
- `retrieved_at`
- `season`
- `raw_snapshot_path`

FPL player IDs and player codes are preserved where source data provides them.

Historical position handling preserves both the raw source label and the FPL element category:

- `source_position`
- `element_type`
- `fpl_position`

Vaastav `2024-25` includes assistant-manager rows with `source_position = AM` and
`players_raw.element_type = 5`; these remain `fpl_position = AM` so later modeling can explicitly
exclude or handle them.

## Validation

`uv run fpl validate-data` checks:

- required columns
- primary-key uniqueness
- null required identifiers
- valid position values
- nonnegative prices and minutes
- plausible player-fixture minutes
- current fixture team references
- valid gameweek values
- duplicate historical rows
- optional expected-goal field availability
- suspicious raw historical `xP`
- provenance fields
- real-data source labels
- synthetic-demo contamination markers

## Identity Mapping

Phase 2 implements cross-season player identity mapping using unique FPL player codes. In the
audited `2022-23`, `2023-24`, and `2024-25` data, every player-season record mapped automatically
through a unique, non-null FPL code.

Team identities are season-aware and use tracked aliases in `data/manual/team_aliases.csv`.
Manual player overrides are supported in `data/manual/player_identity_overrides.csv`, and unresolved
or ambiguous review candidates are written to ignored outputs under `data/review/`.

Future seasons may still require manual review or overrides. Current-season positions, prices, and
team assignments remain season-specific.

## Verification

Run the offline verification suite:

```bash
uv run pytest -q
uv run ruff check .
```

## Phase 2 Panel Workflow

Phase 2 builds identity-aware, leakage-audited player-fixture panel tables. It still does not train
models or produce projections.

```bash
uv run fpl ingest-historical --season 2022-23
uv run fpl ingest-historical --season 2023-24
uv run fpl ingest-historical --season 2024-25

uv run fpl normalize-historical --season 2022-23
uv run fpl normalize-historical --season 2023-24
uv run fpl normalize-historical --season 2024-25

uv run fpl build-identities --seasons 2022-23,2023-24,2024-25
uv run fpl build-panel --seasons 2022-23,2023-24,2024-25
uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25
uv run fpl inspect-panel --seasons 2022-23,2023-24,2024-25
```

Generated Phase 2 tables are written under `data/normalized/phase2/` and ignored by Git.
Generated identity review candidates are written under `data/review/` and ignored by Git. Manual
inputs under `data/manual/` are version-controlled templates.

Phase 2 outputs:

- `dim_team.parquet`
- `team_season_map.parquet`
- `dim_player.parquet`
- `player_season_map.parquet`
- `dim_fixture.parquet`
- `fact_player_fixture.parquet`
- `features_player_fixture.parquet`

Feature definitions live in `src/fpl_forecast/features/registry.json`, and the leakage auditor
checks that produced feature columns are registered, shifted, and backed by source availability
lineage strictly before each row's information cutoff. When exact historical match-completion times
are unavailable, Phase 2 uses the conservative rule `source_available_time = kickoff_time + 3h` for
match-derived results; source kickoff alone is not treated as sufficient evidence that final match
statistics were known.

The fact and feature tables include `entity_type`, distinguishing standard football players from
assistant managers. Standard player modeling panels can filter `entity_type == "player"`.
