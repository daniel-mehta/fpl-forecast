# fpl-forecast

Real-data foundation for a Fantasy Premier League forecasting system.

This repository currently implements Phase 1, Phase 2, Phase 3, Phase 4, Phase 4.1, Phase 5,
Phase 6, and Phase 7. Phase 1 covers ingestion, immutable raw snapshots, normalization to Parquet,
validation, a command-line interface, tests, and documentation.
Phase 2 adds multi-season historical coverage, cross-season team and player identities, a canonical
player-fixture panel, deadline-safe feature primitives, feature lineage, and leakage auditing.
Phase 3 adds rolling-origin baseline backtests, dedicated GW1 cold-start validation, frozen
prediction outputs, metrics, and baseline comparisons.
Phase 4 adds leakage-safe team-fixture strength baselines, opponent-adjusted goal probabilities,
clean-sheet and match-outcome diagnostics, and official future-fixture inference with strict
season and team-identity checks.
Phase 4.1 adds an experimental Dixon-Coles low-score dependence challenger.
Phase 5 adds leakage-safe player expected-minutes and appearance-probability backtests, role-duration
state modeling, lineup-start diagnostics, and guarded current-season input validation.
Phase 6 adds deterministic FPL scoring reconstruction, component-based player xPoints backtests,
team-constrained goal allocation, simulated point distributions, and guarded current xPoints input
validation.
Phase 7 adds integer-tenths price handling, official squad/lineup rule validation, full-candidate
MILP weekly-reset squad optimization, captain/vice-captain/bench selection, autosub scoring,
exhaustive small-case no-chip multi-gameweek transfer planning, and guarded current decision input
validation.

It does not implement chips, scheduling, dashboard work, deployment, or production forecasting.

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

reports/
├── backtests/
├── team_backtests/
├── minutes_backtests/
├── xpoints_backtests/
└── decision_backtests/
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
uv run fpl ingest-historical --season 2024-25 --revision REVISION_SHA

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
Replace `REVISION_SHA` with a real Vaastav repository commit before running that example command.

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

## Phase 3 Baseline Backtesting Workflow

Phase 3 produces honest baseline comparisons over the audited Phase 2 panel. The predictions are
baseline outputs for historical validation only; they are not production forecasts.

```bash
uv run fpl backtest-baselines \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode rolling \
  --run-id phase3_rolling_hardened

uv run fpl backtest-baselines \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode gw1 \
  --run-id phase3_gw1_hardened

uv run fpl compare-baselines --run-id phase3_rolling_hardened
uv run fpl inspect-backtest --run-id phase3_rolling_hardened
```

Generated Phase 3 outputs are written under `reports/backtests/<run_id>/` and ignored by Git.
`PHASE3_REPORT.md` records the real-data baseline results and limitations.

## Phase 4 Team-Model Workflow

Phase 4 models fixture-level team goals and probabilities. It is not a player expected-points
model and must not be compared directly to Phase 3 player-point metrics.

```bash
uv run fpl backtest-team-model \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode rolling \
  --run-id phase4_team_rolling_poisson_v2

uv run fpl backtest-team-model \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode gw1 \
  --run-id phase4_team_gw1_poisson_v2

uv run fpl compare-team-models --run-id phase4_team_rolling_poisson_v2
uv run fpl inspect-team-model --run-id phase4_team_rolling_poisson_v2
```

Team-model configuration lives in `src/fpl_forecast/team_model/config.json`. Generated Phase 4
outputs are written under `reports/team_backtests/<run_id>/` and ignored by Git.

The model families are:

- `T0_LEAGUE_HOME_AWAY`: league-average home and away Poisson rates.
- `T1_SHRUNK_ROLLING_TEAM_RATE`: recent team scoring and conceding rates shrunk toward league rates.
- `T2_REGULARIZED_ATTACK_DEFENCE`: weighted ridge-penalized Poisson attack and
  defensive-weakness effects with `sum(attack) = 0` and `sum(defence) = 0`
  identifiability constraints. Higher attack means stronger scoring; higher defence means the team
  concedes more and is defensively weaker.

Historical backtests restrict result labels by `source_available_time < information_cutoff`, but the
historical fixture schedule is reconstructed from retrospective Vaastav files. This is
result-leakage-safe, but it may know final postponed or rearranged fixture assignments that were not
known at the original historical deadline. Prospective operation should use archived pre-deadline FPL
fixture snapshots.

Official current FPL price is preserved as integer `price_tenths` with snapshot provenance, but Phase
4 does not use price. Launch and pre-deadline price snapshots, integer-tenths budget arithmetic,
personal purchase and selling prices, and any price-as-feature audit belong in a later optimizer or
decision phase.

Future fixture inference:

```bash
uv run fpl forecast-team-fixtures \
  --season TARGET_SEASON \
  --gameweek 1 \
  --as-of AS_OF_TIMESTAMP \
  --run-id RUN_ID
```

This command fails clearly when the normalized current snapshot is season-mismatched, missing, fully
historical relative to `as_of`, or contains teams that cannot be mapped to stable identities. A new
promoted team can be used by adding a stable `dim_team` identity and alias; if it has no historical
fixtures, the model uses the neutral no-history fallback and records the promoted/unseen flag.

## Phase 4.1 Dixon-Coles Challenger

Phase 4.1 evaluates `T3_DIXON_COLES`, a Dixon-Coles low-score dependence challenger
for fixture score probabilities. It is tested as an experiment against the Phase 4
T2 independent-Poisson benchmark and does not replace expected-minutes or player
expected-points modeling.

T2 remains the default team probability model. The Phase 4.1 decision is
`RETAIN AS EXPERIMENTAL CHALLENGER`: T3 is valid and leakage-safe, but the
chronological scoring evidence is mixed and too small to promote for MVP use.

```bash
uv run fpl backtest-team-model \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode rolling \
  --run-id phase4_1_dixon_coles_rolling

uv run fpl backtest-team-model \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode gw1 \
  --run-id phase4_1_dixon_coles_gw1

uv run fpl compare-team-models --run-id phase4_1_dixon_coles_rolling
uv run fpl compare-team-models --run-id phase4_1_dixon_coles_gw1
```

## Phase 5 Expected-Minutes Workflow

Phase 5 models player-fixture minutes, appearance probability, start probability, 60-plus
probability, 90-minute probability, and exact-start role-duration states. It uses only Phase 2
player rows with `entity_type == "player"` and keeps assistant-manager rows preserved upstream but
deterministically excluded from standard player minutes panels.

```bash
uv run fpl backtest-minutes \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode rolling \
  --run-id phase5_minutes_rolling_real

uv run fpl backtest-minutes \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode gw1 \
  --run-id phase5_minutes_gw1_real

uv run fpl compare-minutes --run-id phase5_minutes_rolling_real
uv run fpl inspect-minutes --run-id phase5_minutes_rolling_real
```

Generated Phase 5 outputs are written under `reports/minutes_backtests/<run_id>/` and ignored by
Git. Frozen predictions exclude target minutes, actual starts, role-duration labels, target points,
lineup status, price, and other post-deadline fields.

The versioned candidate-population rule lives in
`src/fpl_forecast/minutes_model/config.json`. All observed player rows remain the primary
data-quality population. Decision-relevant evaluation is reported separately for
`pre_deadline_history_active`, and rows with no prior or in-season history are reported as
`cold_start_no_history`.

Current expected-minutes inference is deliberately guarded:

```bash
uv run fpl forecast-minutes \
  --season TARGET_SEASON \
  --gameweek 1 \
  --as-of AS_OF_TIMESTAMP
```

The command validates official future fixture and player snapshots first and fails clearly when the
current snapshot is missing, stale, season-mismatched, or not genuinely forecastable.

## Phase 6 Component xPoints Workflow

Phase 6 reconstructs FPL points from audited scoring components, estimates leakage-safe component
rates, integrates Phase 4 T2 team expectations and Phase 5 minutes forecasts, and writes both
player-fixture and player-gameweek xPoints distributions. It is still a historical validation layer,
not a production optimizer or live forecast service.

```bash
uv run fpl validate-scoring --seasons 2022-23,2023-24,2024-25

uv run fpl backtest-xpoints \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode rolling \
  --run-id phase6_xpoints_rolling_real

uv run fpl backtest-xpoints \
  --seasons 2022-23,2023-24,2024-25 \
  --test-seasons 2023-24,2024-25 \
  --mode gw1 \
  --run-id phase6_xpoints_gw1_real

uv run fpl compare-xpoints --run-id phase6_xpoints_rolling_real
uv run fpl inspect-xpoints --run-id phase6_xpoints_rolling_real
```

Generated Phase 6 outputs are written under `reports/xpoints_backtests/<run_id>/` and ignored by
Git. Frozen xPoints predictions exclude target points, target components, Vaastav `xP`, price,
ownership and transfer fields.

The selected Phase 6 default is `X2_TEAM_CONSTRAINED_SIM_M3`: it preserves T2 team-goal
conservation while remaining close to the best rolling MAE. `X1_INDEPENDENT_COMPONENT_RATES_M3`
remains the simple component baseline, and `X2_TEAM_CONSTRAINED_SIM_M5` remains a probabilistic and
GW1 challenger.

Current xPoints inference is guarded:

```bash
uv run fpl forecast-xpoints \
  --season TARGET_SEASON \
  --gameweek 1 \
  --as-of AS_OF_TIMESTAMP
```

The command fails clearly when current fixture/player snapshots are stale or season-mismatched, or
when prerequisite current team/minutes forecasts are unavailable.

## Phase 7 Decision Workflow

Phase 7 consumes frozen Phase 6 player-gameweek xPoints and historical Phase 2 prices to produce
legal historical weekly-reset squad and lineup benchmark decisions. This is not a realistic
season-management simulation, live service, or chip planner.

```bash
uv run fpl validate-decision-rules \
  --bootstrap-path data/raw/fpl_api/2025-26/bootstrap_static/20260722T164209744637Z.json

uv run fpl backtest-decisions \
  --seasons 2023-24,2024-25 \
  --mode rolling \
  --run-id phase7_decisions_rolling_real

uv run fpl backtest-decisions \
  --seasons 2023-24,2024-25 \
  --mode gw1 \
  --run-id phase7_decisions_gw1_real

uv run fpl compare-decisions --run-id phase7_decisions_rolling_real
uv run fpl inspect-decision-run --run-id phase7_decisions_rolling_real
uv run fpl plan-transfers --run-id phase7_transfer_demo
```

Generated Phase 7 outputs are written under `reports/decision_backtests/<run_id>/` and ignored by
Git. The historical runner uses SciPy HiGHS MILP over the full gameweek candidate set and reports
solver status, objective bound, gap, and runtime. Small exact cases are verified against brute-force
enumeration. The transfer planner is proven on small exhaustive no-chip multi-gameweek cases because
honest historical manager state, purchase prices, bank, free transfers, and transfer history are not
yet archived.

Current decision inference is guarded:

```bash
uv run fpl forecast-decisions \
  --season TARGET_SEASON \
  --gameweek 1 \
  --as-of AS_OF_TIMESTAMP
```

The command fails before writing outputs when current snapshots are stale, season-mismatched, or
missing prerequisite current xPoints artifacts.
