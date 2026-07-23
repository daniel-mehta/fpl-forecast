# Phase 8 Operational Pipeline And Functional Dashboard Report

Phase 8 makes the project locally operational without claiming that genuine 2026-27 data have
launched. It adds launch detection, operational status, process locking, atomic publication,
last-known-good behavior, frontend data contracts, a local Python dashboard, mocked target-season
transition tests, live-result normalization utilities, and a genuine current-model chain for the
mocked season-launch path.

Phase 8 does not deploy a public dashboard, install a scheduler, implement hosted refreshes, add
visual polish, or begin Phase 9.

## Operational Meaning

Operational means the repository can:

- detect whether an official target season has launched;
- wait safely when the official API still exposes an old season;
- publish complete frontend-ready artifacts atomically only after validation;
- keep the latest successful output after failures;
- refuse concurrent refreshes through a lock;
- expose a stable local dashboard data contract;
- prove target-season transition behavior using representative official-shaped payloads;
- run the Phase 4 T2, Phase 5 M3/M5, Phase 6 X2-M3/X2-M5, and Phase 7 exact MILP decision chain
  over mocked target-season fixtures, teams, players and prices plus eligible prior-season history.

Real 2026-27 operation remains unproven until genuine official 2026-27 payloads pass the launch,
rule, identity and model guards.

## Official Endpoint Coverage

Inspected cached official payloads:

- `bootstrap-static/`: players, teams, positions, prices, status, news, chance-of-playing,
  selected-by/transfer context, events, deadlines, game settings and rules.
- `fixtures/`: fixture IDs, event/gameweek, teams, kickoff time, started, finished,
  `finished_provisional`, scores and fixture stats.

Before the original Phase 8 pass, no cached `event/{gameweek}/live/` payload existed in the
repository. A real network fetch was attempted for
`https://fantasy.premierleague.com/api/event/38/live/` and archived:

```text
snapshot data/raw/fpl_api/2025-26/event_live_38/20260723T040041063444Z.json
retrieved_at 2026-07-23T04:00:41.063444Z
normalized_rows 841
fixtures 10
players 841
```

The endpoint response has two related structures:

- top-level `stats` is a dictionary of player event/gameweek totals;
- `explain` is a list of fixture blocks, each containing scoring components with `identifier`,
  `value`, awarded `points`, and `points_modification`.

The original audit treated partial raw `explain` values as if they were a complete raw-stat source
and did not use the official awarded component points. That confused raw statistic values with FPL
points, missed the 2025-26 `defensive_contribution` scoring component, and left many players
under-reconstructed. The previously observed mismatch distribution was:

```text
point_difference  rows
              -2     1
              -1     1
               0   539
               1   134
               2    93
               3    16
               4     8
               5    14
               6    24
               7     5
               8     3
               9     2
              12     1
```

This closure pass corrected the normalizer and audit. The canonical event-live output is
player-fixture grain keyed by `season, fixture_id, player_uid`. It stores fixture ID, player ID,
stable player UID, team/opponent context, home/away, kickoff time, retrieval/source availability,
entity type, FPL position, starts/minutes, raw component values where fixture-assignable, official
component points, reconstructed fixture total points, event total points for validation only, and
source provenance. Double gameweeks keep one row per represented fixture; event totals are never
copied onto each fixture row.

Corrected real GW38 reconciliation:

```text
raw_element_count=841
normalized_player_fixture_rows=841
represented_fixtures=10
duplicate_key_count=0
exact_match_count=841
exact_match_pct=1.0
unresolved_count=0
excluded_assistant_manager_count=0
corrected_difference_counts={0: 841}
safe_to_use_for_subsequent_forecasting=True
```

Concrete corrected examples:

```text
player_id player_name fpl_position fixture_ids official reconstructed key correction
18        Madueke     MID          373         11       11            goal/CS/bonus awarded points
50        Buendia     MID          376          6        6            defensive_contribution points included
64        Watkins     FWD          376         13       13            two FWD goals use awarded points
101       Kelleher    GKP          375          6        6            saves and bonus reconciled
119       Mbeumo      MID          371          9        9            goal/CS/bonus reconciled
```

## State Machine

States implemented:

- `WAITING_FOR_SEASON_LAUNCH`
- `READY_TO_REFRESH`
- `REFRESHING`
- `NEEDS_RULE_REVIEW`
- `NEEDS_TEAM_IDENTITY_REVIEW`
- `NEEDS_PLAYER_IDENTITY_REVIEW`
- `NO_FORECASTABLE_GAMEWEEK`
- `SUCCEEDED`
- `FAILED_USING_LAST_SUCCESS`

Operational status JSON includes target season, inferred official season, check timestamp, deadline
or completed-gameweek context where available, latest successful run ID, freshness fields, reason,
warning, dashboard safety and retry guidance.

Real cached check:

```text
state=WAITING_FOR_SEASON_LAUNCH
target_season=2026-27
inferred_official_season=2025-26
latest_official_deadline=2025-08-15T17:30:00Z
latest_completed_gameweek=38
latest_completed_fixture=380
reason=Official payload currently identifies as 2025-26, not 2026-27.
```

## Launch And Rule Drift

Launch detection reuses the existing season-identity convention: event deadlines, fixture kickoffs,
20 teams, 38 events and 380 fixtures. Old-season payloads produce `WAITING_FOR_SEASON_LAUNCH`.
Mocked target-season payloads transition to `READY_TO_REFRESH` without a code change.

Rule checking compares official game settings and position quotas against the verified Phase 7 rule
schema. Material changes enter `NEEDS_RULE_REVIEW` and record a concise rule diff.

## Current Results And Panel Policy

Current fixture outcomes become eligible only after a fixture is both `finished` and
`finished_provisional`. The normalized result stores raw retrieval time and uses
`source_available_time` from the official live snapshot retrieval. Feature use must still satisfy:

```text
source_available_time < information_cutoff
```

Repeated rebuilds are idempotent by `season, player_id, fixture_id`, and assistant managers are
excluded from standard player modeling panels.

The publication safety gate rejects completed-result inputs with duplicate player-fixture keys,
unresolved fixture IDs, invalid timestamps, source availability at or after the forecast cutoff,
unexplained eligible-player scoring mismatches, repeated event totals on multiple fixture rows,
incomplete fixtures treated as final, unresolved source limitations, or forbidden outcome columns in
pre-deadline feature inputs. Failure occurs before atomic publication, so the latest successful
pointer remains unchanged.

## Identity And Cold Starts

Phase 8 preserves the Phase 2 identity rules: stable player codes are preferred, season-specific
team and position are retained, and ambiguous or contradictory codes must go to review. The mocked
published artifacts include cold-start and fallback flags in the dashboard contract. Promoted or new
teams are expected to use existing aliases when known and neutral fallback when safe; ambiguous team
collisions enter review.

## Orchestrator

Command:

```bash
uv run fpl refresh-operational --season 2026-27
```

Implemented behavior:

1. create operational directories;
2. acquire refresh lock;
3. check launch;
4. return normal waiting state if target season is not launched;
5. fingerprint inputs, code and configuration;
6. no-op if unchanged and a latest success exists;
7. build mocked official-shaped target-season fixtures, players, teams and prices;
8. combine target-season inputs with eligible prior-season history;
9. run T2 current team forecasts;
10. run M3 and M5 current minutes forecasts;
11. run X2-M3 and X2-M5 current xPoints forecasts;
12. run the full-candidate MILP squad, lineup, captain, vice-captain and bench optimizer;
13. build frontend artifacts in a temporary run directory from those generated outputs;
14. validate schema contract;
15. atomically publish to `outputs/operational/runs/<run_id>/`;
16. update `outputs/operational/latest_successful.json` only after success;
17. record failed attempts separately;
18. preserve latest-successful pointer after injected failures;
19. release the lock.

Representative successful mocked run:

```text
run_id=phase8_mock_success
state=SUCCEEDED
reason=Operational refresh published atomically.
```

Unchanged input no-op is reported through status with `no_op=True`.

The mocked launch path no longer copies Phase 7 historical decision backtest predictions into
operational outputs. It writes generated model-chain outputs under each run's `model_chain/`
directory and publishes only frontend-ready CSV/JSON views derived from the same run.

## In-Season GW1-To-GW2 Boundary

Phase 8 now includes a mocked in-season transition command:

```bash
uv run fpl mock-gw1-to-gw2-operational-transition \
  --season 2026-27 \
  --run-id phase8_gw1_to_gw2_final
```

The transition proof:

1. freezes GW1 forecasts before kickoff;
2. creates completed GW1 player-fixture and team-fixture rows after final source availability;
3. validates those completed rows before publication;
4. appends them to the available history for a GW2 model-chain refresh;
5. keeps GW2 targets absent from the inputs;
6. publishes GW2 projections and an optimized squad;
7. verifies a repeated unchanged run is a no-op;
8. verifies an injected late-source completed-result failure preserves last-known-good.

Representative output:

```text
completed_rows_entered_gw2_history=294
gw2_projection_rows=294
gw2_optimized_squad_rows=15
repeated_unchanged_run_no_op=True
injected_validation_failure_state=FAILED_USING_LAST_SUCCESS
failure_preserved_latest=True
```

## Model Chain Adapter

The Phase 8 adapter builds a combined prior-season plus target-season input set required by the
current forecast chain:

- prior-season history comes from audited 2022-23, 2023-24 and 2024-25 normalized Phase 2 data;
- target-season fixtures are official-shaped GW1 fixtures with future kickoff and deadline fields;
- target-season player rows preserve stable returning player UIDs while using current target-season
  team, position, status, news and price fields;
- new players are marked `cold_start_no_history`;
- transferred players use their stable `player_uid` with target-season team context;
- position-change rows keep their stable `player_uid` while using the current-season FPL position;
- a promoted-team fixture uses neutral team fallback flags.

The production chain run in the operational refresh is:

```text
Phase 4: T2_REGULARIZED_ATTACK_DEFENCE
Phase 5: M3_EWMA_MINUTES and M5_REGULARIZED_STATE_SOFTMAX
Phase 6: X2_TEAM_CONSTRAINED_SIM_M3 and X2_TEAM_CONSTRAINED_SIM_M5
Phase 7: scipy_highs_milp full-candidate squad, lineup, captain, vice-captain and bench
```

Lineage is written to `model_lineage.json`, copied into `run_manifest.json`, and surfaced in
`data_freshness.json`. A representative lineage shape is:

```text
team_model_run_id=<run_id>_team_current
minutes_model_run_id=<run_id>_minutes_current
xpoints_model_run_id=<run_id>_xpoints_current
decision_run_id=<run_id>_decision_current
```

## Publication Contract

Published artifacts:

- `operational_status.json`
- `player_gameweek_projections.csv`
- `optimized_squad.csv`
- `optimized_lineup.csv`
- `model_comparison.csv`
- `data_freshness.json`
- `run_manifest.json`

The pointer `outputs/operational/latest_successful.json` is updated atomically after success. Failed
runs are written under `outputs/operational/failed/<run_id>/`.

The manifest records target and inferred season, run ID, timestamps, launch state, input
fingerprint, code revision, dirty state, model defaults, model lineage, frontend schema version,
warnings, fallbacks, completed stages, previous latest success and completion stage.

## Dashboard

Command:

```bash
uv run fpl dashboard
```

Verification uses:

```bash
uv run fpl dashboard --smoke
```

The dashboard is a local Python stdlib HTML server. It reads only published frontend artifacts and
validates `phase8_frontend_v1` before rendering. It displays the operational source and model-run
lineage from the latest successful publication, which confirms the UI is reading the newly generated
operational outputs rather than historical placeholder artifacts. Views include:

- operational overview;
- data freshness and warnings;
- player projections;
- model comparison;
- recommended squad;
- lineup, captain, vice-captain and bench;
- methodology and limitations.

No dashboard server or watch process is left running after smoke verification.

## Tests And Failure Injection

Added tests cover:

- old-season waiting state;
- mocked target-season transition to ready;
- material rule-change review state;
- event-live per-fixture normalization and null exact-start labels;
- fixture finalization policy;
- idempotent current-panel rebuild and assistant-manager exclusion;
- mocked operational success;
- unchanged-input no-op;
- injected failure preserving latest successful pointer;
- concurrent lock rejection;
- dashboard smoke rendering;
- real cached 2026-27 waiting-state check;
- target fixture/opponent changes moving team and player forecasts;
- price-only changes moving the optimized squad while leaving performance forecasts unchanged;
- new-player, transferred-player, position-change and promoted-team neutral fallback cases through
  the full model chain;
- real official-shaped event-live parsing, awarded points versus raw values, exact reconstruction,
  DNP/substitute/clean-sheet/goals-conceded/save/bonus/card scoring, double-gameweek fixture rows,
  duplicate components, assistant-manager exclusion, incomplete fixtures, late source availability,
  idempotent re-ingestion, revised snapshots, last-known-good preservation and GW1-to-GW2 transition.

## Limitations

- Genuine 2026-27 operation remains blocked until the official API truly identifies as 2026-27.
- A real `event/38/live/` payload was archived, normalized and joined to bootstrap/fixture context.
  Corrected fixture-level reconstruction reconciles 841/841 player events exactly for the cached
  GW38 payload.
- The mocked operational forecast now runs the real model and optimization chain, but it is still
  based on representative official-shaped target-season inputs rather than a live 2026-27 launch.
- The cached real GW38 payload is a single-fixture-per-player gameweek; double-gameweek handling is
  proven with official-shaped tests rather than a real cached double-gameweek event-live payload.
- Current team/minutes/xPoints/decision commands remain guarded for real stale or mismatched current
  data.
- Full transfer management is still not proven for live manager state because bank, purchase prices,
  free transfers and transfer history are not available.
- The dashboard is functional local HTML, not Phase 9 public hosting or final UI polish.

## Phase 9 Scope

Phase 9 should add public deployment, hosted scheduling, authentication or sharing decisions, final
visual polish, and production monitoring. It should not weaken the Phase 8 launch, locking,
publication or last-known-good contracts.
