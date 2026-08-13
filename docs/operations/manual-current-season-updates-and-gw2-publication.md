# Manual current-season updates and GW2+ forecast publication

## Outcome

The manual official-publication path can prepare GW2 and later forecasts from a clean runner. This
is implementation and synthetic rehearsal evidence, not proof that a real GW2 publication has
succeeded. The public site remains static and scheduling is not implemented.

## Pre-change GW1 path

The existing workflow pinned four Vaastav seasons to revision
`f2090d378ebd1b0c3d14884770dde95f38c50a0d`, normalized them at player-fixture grain, rebuilt Phase 2
identities, facts, and features, and ran data-quality and leakage audits. It then archived fresh
official `bootstrap-static/` and `fixtures/` responses, normalized the current player, team, fixture,
and event tables, resolved an unfinished future target, and ran the operational T2, M7, X2, D1, and
D2 chain. The preparation function rejected every resolved target above GW1 before current-season
event-live history could be assembled.

Successful outputs were written to a temporary run directory, validated, atomically renamed, and
only then made `latest_successful`. Failures moved the temporary directory to the failed-run area
without changing that pointer. Authoritative runs already required a clean source tree, hashed the
commit, Git status, tracked diff, and all tracked or non-ignored source contents, rejected reused run
IDs, sanitized seven allowlisted frontend files, and froze the public bundle.

## Current-season reconstruction

For a target gameweek `T`, preparation now:

1. archives fresh `https://fantasy.premierleague.com/api/bootstrap-static/` and
   `https://fantasy.premierleague.com/api/fixtures/` payloads;
2. resolves and validates `T` from official events and fixtures;
3. archives `https://fantasy.premierleague.com/api/event/{gameweek}/live/` for every official event
   strictly earlier than `T`;
4. validates event finality and data checks, fixture finality and provisional status, event
   assignment, teams, players, duplicate keys, and awarded-point reconciliation;
5. normalizes current results at `(season, player_id, fixture_id)` without collapsing double
   gameweeks;
6. builds stable-team fixture results and player results using the operational identity bridge;
7. writes ignored Parquet histories and `current_season_reconstruction.json`; and
8. passes both histories to the existing team, minutes, xPoints, D1, and D2 chain.

Each raw snapshot has a timestamped immutable filename and metadata sidecar containing its endpoint,
retrieval time, byte size, SHA-256, source, target season, and source version. Fresh event-live
metadata also records the requested and resolved target, run ID, Git commit, clean-source status,
and source mode. The reconstruction manifest records the same run context for every source,
included event, normalized row count, blank event, and raw path. Generated official payloads and
normalized tables remain ignored and are uploaded only as non-public workflow audit material.

## Time and leakage contract

The target deadline is the `information_cutoff`. An included current-season row receives the latest
retrieval timestamp among the bootstrap, fixtures, and its event-live payload as
`source_available_time`. The required relation is strict:

```text
source_available_time < information_cutoff
```

Availability is never backdated to kickoff or final whistle. Only gameweeks below the target can be
loaded. Current minutes histories are rebuilt from historical plus earlier current-season rows whose
source availability precedes each event deadline. The target and later event-live endpoints are not
requested, and raw official `xP` remains outside the feature contract.

## Gameweek and identity behavior

- Normal prior events add finalized player and team results.
- Double-gameweek explain blocks remain separate fixture rows; event totals must equal the sum of
  their fixture blocks.
- A team or player without a target fixture receives an explicit zero-fixture projection.
- A completed prior globally blank event is recorded with zero result rows.
- A globally blank target fails with a precise unsupported status because the current publication
  optimizer cannot publish an all-zero event.
- A prior fixture still assigned to an earlier event blocks publication until it is finished and
  provisional. Official reassignment to another event is respected on the next fresh preparation.
- Player codes retain the existing cross-season identity contract. Unknown, duplicate, or
  transferred in-season identities that cannot be reconciled to a fixture fail closed.
- New players retain the existing cold-start fallback until historical or accepted current-season
  history exists. Promoted teams retain the existing neutral newly-observed-team fallback.
- Assistant managers remain archived in raw event-live evidence but are excluded from player-model
  histories and candidate pools.

## Publication and failure behavior

Current snapshot hashes, event-live hashes, reconstructed event and row counts, cutoff policy,
clean source state, target identity, model lineage, optimizer legality, frontend schema, freshness,
and sanitization are publication gates. The frozen data branch is updated only after
synchronization, revalidation, lint, and the production frontend build succeed. An always-run
private audit upload retains reconstruction and failure evidence. No failed preparation, model,
optimizer, validation, or build step deploys Pages or replaces the latest successful operational
pointer.

## Remaining limitation

There is no cron or scheduled publication. The first real GW2 publication should be run manually
from a reviewed clean commit, with the non-public audit artifact inspected before accepting the
deployment. Automatic scheduling should remain disabled until that manual operation succeeds.
