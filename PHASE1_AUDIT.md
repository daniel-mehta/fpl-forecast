# Phase 1 Audit

Audit run from the repository root on July 22, 2026.

| Audit question | Verdict | Evidence | Change made | Remaining limitation |
|---|---|---|---|---|
| Current API season identity | PASS | Latest correctly labelled current cache under `data/raw/fpl_api/2025-26/` has `requested_season = 2025-26`, `inferred_season = 2025-26`, 20 teams, 38 events, 380 fixtures, first kickoff `2025-08-15T19:00:00Z`, last kickoff `2026-05-24T15:00:00Z`. The older `2026-27` cache was shown to contain the same 2025-26 dates. | Added season parsing and inference from event deadlines plus fixture kickoffs. `snapshot-current` and `normalize-current` now fail on season mismatch or malformed dates and write requested/inferred season metadata. README workflow now uses `2025-26` for the audited cache. | The FPL API does not expose an explicit season string; inference depends on dated events/fixtures and standard Premier League structure. |
| `xP` exclusion | PASS | Raw Vaastav `merged_gw.csv` columns include `xP`. Normalized `historical_player_fixtures.parquet` columns contain no case-insensitive `xp` column. Validation reports a warning that raw `xP` exists. | Validation now rejects any normalized historical column whose lowercase name is `xp`. Added regression test for `XP` leakage. | Future feature tables must keep the same exclusion policy. |
| Git hygiene | PASS | `git check-ignore -v` reports `.gitignore:12:data/raw/**` for raw API/Vaastav files and metadata, and `.gitignore:13:data/normalized/**` for Parquet outputs. `outputs/synthetic_demo/README.md` is not ignored because it is intentionally commit-worthy. `git ls-files data outputs` only shows the four pre-existing synthetic demo files at audit time. | `.DS_Store`, raw data, normalized data, virtualenv, pytest cache, Ruff cache, and Python bytecode are ignored. Manual templates and synthetic warning README remain trackable. | Existing quarantined demo CSV/Markdown artifacts are tracked from before this work; they remain quarantined and are not used by code/tests. |
| Historical table grain | PASS | Real normalized table has 27,605 rows. Duplicate count for `season, gameweek, player_id` is 374; duplicate count for `season, gameweek, player_id, fixture_id` is 0. Example double-gameweek rows include Armando Broja and Abdoulaye Doucoure in GW24 with fixtures 144 and 235. Zero-minute rows exist: 16,039. No gameweeks 1..38 are absent in the full 2024-25 data. | Renamed output from `historical_player_gameweeks.parquet` to `historical_player_fixtures.parquet`; normalization removes the stale generated legacy file for that season. Added tests for double gameweek, blank gameweek absence, zero-minute rows, and unique player-fixture key. | The table includes explicit zero-minute rows where Vaastav provides them; it does not synthesize rows for blank gameweeks absent from the source. |
| Identifiers, timestamps, provenance | PASS | Current players preserve player ID, code, team ID, element type/position, price, status, minutes, total points, retrieval timestamp, season, source, source version, and raw snapshot lineage. Current fixtures preserve fixture ID/code, gameweek, home/away team IDs, kickoff, started/finished, scores, and lineage. Historical player-fixture records preserve season, player ID/code/name, fixture ID, gameweek, opponent team, home/away, kickoff, minutes, total points, value, position fields, source revision, retrieved timestamp, raw lineage, expected-goal fields, and FPL component outcomes present in Vaastav. | Added historical `kickoff_time`, `source_position`, `element_type`, `fpl_position`, and actual component outcome columns. Added timestamp validation. | Cross-season identity resolution remains Phase 2/manual work. Historical source does not provide every future modeling field in an authoritative cross-season mapping form. |
| Test quality | PASS | Test suite increased from 15 to 28 tests. Coverage includes HTTP error, timeout, invalid JSON, missing API fields, cache reuse, offline cache hit/miss, refresh behavior, checksum/metadata, historical missing columns, duplicate candidate keys, invalid minutes, season mismatch, malformed/missing dates, `xP` leakage, double-gameweek grain, CLI nonzero validation exit, timestamp parsing, and AM regression. | Added focused failure, boundary, regression, and grain tests without using quarantined demo artifacts. | Tests remain offline by design; live-source availability is verified by the explicit real-data workflow, not unit tests. |
| `AM` positions | PASS | Raw Vaastav `merged_gw.csv` column `position` contains 322 `AM` rows across 20 element IDs. Joined `players_raw` shows those rows have `element_type = 5` and are assistant managers such as Mikel Arteta, Pep Guardiola, and Arne Slot. These are not ordinary player positions. | Preserved `source_position`, mapped `players_raw.element_type` to `fpl_position`, and allowed `AM` only as FPL element type 5. Added regression test for assistant-manager mapping. | Later modeling must explicitly exclude or model element type 5; Phase 1 does not decide forecasting treatment for assistant managers. |

## Requested Versus Inferred API Season

The audited current payload retrieved on July 22, 2026 is inferred as `2025-26`, not `2026-27`.
Evidence:

```text
first_event_deadline=2025-08-15T17:30:00Z
last_event_deadline=2026-05-24T13:30:00Z
first_fixture_kickoff=2025-08-15T19:00:00Z
last_fixture_kickoff=2026-05-24T15:00:00Z
event_count=38
team_count=20
fixture_count=380
```

## Raw And Normalized xP Status

```text
Raw Vaastav merged_gw.csv: xP present
Normalized historical_player_fixtures.parquet: no xP/xp/XP column
Validation: warning only for raw xP presence
```

## Historical Grain Evidence

The normalized historical table is player-fixture grain.

```text
rows=27605
duplicates(season, gameweek, player_id)=374
duplicates(season, gameweek, player_id, fixture_id)=0
zero_minute_rows=16039
missing_gameweeks=[]
```

Double-gameweek examples:

```text
2024-25 GW24 player_id=156 Armando Broja fixture_id=144 minutes=0 points=0
2024-25 GW24 player_id=156 Armando Broja fixture_id=235 minutes=0 points=0
2024-25 GW24 player_id=217 Abdoulaye Doucoure fixture_id=144 minutes=90 points=-1
2024-25 GW24 player_id=217 Abdoulaye Doucoure fixture_id=235 minutes=90 points=8
```

## Test Classification

| Test area | Classification |
|---|---|
| FPL parsing and snapshot current | Happy path / integration with mocked HTTP |
| HTTP status, timeout, invalid JSON, missing fields | Failure cases |
| Cache reuse, refresh, offline with cache, offline without cache | Boundary and regression cases |
| Season parsing, matching, mismatch, malformed/missing dates | Boundary and failure cases |
| Snapshot checksum, metadata, immutable path | Regression cases |
| Current and historical normalization | Happy path |
| Historical missing columns, double gameweek, blank gameweek, AM mapping | Failure/boundary/regression cases |
| Validation duplicate key, invalid minutes, xP leakage, timestamp parsing, synthetic markers | Failure and regression cases |
| CLI validation on empty directory | Smoke / failure case |

## Recommendation

SAFE TO COMMIT PHASE 1
