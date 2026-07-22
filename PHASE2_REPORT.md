# Phase 2 Report

Phase 2 builds a real, identity-aware, leakage-audited player-fixture panel from Vaastav historical
data. It does not train a model, compute predictive metrics, create projections, or optimize squads.

## Scope Completed

- Ingested and normalized three consecutive historical seasons: `2022-23`, `2023-24`, `2024-25`.
- Built cross-season `dim_team`, `team_season_map`, `dim_player`, and `player_season_map`.
- Built canonical `dim_fixture`, `fact_player_fixture`, and `features_player_fixture`.
- Added explicit inferred information cutoffs and feature lineage source-availability timestamps.
- Added a feature registry and leakage auditor.
- Added panel inspection and Phase 2 CLI commands.

## Data Coverage

| Season | Source revision | Raw rows | Normalized rows | Unique players | Fixtures | GW range | Kickoff coverage | Raw xP | Normalized xP |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| 2022-23 | `f2090d378ebd1b0c3d14884770dde95f38c50a0d` | 26,505 | 26,505 | 778 | 380 | 1-38 | 26,505 | present | absent |
| 2023-24 | `f2090d378ebd1b0c3d14884770dde95f38c50a0d` | 29,725 | 29,725 | 865 | 380 | 1-38 | 29,725 | present | absent |
| 2024-25 | `f2090d378ebd1b0c3d14884770dde95f38c50a0d` | 27,605 | 27,605 | 804 | 380 | 1-38 | 27,605 | present | absent |

Available component and expected-metric columns in all three normalized seasons include `starts`,
`goals_scored`, `assists`, `clean_sheets`, `goals_conceded`, `saves`, `bonus`, `bps`,
`expected_goals`, `expected_assists`, `expected_goal_involvements`, and
`expected_goals_conceded`.

## Identity Methodology And Limitations

Team identities use normalized source team names plus tracked aliases in `data/manual/team_aliases.csv`.
The implementation does not assume FPL numeric team IDs are stable across seasons. The real run
created 24 canonical teams and 60 team-season mappings.

Player identities use non-null FPL player codes after checking season-level uniqueness. Manual
overrides are supported through `data/manual/player_identity_overrides.csv`; unresolved or ambiguous
candidates are written to ignored review output in `data/review/`.

Real player identity counts:

| Season | Method | Count |
|---|---|---:|
| 2022-23 | `fpl_code` | 778 |
| 2023-24 | `fpl_code` | 865 |
| 2024-25 | `fpl_code` | 804 |

Identity-code diagnostics:

| Season | Player-season rows | Null FPL codes | Duplicate code rows |
|---|---:|---:|---:|
| 2022-23 | 778 | 0 | 0 |
| 2023-24 | 865 | 0 | 0 |
| 2024-25 | 804 | 0 | 0 |

Contradictory cross-season FPL-code collisions: 0.

Review candidates: 0 rows for this three-season run. This is credible for the audited seasons
because every source player-season row has a non-null FPL code, no code duplicates occur within a
season, and no cross-season code maps to contradictory concurrent identities.

Limitations:

- Name-only linking is intentionally not used for high-confidence identity matching.
- Manual overrides are available but not needed for the current three seasons.
- Assistant managers are preserved as `entity_type = assistant_manager`,
  `fpl_position = AM`, and `element_type = 5`; standard player modeling panels can deterministically
  exclude them with `entity_type == "player"`.

## Proven Table Grains And Keys

Phase 2 generated tables:

| Table | Shape | Grain/key |
|---|---:|---|
| `dim_team` | 24 x 3 | one row per canonical club |
| `team_season_map` | 60 x 7 | one row per source team per season |
| `dim_player` | 1,346 x 5 | one row per canonical player UID |
| `player_season_map` | 2,447 x 11 | one row per source player-season |
| `dim_fixture` | 1,140 x 16 | one row per source fixture per season |
| `fact_player_fixture` | 83,835 x 62 | `season, player_uid, fixture_key` |
| `features_player_fixture` | 83,835 x 38 | `season, player_uid, fixture_key` |

`fact_player_fixture` duplicate key count: 0.

## Cutoff Methodology

Exact historical FPL deadline timestamps were not available in the normalized Vaastav inputs. Phase
2 therefore uses a conservative inferred cutoff:

- `information_cutoff`: earliest fixture kickoff in the target season/gameweek
- `cutoff_method`: `inferred_earliest_gameweek_kickoff`
- `cutoff_source`: `historical_player_fixture.kickoff_time`
- `cutoff_is_exact`: `False`

This is not labelled as an official FPL deadline.

Final match statistics are not assumed to be available at kickoff. Phase 2 stores
`source_available_time` for match-derived results. If an exact completion timestamp is genuinely
available in future sources, it can be used. The current Vaastav-derived workflow does not expose
exact completion timestamps, so it uses the documented conservative rule:

```text
source_available_time = kickoff_time + 3 hours
source_available_method = kickoff_plus_3h_conservative_match_completion
```

Feature lineage must satisfy:

```text
source_available_time < information_cutoff
```

## Feature Definitions And Lineage

Feature definitions are registered in `src/fpl_forecast/features/registry.json`. Produced feature
columns include shifted player rolling sums, season-to-date player totals, prior-season player
aggregates, home/away flag, and team/opponent lagged goals primitives.

Feature missingness:

| Season | Missing previous fixture minutes | Missing prior-season minutes |
|---|---:|---:|
| 2022-23 | 794 | 26,505 |
| 2023-24 | 875 | 10,267 |
| 2024-25 | 806 | 8,508 |

Source-availability lineage evidence:

| Lineage column | Non-null rows | Strictly before cutoff |
|---|---:|---|
| `player_max_source_available_time` | 81,360 | yes |
| `team_max_source_available_time` | 81,988 | yes |
| `opponent_max_source_available_time` | 81,988 | yes |

All 83,835 fact rows use `kickoff_plus_3h_conservative_match_completion`.

## Leakage Tests And Adversarial Results

The leakage auditor checks:

- feature/fact key uniqueness
- no case-insensitive `xP`
- no same-fixture target/component columns among features
- all feature columns are registered
- parseable UTC timestamps
- player/team/opponent `source_available_time` values are strictly before `information_cutoff`
- GW1 current-season history is zero or missing
- prior-season joins point only to earlier seasons
- naive end-of-season cumulative-looking fields are absent

Unit tests deliberately introduce future-source timestamps, a source fixture that kicked off before
the cutoff but was not conservatively available before it, target fixture leakage, unshifted GW1
history, case-varied `XP`, later-season prior joins, and naive cumulative field names; the auditor
returns nonzero errors for those adversarial cases.

Real command result:

```text
uv run fpl audit-leakage --seasons 2022-23,2023-24,2024-25
Leakage audit passed.
```

## Double, Blank, Transfer, And Position-Change Examples

Transfer example:

```text
player_uid=player_code_101178
2022-23 source_player_id=407 James Ward-Prowse Southampton MID
2023-24 source_player_id=664 James Ward-Prowse West Ham MID
2024-25 source_player_id=531 James Ward-Prowse West Ham MID
2024-25 GW1 prior_season_minutes=3000, player_team_uid=team_west_ham
```

The `player_uid` remains stable, the season-specific player ID changes, the team changes from
Southampton to West Ham, prior player history remains available, and the new-season row uses the
West Ham team context.

Position-change example:

```text
player_uid=player_code_165153
2022-23 source_player_id=137 Timo Werner Chelsea FWD
2023-24 source_player_id=776 Timo Werner Spurs FWD
2024-25 source_player_id=509 Timo Werner Spurs MID
2024-25 first target row: fixture=2024-25:10, fpl_position=MID,
player_team_uid=team_tottenham_hotspur, prior_season_minutes=808
```

The `player_uid` remains stable, prior event history remains attached, and the target row keeps the
current-season FPL position `MID`.

Double gameweek example rows in the real fact table:

```text
2022-23 GW19 player_code_101105 Joe Bryan fixture 2022-23:188 minutes=0 points=0
2022-23 GW19 player_code_101105 Joe Bryan fixture 2022-23:64 minutes=0 points=0
2022-23 GW19 player_code_103955 Raheem Sterling fixture 2022-23:184 minutes=4 points=1
2022-23 GW19 player_code_103955 Raheem Sterling fixture 2022-23:64 minutes=0 points=0
```

Blank-gameweek semantics: Phase 2 does not synthesize rows for missing fixtures or blank gameweeks.
It preserves observed Vaastav player-fixture rows only. Candidate zero-minute row generation is not
implemented because effective-dated player registration history is not yet reconstructed.

Zero-minute rows:

| Season | Zero-minute rows |
|---|---:|
| 2022-23 | 15,160 |
| 2023-24 | 18,341 |
| 2024-25 | 16,039 |

Entity counts:

| Season | Player rows | Assistant-manager rows |
|---|---:|---:|
| 2022-23 | 26,505 | 0 |
| 2023-24 | 29,725 | 0 |
| 2024-25 | 27,283 | 322 |

GW1 behavior:

| Season | GW1 rows | Max season-to-date minutes | Prior-season non-null rows |
|---|---:|---:|---:|
| 2022-23 | 573 | 0 | 0 |
| 2023-24 | 658 | 0 | 483 |
| 2024-25 | 616 | 0 | 480 |

## Unresolved Data Limitations

- Exact historical FPL deadlines are not present, so cutoffs are inferred from earliest gameweek
  kickoff.
- Effective-dated player registration histories are not reconstructed; no missing candidate rows
  are generated.
- Player/team identities are evidence-based for available FPL codes and source names, but future
  seasons may require manual overrides.
- Assistant-manager rows exist in `2024-25` only in this run: 322 rows across 20 managers.
- No predictive modeling or target evaluation exists in Phase 2.

## Rebuild Commands

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
uv run fpl validate-data
```
