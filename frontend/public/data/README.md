# Generated frontend data

Run `npm run sync-data` from `frontend/` to copy the seven public
`phase9_frontend_v1` artifacts from the latest successful operational run. Generated files in this
directory are ignored by Git. The public Pages build does not run this command; absent artifacts
produce the frontend's safe waiting state.

`optimized_lineup.csv` may include Phase 9B1.3 expected-realized optimizer diagnostics such as
autosub contribution, captain fallback value, expected substitutions, and the optimizer variant.
The diagnostics distinguish exact 32,768-state evaluation under independent appearances from the
heuristic local-search status, termination reason, and number of unique legal squads scored.
These columns are additive within the `phase9_frontend_v1` contract.
