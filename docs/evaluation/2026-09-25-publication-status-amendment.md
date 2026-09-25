# 2026-09-25 publication status amendment

This is a status amendment to [`live-evaluation-protocol.md`](live-evaluation-protocol.md), not a
change to its registered populations, metrics, authority rule, or no-backfill policy. The original
protocol is retained unchanged at Git blob `6fbc556e84cbf84b6a2af366deaffd94003af3fa` (SHA-256
`34e3f8722ddb67cd94187039b1d07e93c168eb2bbcdc49293bc6c2c0c8369c5b`). Its pre-GW1
phrases describe the implementation state when it was written and are not current status claims.

The official `Publish official FPL forecast` GitHub Actions history now establishes successful
forecast, mandatory-gate, frozen-bundle, and deployment jobs for 2026-27 Gameweeks 1–6. For each
gameweek, the successful run's `REQUESTED_GAMEWEEK`, resolved `target_gameweek`, run manifest,
freeze timestamp, and recorded official deadline agree. All six freezes occurred before the
recorded deadline. Exact run links, deadlines, code SHAs, and frozen-branch commits are in
[`paper/prospective_publication_registry.csv`](../../paper/prospective_publication_registry.csv).

This changes operational status only. It does not establish prospective predictive accuracy,
calibration, decision superiority, or completion of the registered season-long evaluation. The
protocol's rule still applies: a missed deadline or failed authoritative publication must be
recorded and cannot be reconstructed later as prospective. Private prediction-side retention,
outcome-scoring readiness, and any required prospective-registry hashes should be audited before
their metrics are reported. The earlier statement that GW2+ publication was not implemented is
superseded by the verified workflow history.
