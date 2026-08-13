# Publication artifact policy

The repository tracks `scripts/build_paper_evidence.py`, the active decision configurations, and
the evidence registry because they are source inputs to manuscript-facing tables and figures. They
belong in the same reviewed code revision as the model and decision code.

The clean replay from committed revision `034830b041c1` established numerical equivalence and
promoted new immutable successors. The `paper/` directory remains ignored only while the regenerated
bundle is verified by two identical generator runs.

After that comparison, the smallest reviewable repository bundle should track:

- `paper/evidence_manifest.csv` and `paper/evidence_supersession.csv`;
- `paper/tables/*.csv`;
- `paper/figures/*.svg`; and
- `paper/FIGURE_NOTES.md`.

The corresponding PNG renderings should be generated from the same clean revision and attached to
the release in a checksummed publication-support archive. A release archive should also contain the
generator command and stdout, the clean replay manifests, and a SHA-256 inventory. Duplicate PNG
files do not need to be committed when the SVG source and deterministic generator are tracked.

The manuscript PDF, DOCX or other editing copy, operating-system metadata, temporary render files,
and superseded generated assets should remain excluded. Raw or normalized third-party FPL data and
row-level research evidence must not be added under this policy. Their availability,
redistribution rights, data dictionary, attribution, licensing and clean-clone packaging are a
separate research-dataset task.

Until that legally publishable evidence bundle exists, a clean clone can inspect and run the
generator only after the required hashed input artifacts are supplied. Documentation must not claim
that the public repository alone reproduces every manuscript table or figure.

## Clean replay record

The completed replay started from clean commit `034830b041c1`, used non-overwriting run IDs, and
recorded commit, source-tree, input and output hashes in the successor manifests and
`reports/goalkeeper_scoring_fix/clean_replay_inventory_034830b041c1.json`. It replayed the corrected
hybrid GW1 xPoints evidence, convergence and closure checks, rolling and GW1 decision evidence, and
the corrected prospective validation. Separate comparison IDs established deterministic equality
after excluding only run identity, documented timestamps, runtime and process-memory fields.

The prospective validation and simulation checks must use
`scripts/replay_clean_prospective_evidence.py`; the prior `/tmp` helpers are not replay inputs.

For successor IDs that differ from the current registry, the generator accepts
`FPL_PAPER_HYBRID_GW1_RUN`, `FPL_PAPER_DECISION_RUN`, `FPL_PAPER_PROSPECTIVE_RUN`, and
`FPL_PAPER_EVIDENCE_INVENTORY`. These overrides select immutable replay inputs; they do not permit
overwriting an existing run. The replay must not revise historical manifests or backfill a missed
official deadline.
