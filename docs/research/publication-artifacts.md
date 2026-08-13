# Publication artifact policy

The repository tracks `scripts/build_paper_evidence.py`, the active decision configurations, and
the evidence registry because they are source inputs to manuscript-facing tables and figures. They
belong in the same reviewed code revision as the model and decision code.

The existing `paper/` directory remains ignored until the authoritative evidence is replayed from a
clean committed revision. The current assets are numerically verified, but some upstream manifests
identify an unavailable dirty source state. Moving those assets into Git before the clean replay
would preserve the provenance defect that the replay is intended to resolve.

After the clean replay, the smallest reviewable repository bundle should track:

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

## Clean replay boundary

The next evidence replay must start from a clean commit, use new non-overwriting run IDs, and record
that commit plus exact source-tree and input/output hashes in every successor manifest. It must
replay the corrected hybrid GW1 xPoints evidence, its convergence and closure checks, the rolling
and GW1 decision evidence affected by the unavailable dirty state, and the corrected prospective
validation. The evidence registry and correction inventory must then point to those successors
before the publication generator is run twice and its output digest is compared.

The prospective validation and simulation checks must use
`scripts/replay_clean_prospective_evidence.py`; the prior `/tmp` helpers are not replay inputs.

For successor IDs that differ from the current registry, the generator accepts
`FPL_PAPER_HYBRID_GW1_RUN`, `FPL_PAPER_DECISION_RUN`, `FPL_PAPER_PROSPECTIVE_RUN`, and
`FPL_PAPER_EVIDENCE_INVENTORY`. These overrides select immutable replay inputs; they do not permit
overwriting an existing run. The replay must not revise historical manifests or backfill a missed
official deadline.
