# Publication artifact policy

The repository tracks `scripts/build_paper_evidence.py`, the active decision configurations, and
the evidence registry because they are source inputs to manuscript-facing tables and figures. They
belong in the same reviewed code revision as the model and decision code.

The clean replay from committed revision `034830b041c1` established numerical equivalence and
promoted new immutable successors. The reviewed `paper/` directory is now visible to Git. It tracks:

- `paper/evidence_manifest.csv` and `paper/evidence_supersession.csv`;
- aggregate `paper/tables/*.csv` through Table 7 and the calibration bins;
- `paper/figures/*.svg` through Figure 7, plus the exact embedded PNG used as Figure 1 in v7;
- `paper/source_input_manifest.csv`, `paper/research_schema.csv`, `paper/manuscript_value_map.csv`,
  `paper/output_hashes.csv`, and `paper/prospective_publication_registry.csv`; and
- `paper/FIGURE_NOTES.md`.

The author-supplied v7 PDF is not tracked or altered. Figure 1's tracked PNG is byte-identical to
embedded image `X47.png` on PDF page 12; the PDF and image hashes and re-extraction procedure are
in `paper/FIGURE_NOTES.md`. The separate `figure1_system_architecture.svg` is an auxiliary
repository diagram, not the v7 image. Figure 4's rolling panel has 114 folds across three seasons;
v7's 76-fold xPoints claim is mapped separately to the Phase 6 run as E-M01.

The manuscript PDF, DOCX or other editing copy, operating-system metadata, duplicate PNG renders,
and superseded generated assets remain excluded. Raw or normalized third-party FPL data and
row-level research evidence must not be added under this policy. Prospective player-level Table 8
and Figure 8 have been removed from the current tracked bundle pending rights review; the paper
generator no longer creates them. This does not erase their earlier Git history.

The clean clone can inspect aggregate assets and run publication validation without restricted
inputs. Rebuilding every result requires the hashed excluded inputs and the source availability
record described in `README_SLOAN.md`; the public repository alone does not reproduce every table
or figure. No release/archive work is part of the Oct. 1 readiness pass.

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
