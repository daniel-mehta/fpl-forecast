# SSAC27 research repository guide (manuscript v7)

This guide describes the public evidence for *From Forecasts to Decisions*, v7, as prepared for the
October 1 abstract submission. It is a research audit guide, not a claim that the public repository
contains every input row. The author-supplied v7 PDF is the wording authority and is not tracked.
The exact material wording handoff is in
[`docs/research/v7-material-corrections.md`](docs/research/v7-material-corrections.md); no manuscript
file has been silently edited.

## Start here

| Item | Location and meaning |
| --- | --- |
| Claim-to-evidence index | [`paper/manuscript_value_map.csv`](paper/manuscript_value_map.csv): v7 page, value, authoritative run, source hash, and scope. |
| Evidence provenance | [`paper/evidence_manifest.csv`](paper/evidence_manifest.csv) and [`paper/evidence_supersession.csv`](paper/evidence_supersession.csv): source artifacts and superseded run IDs. |
| Inputs and schema | [`paper/source_input_manifest.csv`](paper/source_input_manifest.csv) and [`paper/research_schema.csv`](paper/research_schema.csv): source URLs, revisions, retrieval times, SHA-256 hashes, roles, and normalized-column descriptions. |
| Aggregate assets and hashes | [`paper/tables/`](paper/tables), [`paper/figures/`](paper/figures), [`paper/FIGURE_NOTES.md`](paper/FIGURE_NOTES.md), and [`paper/output_hashes.csv`](paper/output_hashes.csv). |
| Live publication record | [`paper/prospective_publication_registry.csv`](paper/prospective_publication_registry.csv): official GW1–GW6 publication runs, not scored accuracy results. |

The 83,835 historical observations across 2022-23 to 2024-25 are **entity-fixture rows**. They
include 322 Assistant Manager records excluded from football-player models; thus 83,513 of these
rows are football-player rows. The v7 76-fold xPoints result uses
`phase6_xpoints_rolling_goalkeeper_corrected_exact` (E-M01). The later Figure 4 rolling panels use
114 folds across three seasons and must not be cited as the source for the 76-fold v7 result. The
v7 D1 count uses the corrected 380-decision run (E-M02). Figure 1 in v7 is the tracked embedded
PNG from PDF page 12, not the separate repository architecture SVG.

## Data and rights boundary

The public `main` bundle contains code, configurations, aggregate manuscript evidence, a schema,
hashes, and provenance metadata. It excludes raw Vaastav CSVs, raw official FPL snapshots,
normalized Parquet panels, row-level scored predictions/decisions, and private operational audits.
These inputs derive from FPL/Premier League data; Vaastav's repository licence covers its code but
does not convey ownership of the underlying FPL or Understat data. We do not infer redistribution
rights from public accessibility or from a project's own code licence. Former player-level
prospective Table 8 and Figure 8 are excluded from the current submission bundle pending review.
The separate frozen public forecast branch and dashboard are operational publications, not a
licence for research-dataset redistribution.

Each GW1–GW6 registry row links to its frozen public `run_manifest.json`. That manifest records
the official snapshot URLs, retrieval times, and hashes used for that publication. The July 2026
official snapshots in the input manifest belong to the separate preseason validation/convergence
experiment; they must not be substituted for any later official run's inputs.

Sloan must confirm whether pinned source links, retrieval code, hashes, schema, and aggregate
outputs meet its research-data requirement, or whether additional licensed data access is needed.
That question does not delay preparation of the rights-safe materials here. See
[`DATA_NOTICE.md`](DATA_NOTICE.md) for the project boundary; check each upstream source's current
terms before retrieval or sharing.

## Reproduction levels

1. **Public clone only:** inspect the manuscript mappings and aggregate outputs; verify output
   hashes and restricted-data policy. This needs no third-party dataset.
2. **Source retrieval where lawful and available:** use the pinned Vaastav Git revision
   `f2090d378ebd1b0c3d14884770dde95f38c50a0d` and the exact URLs and expected hashes in the
   input manifest. The FPL API snapshots are time-specific: a later response cannot replace their
   recorded bytes merely because the endpoint URL is the same.
3. **Exact retained-input replay:** regenerate aggregate evidence only after placing the original
   hashed raw, normalized, and scored-run inputs in their documented ignored paths. The archived
   three-season Phase 2 panel has SHA-256
   `a30006cd181249edea06efc92a7356fd3905d57f3bf8e7e8d37d570f9f0dfb4a`; a newer local
   panel containing 2025-26 does not substitute for it. Some run outputs are not redistributed, so
   this level requires the author's lawful retained artifacts and source availability.

For the public-clone check, from a disposable clone at the submitted commit:

```bash
uv sync --locked --dev
.venv/bin/python scripts/check_sloan_bundle.py
.venv/bin/python -m pytest -q tests/test_sloan_bundle.py tests/test_phase9b2_publication.py -m "not slow"
```

To retrieve historical inputs **only where permitted**, run the following in a disposable clone
and compare the actual CSV bytes with `paper/source_input_manifest.csv`. Retrieval times and local
snapshot names will differ. A hash mismatch means the input is not an exact retained-source replay.

```bash
for season in 2022-23 2023-24 2024-25 2025-26; do
  uv run fpl ingest-historical --season "$season" \
    --revision f2090d378ebd1b0c3d14884770dde95f38c50a0d
  uv run fpl normalize-historical --season "$season"
done
uv run fpl build-panel --seasons 2022-23,2023-24,2024-25
```

The tracked [`scripts/build_paper_evidence.py`](scripts/build_paper_evidence.py) reads existing
ignored run artifacts and writes aggregate tables and figures; it does **not** run the models. Run
it only in a disposable clone after all required inputs match their hashes. Its optional PNG
rendering needs Node `sharp`; the tracked research figures are SVG except the exact PDF-embedded
Figure 1 PNG. A successful command on substitute inputs does not validate manuscript equivalence.
The Python dependency lock is `uv.lock`; frontend dependencies are locked in
`frontend/package-lock.json`. Exact full-paper clean-clone replay and a DOI/archive are outside the
October 1 readiness pass.

## Known provenance limits

The original Phase 6 76-fold xPoints manifest records its data and configuration hashes but lacks
an exact source-code commit/tree identity; the exact historical configuration bytes are not
established in the current tree. Its metrics file and SHA-256 are retained and mapped. Some older
team/minutes manifests identify dirty commits without a complete tracked-diff hash. These gaps
limit exact source-tree replay and are disclosed rather than guessed. The live publication record
proves successful pre-deadline publication through GW6, not accuracy or completion of the
prospective scoring protocol. Missing deadlines must be recorded, never backfilled.
