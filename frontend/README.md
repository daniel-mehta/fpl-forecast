# FPL Forecast frontend

A basic static Vite, React and TypeScript view of the latest successful
`phase9_frontend_v1` publication. The Python pipeline remains responsible for data ingestion,
modeling, validation and atomic publication.

## Requirements

- Node.js 20 or newer
- npm
- a successful local operational run under `outputs/operational/`

## Local use

```bash
npm install
npm run sync-data
npm run dev
```

Open the Vite URL ending in `/fpl-forecast/`.

Build and preview the production bundle:

```bash
npm test
npm run lint
npm run build
npm run preview
```

`npm run sync-data` safely resolves `outputs/operational/latest_successful.json` and copies only the
seven frontend contract artifacts into `public/data/`. The generated data files, `node_modules/`
and `dist/` are ignored by Git. Original operational outputs are never modified.

Local operational publications can be either mocked or official-current. Mocked publications display
a prominent `DEMO DATA` warning. Official-current publications display player prices, opponents,
home/away context, forecast timestamps, and source/disclaimer text without the demo banner. When
generated artifacts are absent, including on the initial GitHub Pages build, the UI displays a safe
waiting state instead of forecast recommendations.

The projections table supports player/team search, position filtering, and inclusive minimum and
maximum price filters using official prices. Technical optimizer labels have keyboard- and
touch-accessible explanations. Publication timestamps are shown explicitly in UTC, and the
recommended squad is separated into the starting XI and ordered bench with total cost and remaining
bank.

The hash-based `#your-team` view works on the same static GitHub Pages document and uses only the
single frozen official forecast already loaded by the dashboard. Users manually select a legal
15-player squad, enter bank and free transfers, and correct each player's selling price when it
differs from the current official market price. The squad, selling prices, bank and free-transfer
count are stored only in browser `localStorage`; the saved record is invalidated when season,
Gameweek, run ID or the frozen player identity changes. No squad data are transmitted.

Your Team ports the Python D2 fixed-squad lineup refinement and exact independent-appearance
objective to TypeScript. Shared deterministic fixtures are recalculated by Python and asserted by
Vitest within an absolute `1e-8` tolerance. Transfer recommendations cover the entered maximum of
distinct same-position moves and the currently published Gameweek only. Affordability uses every
outgoing entered selling price plus the evolving bank; a four-point hit is applied only when the
user enters zero free transfers, and a paid move is shown only when its net improvement is positive.
Unused free transfers are rolled when the best exact retained plan uses fewer moves than the entered
maximum.

The multi-transfer search is deliberately bounded. For every owned player, the six strongest
same-position replacements by authoritative unconditional expected points enter a deterministic
beam of 120 partial legal-plan candidates. The strongest retained legal plan at each permitted
transfer depth receives exact inherited D2 refinement across 32,768 appearance states. The chosen
plan is the strongest exact result across depth zero and every permitted transfer depth; an
individually negative move can therefore survive the bounded beam when it releases funds for a
positive later upgrade. Up to two additional replacements per outgoing player are then exactly
evaluated with all other primary transfers fixed; illegal, duplicated or unaffordable dependent
alternatives are omitted. Reusable appearance distributions, bench probabilities and autosub
legality tables are memoized without changing the exact objective. The UI paints a calculating
state and discloses that this is a bounded shortlist, not a global multi-transfer optimum.

`expected_points_given_appearance` is an additive required column in
`player_gameweek_projections.csv`. It is exported directly from the xPoints simulation, never
reconstructed from rounded public xP. Frozen-bundle validation and publication validation fail
closed when it is absent. An existing frozen bundle must therefore be republished through the
normal authoritative publication workflow before Your Team becomes available; the dashboard itself
continues to render while the new view reports the missing contract clearly.

The manual official Pages workflow runs `sync-data` only after clean-runner reconstruction, an
official forecast, and fail-closed publication validation. Frontend-only CI still does not deploy,
because its clean runner has no last-successful ignored forecast artifacts. See
`docs/deployment/github-pages.md` from the repository root for setup and manual publication steps.

The repository documentation link defaults to the repository URL found in Git configuration. It
can be changed at build time with `VITE_REPOSITORY_URL`.
