# FPL Forecast frontend

A basic static Vite, React and TypeScript view of the latest successful
`phase8_frontend_v1` publication. The Python pipeline remains responsible for data ingestion,
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
npm run build
npm run preview
```

`npm run sync-data` safely resolves `outputs/operational/latest_successful.json` and copies only the
seven frontend contract artifacts into `public/data/`. The generated data files, `node_modules/`
and `dist/` are ignored by Git. Original operational outputs are never modified.

The current operational publication uses representative mocked target-season data, so the UI
displays a prominent `DEMO DATA` warning. GitHub Pages and public deployment are not configured.

The repository documentation link defaults to the repository URL found in Git configuration. It
can be changed at build time with `VITE_REPOSITORY_URL`.
