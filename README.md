# FPL Forecast

FPL Forecast is an end-to-end, time-aware Fantasy Premier League forecasting and
decision system. It combines data engineering, statistical football modelling,
probabilistic FPL point simulation, backtesting, operational publication, and exact squad
optimization in one reproducible `uv` workspace.

The project is also a technical foundation for later football performance forecasting and player
valuation work: the current FPL xPoints stack establishes the data lineage, modelling interfaces,
simulation machinery, and decision constraints needed before moving beyond FPL scoring.

Unofficial project. Not affiliated with, endorsed by, or associated with the Premier League or
Fantasy Premier League.

## Current Status

Phases 1 through 8 are implemented and covered by offline tests and phase reports. The repository is
locally operational and prepared for static GitHub Pages publication through the Vite frontend.
Genuine 2026-27 operation is still guarded until the official FPL payload identifies the target
season and passes the launch, rule, identity, and model checks. The project should not be described
as production proven or as evidence of sustained live-season FPL performance.

The latest public frontend preparation deliberately supports a waiting state. GitHub Pages hosts
only static frontend assets; it does not run the Python forecasting pipeline in the browser.

## What The System Does

- Archives immutable raw snapshots from official FPL endpoints and Vaastav historical files.
- Normalizes source data into player-fixture and fixture-grain Parquet tables with provenance.
- Builds stable cross-season team and player identities without trusting numeric IDs across seasons.
- Enforces source-availability lineage and leakage audits before model training.
- Evaluates baselines and model families with rolling-origin and Gameweek 1 backtests.
- Estimates team fixture probabilities, player minutes, component xPoints, and point distributions.
- Selects legal FPL squads, lineups, captaincy, vice-captaincy, and bench order with exact MILP
  optimization.
- Publishes validated frontend-ready artifacts atomically and preserves the last successful
  publication after failures.

## What Makes The Framework Different

### Time-Aware Lineage And Backtesting

Unlike approaches that train on retroactively aggregated season data, this framework treats
information availability as a first-class constraint. Historical match-derived features carry a
conservative `source_available_time`, normally `kickoff_time + 3 hours`, and are eligible only when
`source_available_time < information_cutoff`. The `+3h` rule is a conservative proxy when exact
completion or publication timestamps are unavailable; it reduces look-ahead inflation but is not a
mathematical guarantee against every possible leakage mode.

Rolling-origin and Gameweek 1 backtests reproduce the information that would reasonably have been
available before each forecast. Same gameweek/deadline-block results are excluded from training.

### Separation Of Team State And Player Expectation

The system separates team-level match expectations from player-level opportunity. Opponent-adjusted
Poisson models estimate fixture score distributions, clean-sheet probabilities, and outcome
probabilities. A Dixon-Coles low-score dependence model is retained as an experimental challenger
because audited backtests showed only marginal, mixed differences versus the independent Poisson
default. Separate expected-minutes models estimate appearance and role duration. The component
simulation combines team context, minutes, position, and player event rates into discrete FPL point
distributions.

### Closed-Loop Execution

This is an operational engine rather than a standalone notebook. A uniform `uv` workspace connects
versioned ingestion, normalization, identity resolution, data-quality validation, leakage audits,
rolling backtests, probabilistic forecasting, full-candidate MILP squad optimization, atomic
publication, failure recovery, and frontend-ready artifacts. The MILP layer uses SciPy's
HiGHS-backed solver and records solver status, objective bounds, gaps, runtimes, and diagnostics.

## Architecture

```mermaid
flowchart LR
  A["Official FPL API<br/>bootstrap-static, fixtures, event live"] --> B["Immutable raw snapshots"]
  C["Vaastav historical FPL data"] --> B
  B --> D["Normalized player-fixture<br/>and fixture tables"]
  D --> E["Identity, feature, cutoff,<br/>and leakage-audit layer"]
  E --> F["Team probabilities<br/>T0/T1/T2 and T3 challenger"]
  E --> G["Expected-minutes models<br/>M0-M6"]
  F --> H["Component xPoints simulation<br/>X0/X1/X2"]
  G --> H
  H --> I["MILP decision layer<br/>squad, lineup, captain, bench"]
  I --> J["Operational publication<br/>atomic latest-successful artifacts"]
  J --> K["Static Vite dashboard<br/>React, TypeScript, CSS"]
```

## Models And Decision System

Default model chain used by the current operational adapter:

- `T2_REGULARIZED_ATTACK_DEFENCE`: weighted ridge-penalized independent Poisson team model.
- `M3_EWMA_MINUTES`: deterministic expected-minutes baseline with strong rolling performance.
- `X2_TEAM_CONSTRAINED_SIM_M3`: team-goal-constrained xPoints simulation using M3 minutes.
- SciPy HiGHS MILP: full-candidate weekly squad, lineup, captain, vice-captain, and bench
  optimization.

Experimental challengers:

- `T3_DIXON_COLES`: low-score dependence correction, retained for research but not promoted over T2.
- `M5_REGULARIZED_STATE_SOFTMAX` and `M6_NONLINEAR_RECENCY_ENSEMBLE`: learned minutes/state
  challengers.
- `X2_TEAM_CONSTRAINED_SIM_M5`: xPoints challenger with M5 state probabilities.

Diagnostic baselines:

- Phase 3 point baselines such as global mean, position mean, recent form, recent minutes, and
  empirical-Bayes points per 90.
- Phase 4 team baselines `T0_LEAGUE_HOME_AWAY` and `T1_SHRUNK_ROLLING_TEAM_RATE`.
- Phase 7 `D0_PRICE_VALUE_BASELINE` for market-price comparison in decision backtests.

Operational safeguards:

- Launch detection rejects stale current-season payloads.
- Rule drift enters review instead of silently running changed FPL rules.
- Team/player identity ambiguity enters review.
- Publication is atomic and preserves the last-known-good output on failure.
- Frontend artifacts use a stable `phase8_frontend_v1` contract.

## Backtesting And Selected Results

Detailed evidence is in the phase reports. These validation seasons are not an untouched final
holdout; they were used during iterative development. Gameweek 1 results contain only two folds and
therefore have high uncertainty.

| Layer | Seasons / Mode | Population Or Grain | Selected Evidence |
| --- | --- | --- | --- |
| Team goals | `2023-24`, `2024-25`; rolling; fixture goal sides | T2 improved goal MAE to `0.9489` versus T1 `0.9847` and T0 `1.0269`; block-bootstrap T2 minus T0 goal-MAE difference `-0.0754`, 95% CI `[-0.0967, -0.0538]`. |
| Team probabilities | `2023-24`, `2024-25`; rolling; fixture/team-fixture | T2 clean-sheet Brier `0.1643` versus T1 `0.1675` and T0 `0.1727`; match-outcome log loss `0.9679` versus T1 `1.0118` and T0 `1.0696`. |
| Dixon-Coles challenger | `2023-24`, `2024-25`; rolling; fixture | T3 changed scores only marginally: outcome log loss `0.9677` versus T2 `0.9679`, but joint scoreline NLL worsened to `3.0332` versus T2 `3.0324`; T2 remains default. |
| xPoints | `2023-24`, `2024-25`; rolling; all observed player rows | Default X2-M3 MAE `0.9060`, RMSE `1.9543`, Spearman `0.7044`; Phase 3 B5 reference MAE `0.9824`, RMSE `2.0295`, Spearman `0.6501`. |
| xPoints distribution | `2023-24`, `2024-25`; rolling; player rows | X2-M3 conserved expected team goals with max absolute error `4.44e-16`; X2-M5 had better 5+ point Brier and zero-rate calibration but was not selected as the default MAE frontier. |
| Decision optimization | `2023-24`, `2024-25`; rolling weekly-reset benchmark | 380 MILP decisions solved with status `optimal`, max gap `2.96e-16`, mean runtime `0.0892s`; all squads legal. X2-M3 had the highest realized rolling weekly-reset score in this pass, but paired intervals versus X2-M5 crossed zero. |
| Operations | Mocked 2026-27 launch and GW1-to-GW2 transition | Mocked target-season runs execute the real T2, minutes, xPoints, and MILP chain; injected failures preserve latest-successful artifacts. Genuine 2026-27 production evidence is still pending. |

## Frontend And Dashboard

The frontend in `frontend/` is Vite, React, TypeScript, and ordinary CSS. It reads static copies of
the latest `phase8_frontend_v1` artifacts from `frontend/public/data/`. It does not execute Python,
fetch live FPL data, run optimization, or modify operational outputs.

The manual GitHub Pages workflow builds the frontend without `npm run sync-data`, so the initial
public deployment can show a safe waiting state instead of local mocked recommendations. Mocked data
must remain visibly labelled when synchronized for local review. Forecast publication automation
remains separate from static hosting.

Local Python dashboard support from Phase 8 still exists:

```bash
uv run fpl dashboard --smoke
uv run fpl dashboard
```

## Quick Start

Requirements:

- Python 3.12 or newer
- `uv`
- Node.js 20 or newer for the frontend
- CPU-only execution is sufficient

Install and verify the Python workspace:

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

Run the static frontend locally:

```bash
cd frontend
npm ci
npm run dev
```

Open the Vite URL ending in `/fpl-forecast/`. To review a local mocked operational publication in
the frontend, first run a mocked refresh from the repository root and then synchronize the frontend
data:

```bash
uv run fpl refresh-operational --season 2026-27 --mock-launch --force
cd frontend
npm run sync-data
npm run dev
```

## Operational Workflow

Safe current-season status checks:

```bash
uv run fpl check-season-launch --season 2026-27
uv run fpl refresh-operational --season 2026-27 --status-only
uv run fpl operational-status
```

Representative target-season operation without claiming genuine live data:

```bash
uv run fpl refresh-operational \
  --season 2026-27 \
  --mock-launch \
  --run-id phase8_mock_transition \
  --force

uv run fpl verify-operational-readiness
uv run fpl dashboard --smoke
```

Generated raw data, normalized data, reports, logs, operational outputs, frontend synchronized data,
`node_modules/`, and build outputs are ignored by Git.

## Repository Structure

```text
src/fpl_forecast/        Python package and CLI implementation
tests/                   Offline tests and fixtures
data/manual/             Versioned manual identity templates
frontend/                Static Vite frontend
docs/deployment/         GitHub Pages setup notes
PHASE*_REPORT.md         Phase evidence and limitations
PHASE1_AUDIT.md          Phase 1 audit evidence
outputs/synthetic_demo/  Quarantined pre-real-data synthetic demo artifacts
```

Generated real-data artifacts live under ignored directories such as `data/raw/`,
`data/normalized/`, `reports/`, `outputs/operational/`, and `logs/operational/`.

## Known Limitations

- Genuine 2026-27 operation is not yet proven.
- Historical candidate universes are reconstructed from observed player-fixture rows, not complete
  archived pre-deadline squad-registration snapshots.
- Historical fixture schedules from Vaastav are retrospective, so result leakage is controlled but
  original deadline-time fixture uncertainty can remain.
- GW1 validation has only two folds.
- Transfer planning is proven only for small no-chip cases; full manager-state transfer backtesting
  needs bank, purchase prices, free transfers, and transfer history.
- Chips, hosted scheduling, authentication, production monitoring, and commercial data-provider
  integration are not implemented.
- No claims are made about real-world FPL rank, profitability, causal performance, or professional
  club validation.

## Roadmap

Phase A: FPL forecasting and decision system.

- Phases 1-8: data ingestion, identity, leakage-safe backtesting, team models, minutes, xPoints,
  optimization, operational readiness, and local dashboard contract.
- Phase 9: public dashboard and clean web publication.
- Phase 10: xPoints post-mortem after live evidence exists.

Next major branch: B, football performance forecasting and player valuation.

The current FPL xPoints system is the technical foundation. Later work would estimate expected
on-field contribution, expected minutes, role-adjusted performance, team and opponent context,
future performance over defined periods, and value relative to age, cost, or contract. That branch
is not implemented here.

## Data Sources And Attribution

This repository uses or supports ingestion from:

- Official Fantasy Premier League API endpoints used by the ingestion code:
  `bootstrap-static/`, `fixtures/`, and `event/{gameweek}/live/`.
- Vaastav's Fantasy Premier League historical dataset repository for historical FPL CSV files.

Raw and normalized third-party data are intentionally excluded from Git. See [DATA_NOTICE.md](DATA_NOTICE.md)
for data-rights boundaries. No Premier League, Fantasy Premier League, or club logos, crests, or
copied visual identity are included in the tracked source tree.

## References

- [Vaastav Fantasy Premier League historical dataset](https://github.com/vaastav/Fantasy-Premier-League)
- [Official Fantasy Premier League rules](https://fantasy.premierleague.com/help/rules)
- [Premier League Terms of Use](https://www.premierleague.com/terms-and-conditions)
- Dixon, M. J. and Coles, S. G. (1997), "Modelling Association Football Scores and Inefficiencies
  in the Football Betting Market," *Applied Statistics*, 46(2), 265-280,
  [doi:10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065)
- [SciPy `optimize.milp` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
- [HiGHS documentation](https://highs.dev/)
- [GNU Affero General Public License version 3](https://www.gnu.org/licenses/agpl-3.0.en.html)
- [Vaastav repository licence](https://github.com/vaastav/Fantasy-Premier-League/blob/master/LICENSE)
- Graham, Ian (2024), *How to Win the Premier League*; included here as conceptual inspiration, not
  as a technical specification or a reproduced proprietary model.

## Licence

Daniel Mehta's original source code in this repository is licensed under
`AGPL-3.0-only`; see [LICENSE](LICENSE). The AGPL permits use, modification, distribution, and
commercial use, subject to its terms, including the requirement that covered modified network
versions provide corresponding source under the AGPL.

Third-party dependencies remain under their respective licences; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The source-code licence does not relicense
third-party data, official FPL content, Premier League material, player or team identities,
trademarks, or database rights; see [DATA_NOTICE.md](DATA_NOTICE.md).

## Disclaimer

Unofficial project. Not affiliated with, endorsed by, or associated with the Premier League or
Fantasy Premier League. This repository is for experimental research and engineering demonstration
only and does not provide official FPL advice, legal advice, or data-rights clearance.
