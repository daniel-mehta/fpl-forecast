# FPL Forecast

FPL Forecast is an end-to-end, time-aware Fantasy Premier League forecasting and
decision system. It combines data engineering, statistical football modelling,
probabilistic FPL point simulation, backtesting, operational publication, and squad
optimization in one reproducible `uv` workspace.

**Public dashboard:** [daniel-mehta.github.io/fpl-forecast](https://daniel-mehta.github.io/fpl-forecast/)

Unofficial project. Not affiliated with, endorsed by, or associated with the Premier League or
Fantasy Premier League.

## Status
The first official-data 2026-27 GW1 forecast is published through the manually triggered,
clean-runner workflow. The reviewed implementation now reconstructs completed current-season
results for manual GW2-and-later publication, but no real GW2 publication has yet validated that
path operationally. Launch verification covers the public artifacts and operational lineage, not
predictive superiority: live-season forecast performance has not yet been established and
scheduling remains disabled. The promoted preseason
implementation is commit `365f0009a4e12397555c71cc950c7b4ef80c3ca4`.

The currently deployed dashboard contains the first verified official-data GW1 publication. The
promoted preseason hybrid simulator is implemented at commit `365f0009`, but will not appear on the
public dashboard until the official publication workflow is run again.

## What The System Does

- Archives immutable raw snapshots from official FPL endpoints and Vaastav historical files.
- Normalizes source data into player-fixture and fixture-grain Parquet tables with provenance.
- Builds stable cross-season team and player identities without trusting numeric IDs across seasons.
- Enforces source-availability lineage and leakage audits before model training.
- Evaluates baselines and model families with rolling-origin and Gameweek 1 backtests.
- Estimates team fixture probabilities, player minutes, component xPoints, and point distributions.
- Selects legal FPL squads, lineups, captaincy, vice-captaincy, and bench order with exact MILP
  baselines plus an expected-realized optimizer that accounts for ordinary automatic substitutions.
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
model computes analytic, full-precision component expectations and mean xPoints. A deterministic
joint fixture simulation then combines team context, minutes, position, and player event rates into
shared scorelines and discrete FPL point distributions.

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
  E --> G["Expected-minutes models<br/>M0-M7"]
  F --> H["Hybrid xPoints X2<br/>analytic means + joint fixtures"]
  G --> H
  H --> I["Decision layer<br/>D1 MILP and D2 expected-realized"]
  I --> J["Operational publication<br/>atomic latest-successful artifacts"]
  J --> K["Static Vite dashboard<br/>React, TypeScript, CSS"]
```

## Models And Decision System

Default model chain used by the current operational adapter:

- `T2_REGULARIZED_ATTACK_DEFENCE`: weighted ridge-penalized independent Poisson team model.
- `M7_HIERARCHICAL_AVAILABILITY_STATE`: explicit DNP, substitute, start-under-60, start-60-to-89,
  and start-90 state model used for official GW1 operation because it gives coherent GW1
  appearance/start/reached-60 probabilities.
- `X2_TEAM_CONSTRAINED_SIM_M7`: promoted hybrid xPoints model using M7 minutes, hierarchical
  attacking-rate shrinkage, and current-squad share allocation. Simulator
  `preseason_hybrid_fixture_v1`, contract `xpoints_hybrid_v1`, calculates analytic full-precision
  component expectations and mean xPoints, including direct conditional points given appearance.
  It uses 10,000 joint fixture draws for shared scorelines and player outcome distributions, with
  team-goal conservation and stable SHA-256-derived random streams.
- `D2_EXPECTED_REALIZED_POINTS`: full-candidate D1 MILP seed plus deterministic one-swap
  expected-realized search. It scores ordinary automatic substitutions, bench order, goalkeeper-only
  goalkeeper replacement, captain non-appearance, and vice-captain fallback from M7 appearance
  probabilities by enumerating all 32,768 squad appearance states under an explicit independence
  assumption. The evaluator is exact under that assumption; D2 remains a bounded heuristic search,
  not a global optimizer proof. This exact appearance-state evaluator is separate from the 10,000
  player-outcome draws.

Experimental challengers:

- `T3_DIXON_COLES`: low-score dependence correction, retained for research but not promoted over T2.
- `M5_REGULARIZED_STATE_SOFTMAX` and `M6_NONLINEAR_RECENCY_ENSEMBLE`: learned minutes/state
  challengers. M5 remains competitive and is retained as the main operational challenger.
- `X2_TEAM_CONSTRAINED_SIM_M5`: xPoints challenger with M5 state probabilities.

Diagnostic baselines:

- Phase 3 point baselines such as global mean, position mean, recent form, recent minutes, and
  empirical-Bayes points per 90.
- Phase 4 team baselines `T0_LEAGUE_HOME_AWAY` and `T1_SHRUNK_ROLLING_TEAM_RATE`.
- `M3_EWMA_MINUTES`, retained as a deterministic expected-minutes baseline. Its historical rolling
  MAE is strong, but its mechanically derived appearance/start probabilities are not treated as
  calibrated state probabilities.
- Phase 7 `D0_PRICE_VALUE_BASELINE` for market-price comparison in decision backtests.
- `D1_MEAN_ONLY_MILP`, retained as the exact full-candidate weekly MILP benchmark for nominal
  starting-XI expected points and captain bonus.

Operational safeguards:

- Launch detection rejects stale current-season payloads.
- Rule drift enters review instead of silently running changed FPL rules.
- Team/player identity ambiguity enters review.
- Teams without Premier League history use a fold-fitted newly observed/promoted-team prior instead
  of an exact neutral attack/defence effect.
- Analytic component expectations reconcile with analytic total xPoints at full precision; joint
  draws share fixture scorelines and conserve simulated team goals.
- Publication is atomic and preserves the last-known-good output on failure.
- Frontend artifacts use a stable `phase9_frontend_v1` contract.

### Preseason Simulation Audit

The preseason audit found and corrected semantic and numerical defects before any 2026-27 outcomes
existed: appearance was applied more than once to some expectations; clean-sheet and
goals-conceded eligibility could be discounted more than once; player draws did not share fixture
scorelines; random streams depended on row order; 80 draws were unstable; and some threshold-based
scoring used mean/divisor approximations. These were audit corrections, not evidence that the
hybrid simulator is universally more accurate.

Goal scoring is season-aware: goalkeeper goals receive six points through `2023-24` and ten points
from `2024-25` onward; defender, midfielder, and forward goal values are unchanged. Historical
realised reconstruction still matches all 113,260 evaluated rows exactly because those data contain
no realised goalkeeper goals. Expected points and simulated distributions nevertheless change when
goalkeepers have non-zero scoring probability.

## Backtesting And Selected Results

Detailed evidence is in the phase reports. These validation seasons are not an untouched final
holdout; they were used during iterative development. Gameweek 1 results contain only three folds
and therefore have high uncertainty. In particular, 2025-26 informed model hardening and is not an
untouched final holdout.

| Layer | Evaluation scope | Selected evidence |
| --- | --- | --- |
| Team goals | `2023-24`, `2024-25`; rolling; fixture goal sides | T2 improved goal MAE to `0.9489` versus T1 `0.9847` and T0 `1.0269`; block-bootstrap T2 minus T0 goal-MAE difference `-0.0754`, 95% CI `[-0.0967, -0.0538]`. |
| Team probabilities | `2023-24`, `2024-25`; rolling; fixture/team-fixture | T2 clean-sheet Brier `0.1643` versus T1 `0.1675` and T0 `0.1727`; match-outcome log loss `0.9679` versus T1 `1.0118` and T0 `1.0696`. |
| Dixon-Coles challenger | `2023-24`, `2024-25`; rolling; fixture | T3 changed scores only marginally: outcome log loss `0.9677` versus T2 `0.9679`, but joint scoreline NLL worsened to `3.0332` versus T2 `3.0324`; T2 remains default. |
| xPoints | `2023-24`, `2024-25`; rolling; all observed player rows | Season-aware default X2-M3 MAE `0.9065`, RMSE `1.9546`, Spearman `0.7044`; Phase 3 B5 reference MAE `0.9824`, RMSE `2.0295`, Spearman `0.6501`. |
| xPoints distribution | `2023-24`, `2024-25`; rolling; player rows | X2-M3 conserved expected team goals with max absolute error `4.44e-16`; X2-M5 had better 5+ point Brier and zero-rate calibration but was not selected as the default MAE frontier. |
| Decision optimization | `2023-24`, `2024-25`; corrected rolling weekly-reset benchmark | 380 D1 MILP decisions solved with status `optimal` and reported gap exactly `0` in every case; season-aware X2-M3 mean runtime was `0.1690s`, and all squads passed the complete legality audit. X2-M3 had the highest realized rolling weekly-reset score in this pass. |
| Operations | Official-data 2026-27 GW1 publication | The clean-runner chain published the frozen GW1 forecast successfully. This is operational evidence only; no 2026-27 accuracy result exists. |
| Phase 9B1.2 GW1 hardening | `2023-24`-`2025-26`; GW1 folds; 1,964 rows | M7 was selected for coherent operational probabilities, not as a universal rolling winner. Its legacy 80-draw evidence is compared with the promoted hybrid below. |
| Phase 9B1.3 optimizer hardening | `2023-24`-`2025-26`; three GW1 weekly-reset decisions | After the captaincy and season-aware goalkeeper-scoring corrections, D2's realized advantage over D1 remained `+1.33` points (fold differences `+2`, `0`, `+2`). Mean expected-realised points were `48.7583` for D1 and `49.4873` for D2. This descriptive result is not proof of season-level superiority. |

`expected_points` is an unconditional mean. D1 therefore maximizes the sum of unconditional means
for the starting XI plus the same unconditional mean once more for the captain; it does not multiply
the captain coefficient by appearance probability again. D2 did not contain that same
double-discounting error: its fixed-squad evaluator combines conditional-on-appearance points with
explicit appearance-state probabilities. Its historical results nevertheless changed because every
D2 search begins from the corrected D1 solution.

The corrected authoritative decision runs are
`phase7_goalkeeper_scoring_corrected_decisions_rolling_real_clean_034830b041c1` and
`phase9b13_goalkeeper_scoring_corrected_exact_decisions_gw1_clean_034830b041c1`. The structured
registry at
`src/fpl_forecast/decision/evidence_registry.json` records which pre-fix and earlier corrected runs
they supersede. Those earlier artifacts, including the decision evidence published with `v0.1.0`,
remain immutable historical records and must not be used as current publication evidence.

### Promoted Hybrid Simulator: Comparable Historical GW1 Evidence

The same 1,964 player rows and three GW1 folds were evaluated with the old 80-draw M7 simulator and
the promoted hybrid:

| Metric | Old 80-draw M7 | Hybrid |
| --- | ---: | ---: |
| MAE | 1.3032 | 1.3726 |
| RMSE | 2.1464 | 2.1296 |
| Spearman | 0.4895 | 0.4937 |
| P(5+) Brier | 0.08051 | 0.07755 |
| Central 80% coverage | 0.8819 | 0.9078 |

MAE worsened in all three folds. RMSE, P(5+) Brier, and central-80 coverage improved in all three;
pooled Spearman improved slightly but was mixed by fold. No retuning was performed after observing
these results. With only three historical GW1 folds, the hybrid's main justification is corrected
semantics, deterministic behavior, shared fixture coherence, and stronger distributional evidence,
not universal predictive superiority.

Ten thousand production draws were retained after a deterministic comparison with 20,000 draws:
the P95 absolute simulated-mean difference was `0.03069` points, the P95 absolute P(5+) difference
was `0.00525`, rank correlation was `0.999862`, top-15 overlap was `15/15`, and exact reruns were
identical.

Mean xPoints and component expectations are calculated analytically and therefore do not depend on
draw count. The 10,000 joint-fixture draws estimate discrete outcome distributions, prediction
intervals, zero-point probabilities, and tail probabilities such as P(5+).

### Prospective 2026-27 GW1 Example

**Prospective preseason example, not an accuracy result.** The non-published season-aware validation
successor contains 554 eligible player projections across 10 fixtures. D2 retains the `3-5-2`
formation, Saka as captain, and B.Fernandes as vice-captain, with expected-realized objective
`55.5161`, squad cost `£100.0m`, bank `£0.0m`, and optimizer status `heuristic_feasible`. It replaces
two bench squad members and changes the bench order while retaining the starting XI; no corrected
forecast has been published.

## Frontend And Dashboard

The frontend in `frontend/` is Vite, React, TypeScript, and ordinary CSS. It reads static copies of
the latest `phase9_frontend_v1` artifacts from `frontend/public/data/`. It does not execute Python,
fetch live FPL data, run optimization, or modify operational outputs.

The dashboard's Player Finder returns up to five highest-projected players for a chosen position and
maximum official price. Optionally choose a player to compare same-position replacements from the
same current official forecast. It does not account for complete-squad legality, selling prices,
bank balance, free transfers, transfer hits, or multi-gameweek planning.

The static `Your Team` view accepts a manually entered legal 15-player squad and applies the
Python-authoritative fixed-squad D2 lineup, bench-order and captaincy logic in the browser. It uses
only the frozen target-Gameweek projection, including the additive direct
`expected_points_given_appearance` simulation output. Squad selections, entered selling prices,
bank, free transfers and transfer-mode preference are saved only in browser `localStorage` and are
invalidated when the frozen forecast identity changes. Independent suggestions are the default:
each outgoing-player group is a separate exact one-transfer comparison against the unchanged squad,
using the original bank and club composition. Users may instead select a combined plan, where the
entered free-transfer count is the maximum connected number of moves and up to two alternatives per
outgoing player are evaluated with every other primary choice fixed. Combined search uses entered
selling prices and the evolving bank, rolls unused free transfers, and searches every permitted
depth so a temporary downgrade may fund a stronger later upgrade. Either mode recommends a paid
move only when its improvement exceeds the four-point hit. Both optimize only the currently
published Gameweek; projections are estimates rather than guarantees.

`Frontend CI` validates frontend changes with lint and a production build. `Deploy frontend to
GitHub Pages` automatically deploys qualifying frontend-only pushes to `main`, or can be selected
manually in Actions. It retrieves the exact frozen, sanitized official forecast bundle and fails if
that bundle cannot be retrieved and validated; it never generates a forecast, runs Python, or uses
sample data. Backend or forecasting changes still require `Publish official FPL forecast`, which
reconstructs the pinned historical inputs, retrieves fresh official `bootstrap-static/`,
`fixtures/`, and required prior `event/{gameweek}/live/` inputs, runs the verified
forecast chain, freezes the sanitized public bundle, and then deploys it. Ordinary UI work therefore
does not require backend forecast generation. For GW2 and later, every prior official event must be
finished and data-checked, every assigned fixture must be finished and provisional, and every raw
source retrieval must precede the target deadline.

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
uv run pytest -q -m "not slow"
```

The complete suite runs separately through the manually triggered and nightly `Full Python suite`
workflow.

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

Manual official publication, frozen-bundle migration, and failure-preservation instructions are
documented in `docs/deployment/github-pages.md`. Operational review and recovery procedures are in
`docs/operations/manual-publication-and-recovery.md`.

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
The tracked publication generator and the policy for manuscript-facing tables, figures, manifests,
and release attachments are documented in
[`docs/research/publication-artifacts.md`](docs/research/publication-artifacts.md). The current
repository does not yet claim clean-clone reproduction of those assets because the required
research evidence bundle and redistribution review are separate pending work.

## Known Limitations

- The verified public forecast covers 2026-27 GW1 only; no sustained live-season accuracy evidence
  exists yet.
- D2 exactly enumerates independent appearance states for ordinary bench and captain contingency,
  but player absences can be correlated in reality. Its full-pool one-swap search is bounded after
  the exact D1 seed and is not a transfer-aware season optimizer. Its corrected observed
  `+1.33`-point realized advantage over D1 (fold differences `+2`, `0`, `+2`) comes from only three
  historical GW1 decisions and is descriptive, not proof of superiority. D2 did not share D1's
  double-discounting error, but its result changed because its search starts from the D1 solution.
- Exact event timing within matches is not simulated. Under official scoring, a player substituted
  after 60 minutes may retain a clean sheet despite a later concession and incurs goals-conceded
  deductions only for goals conceded while on the pitch. The simulator approximates these cases
  with the final simulated scoreline and reached-60 state.
- M7 is selected for official GW1 coherence, but M3/M5 remain important challengers. M7 performs
  worse on rolling all-row MAE and should not be treated as broadly superior.
- Historical candidate universes are reconstructed from observed player-fixture rows, not complete
  archived pre-deadline squad-registration snapshots.
- Historical fixture schedules from Vaastav are retrospective, so result leakage is controlled but
  original deadline-time fixture uncertainty can remain.
- GW1 validation has only three folds, and 2025-26 is hardening evidence rather than an untouched
  final holdout.
- Transfer planning is proven only for small no-chip cases; full manager-state transfer backtesting
  needs bank, purchase prices, free transfers, and transfer history.
- Chips, hosted scheduling, authentication, production monitoring, and commercial data-provider
  integration are not implemented.
- No claims are made about real-world FPL rank, profitability, causal performance, or professional
  club validation.

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
