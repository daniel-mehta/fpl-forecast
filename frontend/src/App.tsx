import { useEffect, useMemo, useState } from "react";
import { type FrontendData, type JsonRecord, loadFrontendData } from "./data";
import {
  availablePrices,
  filterProjections,
  groupSquad,
  latestOfficialRetrievalTimestamp,
  normalizePriceRange,
  squadRoleLabel,
  timestampLine,
} from "./dashboard";
import { InfoTooltip, InformationLabel } from "./InfoTooltip";
import {
  formatDate,
  formatComparisonValue,
  formatLabel,
  formatNumber,
  formatPercentage,
  formatPlayerStatus,
  formatPrice,
  formatTeam,
} from "./formatting";

const repositoryUrl =
  import.meta.env.VITE_REPOSITORY_URL ?? "https://github.com/daniel-mehta/fpl-forecast#readme";

function textValue(record: JsonRecord, key: string): string {
  const value = record[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

function isDemoData(data: FrontendData): boolean {
  const warnings = Array.isArray(data.manifest.warnings) ? data.manifest.warnings.join(" ") : "";
  return [textValue(data.status, "warning"), textValue(data.freshness, "source"), warnings].some(
    (value) => /mock|representative|demo/i.test(value),
  );
}

interface AppProps {
  initialData?: FrontendData;
}

function App({ initialData }: AppProps) {
  const [data, setData] = useState<FrontendData | null>(initialData ?? null);
  const [waiting, setWaiting] = useState(false);
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState("ALL");
  const [minPriceTenths, setMinPriceTenths] = useState<number | null>(null);
  const [maxPriceTenths, setMaxPriceTenths] = useState<number | null>(null);
  const [descending, setDescending] = useState(true);

  useEffect(() => {
    if (initialData) {
      return;
    }
    loadFrontendData().then(setData).catch(() => setWaiting(true));
  }, [initialData]);

  const projections = useMemo(() => {
    if (!data) {
      return [];
    }
    return filterProjections(data.projections, {
      search,
      position,
      minPriceTenths,
      maxPriceTenths,
    })
      .sort((left, right) => {
        const difference = Number(right.expected_points) - Number(left.expected_points);
        return descending ? difference : -difference;
      });
  }, [data, descending, maxPriceTenths, minPriceTenths, position, search]);

  if (waiting) {
    return (
      <>
        <main className="page-shell">
          <section className="load-state" role="status">
            <p className="eyebrow">Upcoming season</p>
            <h1>FPL Forecast</h1>
            <p>The forecast dashboard is preparing for the upcoming season.</p>
            <p>
              Projections will appear after official season data are available and validated. This
              project remains experimental.
            </p>
          </section>
        </main>
        <SiteFooter />
      </>
    );
  }

  if (!data) {
    return (
      <main className="page-shell">
        <section className="load-state" aria-live="polite">
          <h1>FPL Forecast</h1>
          <p>Loading forecast data...</p>
        </section>
      </main>
    );
  }

  const demo = isDemoData(data);
  const warning = textValue(data.status, "warning");
  const modelVariant = data.projections[0]?.model_variant ?? textValue(data.manifest, "models");
  const gameweek = textValue(data.status, "target_gameweek");
  const state = textValue(data.status, "state");
  const lineupSummary = data.lineup[0];
  const prices = availablePrices(data.projections);
  const { starters, bench } = groupSquad(data.squad);
  const officialRetrievalLine = timestampLine(
    "Official data retrieved",
    latestOfficialRetrievalTimestamp(data.freshness),
  );
  const forecastGenerationLine = timestampLine(
    "Forecast generated",
    data.freshness.generated_at,
  );
  const officialSuccess =
    state.toLowerCase() === "succeeded" &&
    textValue(data.freshness, "source_mode") === "official_current_season";

  function updatePrice(changed: "min" | "max", value: string) {
    const normalized = normalizePriceRange(changed, value === "" ? null : Number(value), {
      minPriceTenths,
      maxPriceTenths,
    });
    setMinPriceTenths(normalized.minPriceTenths);
    setMaxPriceTenths(normalized.maxPriceTenths);
  }

  function resetFilters() {
    setSearch("");
    setPosition("ALL");
    setMinPriceTenths(null);
    setMaxPriceTenths(null);
  }

  return (
    <>
      {demo && (
        <div className="demo-banner" role="status">
          DEMO DATA: Representative mocked recommendations, not valid live FPL advice
        </div>
      )}
      <header className="site-header">
        <div className="page-shell header-inner">
          <div>
            <p className="eyebrow">Experimental forecasting</p>
            <h1>FPL Forecast</h1>
          </div>
          <dl className="header-meta">
            <div>
              <dt>Season</dt>
              <dd>{textValue(data.status, "target_season") || "Not available"}</dd>
            </div>
            {gameweek && (
              <div>
                <dt>Forecast</dt>
                <dd>Gameweek {gameweek}</dd>
              </div>
            )}
            <div>
              <dt>Updated</dt>
              <dd>{formatDate(data.freshness.generated_at)}</dd>
            </div>
            <div>
              <dt>State</dt>
              <dd>
                <span className="state-badge">{formatLabel(state)}</span>
              </dd>
            </div>
          </dl>
        </div>
      </header>

      <main className="page-shell content">
        <section aria-labelledby="status-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Latest successful publication</p>
              <h2 id="status-heading">Status summary</h2>
            </div>
          </div>
          <div className="status-grid">
            <article className="summary-card">
              <h3>Operational state</h3>
              <p>{formatLabel(state)}</p>
            </article>
            <article className="summary-card">
              <h3>Data freshness</h3>
              {officialRetrievalLine && <p>{officialRetrievalLine}</p>}
              {forecastGenerationLine && <p>{forecastGenerationLine}</p>}
              {!officialRetrievalLine && !forecastGenerationLine && (
                <p>{data.freshness.stale === true ? "Stale data" : "Timestamp not available"}</p>
              )}
            </article>
            <article className="summary-card">
              <h3>
                <InformationLabel label="Model variant">
                  The forecasting model used to produce the displayed player projections.
                </InformationLabel>
              </h3>
              <p>{formatLabel(modelVariant)}</p>
            </article>
            <article className="summary-card">
              <h3>Forecast status</h3>
              <p>
                {officialSuccess
                  ? "Official data validated and forecast published successfully."
                  : textValue(data.status, "reason") || "No status detail available."}
              </p>
            </article>
          </div>
          {(warning || demo) && (
            <div className="notice" role="note">
              <strong>Limitations:</strong>{" "}
              {warning || "The displayed output is representative mocked data."}
            </div>
          )}
        </section>

        <section aria-labelledby="squad-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Expected-realized optimization</p>
              <h2 id="squad-heading">Recommended squad</h2>
            </div>
            {lineupSummary && (
              <div className="section-note optimizer-status-line">
                <span>Formation: {lineupSummary.formation}</span>
                <span>
                  Squad search: {formatLabel(lineupSummary.solver_status)}{" "}
                  <InfoTooltip label="Heuristic feasible">
                    The squad is legal and improved through deterministic search, but is not
                    guaranteed to be the global optimum.
                  </InfoTooltip>
                </span>
                {lineupSummary.lineup_refinement_status && (
                  <span>
                    Lineup refinement: {formatLabel(lineupSummary.lineup_refinement_status)}{" "}
                    <InfoTooltip label="Lineup refinement status">
                      The selected squad's lineup and bench order were refined within the documented
                      fixed-squad candidate space.
                    </InfoTooltip>
                  </span>
                )}
                {lineupSummary.termination_reason && (
                  <span>
                    Search limit: {formatLabel(lineupSummary.termination_reason)}{" "}
                    <InfoTooltip label="Configured iteration bound reached">
                      The squad search stopped after reaching its configured search limit.
                    </InfoTooltip>
                  </span>
                )}
              </div>
            )}
          </div>
          {lineupSummary && (
            <dl className="optimizer-summary" aria-label="Optimizer expected-realized diagnostics">
              <div>
                <dt>
                  <InformationLabel label="Optimizer">
                    The decision method used to select the legal squad, lineup, bench order and
                    captaincy.
                  </InformationLabel>
                </dt>
                <dd>{formatLabel(lineupSummary.optimizer_variant || lineupSummary.solver_name || "available")}</dd>
              </div>
              <div>
                <dt>
                  <InformationLabel label="Expected realized">
                    Projected points after accounting for appearances, automatic substitutions and
                    captain fallback.
                  </InformationLabel>
                </dt>
                <dd>{formatNumber(lineupSummary.expected_realized_total, 2)}</dd>
              </div>
              <div>
                <dt>
                  <InformationLabel label="Autosub value">
                    Expected points contributed by bench players replacing starters who do not
                    appear.
                  </InformationLabel>
                </dt>
                <dd>{formatNumber(lineupSummary.expected_autosub_contribution, 2)}</dd>
              </div>
              <div>
                <dt>
                  <InformationLabel label="Captain bonus">
                    Additional expected points generated by doubling the captain's score.
                  </InformationLabel>
                </dt>
                <dd>{formatNumber(lineupSummary.expected_captain_bonus, 2)}</dd>
              </div>
              <div>
                <dt>
                  <InformationLabel label="Vice fallback">
                    Expected captain bonus received when the captain does not play and the
                    vice-captain does.
                  </InformationLabel>
                </dt>
                <dd>{formatNumber(lineupSummary.expected_vice_captain_contingency, 2)}</dd>
              </div>
              <div>
                <dt>
                  <InformationLabel label="Expected subs">
                    Average number of automatic substitutions expected for this lineup.
                  </InformationLabel>
                </dt>
                <dd>{formatNumber(lineupSummary.expected_automatic_substitutions, 2)}</dd>
              </div>
              <div>
                <dt>
                  <InformationLabel label="Legal squads scored">
                    Number of legal candidate squads evaluated during optimization.
                  </InformationLabel>
                </dt>
                <dd>{formatNumber(lineupSummary.evaluated_squads, 0)}</dd>
              </div>
              <div>
                <dt>
                  <InformationLabel label="Unreplaced risk">
                    Probability that at least one absent starter cannot be replaced by an eligible
                    bench player.
                  </InformationLabel>
                </dt>
                <dd>{formatPercentage(lineupSummary.probability_unreplaced_starter)}</dd>
              </div>
            </dl>
          )}
          {lineupSummary && (
            <dl className="squad-meta" aria-label="Squad cost and budget">
              <div>
                <dt>Formation</dt>
                <dd>{lineupSummary.formation}</dd>
              </div>
              <div>
                <dt>Total squad cost</dt>
                <dd>{formatPrice(lineupSummary.cost_tenths)}</dd>
              </div>
              <div>
                <dt>Remaining bank</dt>
                <dd>{formatPrice(lineupSummary.bank_tenths)}</dd>
              </div>
            </dl>
          )}
          <div className="squad-groups">
            <SquadTable title="Starting XI" players={starters} />
            <SquadTable title="Bench" players={bench} />
          </div>
        </section>

        <section aria-labelledby="projections-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">
                {projections.length} matching {projections.length === 1 ? "player" : "players"}
              </p>
              <h2 id="projections-heading">Player projections</h2>
            </div>
          </div>
          <div className="controls">
            <label>
              Search players or teams
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search"
              />
            </label>
            <label>
              Position
              <select value={position} onChange={(event) => setPosition(event.target.value)}>
                <option value="ALL">All positions</option>
                <option value="GKP">Goalkeepers</option>
                <option value="DEF">Defenders</option>
                <option value="MID">Midfielders</option>
                <option value="FWD">Forwards</option>
              </select>
            </label>
            <label>
              Minimum price
              <select
                aria-label="Minimum price"
                value={minPriceTenths ?? ""}
                onChange={(event) => updatePrice("min", event.target.value)}
              >
                <option value="">No minimum</option>
                {prices.map((price) => (
                  <option key={price} value={price}>
                    {formatPrice(price)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Maximum price
              <select
                aria-label="Maximum price"
                value={maxPriceTenths ?? ""}
                onChange={(event) => updatePrice("max", event.target.value)}
              >
                <option value="">No maximum</option>
                {prices.map((price) => (
                  <option key={price} value={price}>
                    {formatPrice(price)}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={() => setDescending((current) => !current)}>
              Expected points: {descending ? "high to low" : "low to high"}
            </button>
            <button type="button" onClick={resetFilters}>
              Reset filters
            </button>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th scope="col">Player</th>
                  <th scope="col">Team</th>
                  <th scope="col">Position</th>
                  <th scope="col">Opponent</th>
                  <th scope="col" className="numeric">Price</th>
                  <th scope="col" className="numeric">Expected points</th>
                  <th scope="col" className="numeric">Expected minutes</th>
                  <th scope="col" className="numeric">Appearance</th>
                  <th scope="col" className="numeric">Start</th>
                  <th scope="col" className="numeric">At least 5 points</th>
                  <th scope="col">Status</th>
                </tr>
              </thead>
              <tbody>
                {projections.map((player) => (
                  <tr key={player.stable_player_id}>
                    <th scope="row">{player.player}</th>
                    <td>{formatTeam(player.team)}</td>
                    <td>{player.position}</td>
                    <td>
                      <span aria-label={`Opponent fixture: ${player.opponent_display || "No fixture"}`}>
                        {player.opponent_display || "No fixture"}
                      </span>
                    </td>
                    <td className="numeric">{formatPrice(player.price_tenths)}</td>
                    <td className="numeric">{formatNumber(player.expected_points)}</td>
                    <td className="numeric">{formatNumber(player.expected_minutes, 1)}</td>
                    <td className="numeric">{formatPercentage(player.p_appearance)}</td>
                    <td className="numeric">{formatPercentage(player.p_start)}</td>
                    <td className="numeric">{formatPercentage(player.prob_points_ge_5)}</td>
                    <td>{formatPlayerStatus(player.status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section aria-labelledby="comparison-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Available challengers</p>
              <h2 id="comparison-heading">Model comparison</h2>
            </div>
          </div>
          <div className="table-scroll compact-table">
            <table>
              <thead>
                <tr>
                  {Object.keys(data.comparison[0] ?? {}).map((column) => (
                    <th scope="col" key={column}>{formatLabel(column)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.comparison.map((row, index) => (
                  <tr key={`${row.left_model}-${row.right_model}-${index}`}>
                    {Object.entries(row).map(([column, value]) => (
                      <td key={column}>{formatComparisonValue(column, value)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="methodology" aria-labelledby="methodology-heading">
          <p className="eyebrow">Read before using</p>
          <h2 id="methodology-heading">Methodology and limitations</h2>
          <p>
            xPoints are model estimates, not guarantees. Expected minutes and appearance
            probabilities directly affect player projections. The displayed squad is selected by
            an exact weekly MILP seed and a deterministic expected-realized search that accounts
            for ordinary automatic substitutions and captain fallback.
          </p>
          <p>
            Current official FPL data are joined to historical model features when available. This
            project is experimental and the displayed recommendations should be interpreted accordingly.
          </p>
          <p>
            Unofficial project. Not affiliated with, endorsed by, or associated with the Premier
            League or Fantasy Premier League.
          </p>
          <a href={repositoryUrl}>Read the repository documentation</a>
        </section>
      </main>

      <SiteFooter />
    </>
  );
}

function SquadTable({ title, players }: { title: string; players: FrontendData["squad"] }) {
  return (
    <section className="squad-table-group" aria-labelledby={`squad-${title.toLowerCase().replace(" ", "-")}`}>
      <h3 id={`squad-${title.toLowerCase().replace(" ", "-")}`}>{title}</h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Player</th>
              <th scope="col">Position</th>
              <th scope="col">Team</th>
              <th scope="col">Opponent</th>
              <th scope="col" className="numeric">Price</th>
              <th scope="col" className="numeric">Expected points</th>
              <th scope="col">Selected role</th>
            </tr>
          </thead>
          <tbody>
            {players.map((player) => (
              <tr className={`role-${player.selected_role}`} key={player.player_uid}>
                <th scope="row">{player.player_name}</th>
                <td>{player.fpl_position}</td>
                <td>{formatTeam(player.player_team_uid)}</td>
                <td>
                  <span aria-label={`Opponent fixture: ${player.opponent_display || "No fixture"}`}>
                    {player.opponent_display || "No fixture"}
                  </span>
                </td>
                <td className="numeric">{formatPrice(player.price_tenths)}</td>
                <td className="numeric">{formatNumber(player.expected_points)}</td>
                <td>
                  <span className="role-label">{squadRoleLabel(player)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SiteFooter() {
  return (
    <footer>
      <div className="page-shell footer-inner">
        <span>
          Built by Daniel Mehta. Unofficial project; no Premier League or Fantasy Premier League
          affiliation.
        </span>
        <a
          className="github-link"
          href={repositoryUrl}
          aria-label="View FPL Forecast on GitHub"
          title="View repository on GitHub"
        >
          <svg aria-hidden="true" viewBox="0 0 24 24">
            <path
              fill="currentColor"
              d="M12 2C6.48 2 2 6.59 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49v-1.91c-2.78.62-3.37-1.21-3.37-1.21-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .08 1.53 1.06 1.53 1.06.89 1.57 2.34 1.12 2.91.86.09-.67.35-1.12.64-1.38-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.04 1.03-2.76-.1-.26-.45-1.31.1-2.72 0 0 .84-.28 2.75 1.05A9.3 9.3 0 0 1 12 6.99a9.3 9.3 0 0 1 2.5.34c1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.46.1 2.72.64.72 1.03 1.64 1.03 2.76 0 3.94-2.34 4.81-4.57 5.07.36.32.68.94.68 1.9v2.76c0 .27.18.59.69.49A10.26 10.26 0 0 0 22 12.25C22 6.59 17.52 2 12 2Z"
            />
          </svg>
        </a>
      </div>
    </footer>
  );
}

export default App;
