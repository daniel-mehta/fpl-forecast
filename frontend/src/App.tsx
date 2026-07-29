import { useEffect, useMemo, useRef, useState } from "react";
import {
  type FrontendData,
  type JsonRecord,
  type ProjectionRow,
  loadFrontendData,
} from "./data";
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
import {
  findPlayers,
  type OfficialPosition,
  validForecastPlayers,
} from "./playerFinder";

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

        <PlayerFinder
          projections={data.projections}
          gameweek={gameweek}
          generatedAt={data.freshness.generated_at}
          runId={textValue(data.status, "run_id") || textValue(data.manifest, "run_id")}
          officialForecast={officialSuccess}
        />

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

const positionLabels: Record<OfficialPosition, string> = {
  GKP: "Goalkeeper",
  DEF: "Defender",
  MID: "Midfielder",
  FWD: "Forward",
};

function formatDifference(value: number | null): string {
  if (value === null) {
    return "N/A";
  }
  if (Math.abs(value) < 0.005) {
    return "±0.00";
  }
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}

function PlayerFinder({
  projections,
  gameweek,
  generatedAt,
  runId,
  officialForecast,
}: {
  projections: ProjectionRow[];
  gameweek: string;
  generatedAt: unknown;
  runId: string;
  officialForecast: boolean;
}) {
  const [finderPosition, setFinderPosition] = useState<OfficialPosition>("MID");
  const [maximumBudget, setMaximumBudget] = useState("15.0");
  const [replacedPlayerId, setReplacedPlayerId] = useState("");
  const [replacementSearch, setReplacementSearch] = useState("");
  const [replacementMenuOpen, setReplacementMenuOpen] = useState(false);
  const replacementSearchRef = useRef<HTMLInputElement>(null);
  const selectablePlayers = useMemo(() => validForecastPlayers(projections), [projections]);
  const replacementOptions = useMemo(() => {
    const query = replacementSearch.trim().toLowerCase();
    if (!query) {
      return selectablePlayers;
    }
    return selectablePlayers.filter(
      (player) =>
        player.stable_player_id === replacedPlayerId ||
        `${player.player} ${formatTeam(player.team)} ${player.position}`
          .toLowerCase()
          .includes(query),
    );
  }, [replacedPlayerId, replacementSearch, selectablePlayers]);
  const result = useMemo(
    () => findPlayers(projections, finderPosition, maximumBudget, replacedPlayerId || undefined),
    [finderPosition, maximumBudget, projections, replacedPlayerId],
  );

  useEffect(() => {
    if (replacementMenuOpen) {
      replacementSearchRef.current?.focus();
    }
  }, [replacementMenuOpen]);

  function selectReplacedPlayer(playerId: string) {
    setReplacedPlayerId(playerId);
    const player = selectablePlayers.find((row) => row.stable_player_id === playerId);
    if (player) {
      setFinderPosition(player.position as OfficialPosition);
      setMaximumBudget((Number(player.price_tenths) / 10).toFixed(1));
    }
    setReplacementSearch("");
    setReplacementMenuOpen(false);
  }

  const count = result.recommendations.length;
  const selectedReplacement = result.replacedPlayer;

  return (
    <section className="player-finder" aria-labelledby="player-finder-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Current official forecast</p>
          <h2 id="player-finder-heading">Player Finder</h2>
        </div>
        <p className="section-note">
          Find up to five strong projected options without entering a complete squad.
        </p>
      </div>

      {!officialForecast ? (
        <div className="finder-empty" role="status">
          Player Finder is unavailable until a current official forecast has been published.
        </div>
      ) : (
        <>
          <div className="finder-controls">
            <div className="finder-control-group">
              <span className="finder-information-label">
                <label htmlFor="finder-position">Position</label>{" "}
                <InfoTooltip label="Position">
                  Filters recommendations to one official FPL position. When replacing a player,
                  their position is used automatically.
                </InfoTooltip>
              </span>
              <select
                id="finder-position"
                value={finderPosition}
                disabled={Boolean(replacedPlayerId)}
                aria-describedby={replacedPlayerId ? "replacement-position-note" : undefined}
                onChange={(event) => setFinderPosition(event.target.value as OfficialPosition)}
              >
                <option value="GKP">Goalkeeper</option>
                <option value="DEF">Defender</option>
                <option value="MID">Midfielder</option>
                <option value="FWD">Forward</option>
              </select>
            </div>
            <div className="finder-control-group">
              <span className="finder-information-label">
                <label htmlFor="finder-maximum-budget">Maximum budget (£m)</label>{" "}
                <InfoTooltip label="Maximum budget (£m)">
                  The highest official player price to include. Selecting a player fills their
                  price, and you can edit it.
                </InfoTooltip>
              </span>
              <input
                id="finder-maximum-budget"
                type="text"
                inputMode="decimal"
                value={maximumBudget}
                aria-invalid={Boolean(result.error)}
                aria-describedby={result.error ? "finder-budget-error" : "finder-budget-help"}
                onChange={(event) => setMaximumBudget(event.target.value)}
                placeholder="7.5"
              />
              <span className="field-help" id="finder-budget-help">
                Official player price in millions
              </span>
            </div>
            <div className="finder-replacement-controls">
              <span className="finder-control-label">
                <span id="replacement-player-label">Optional player being replaced</span>{" "}
                <InfoTooltip label="Optional player being replaced">
                  Choose a player from the current official forecast to compare same-position
                  replacements and expected-points differences.
                </InfoTooltip>
              </span>
              <div className="replacement-combobox">
                <button
                  className="replacement-combobox-trigger"
                  type="button"
                  aria-expanded={replacementMenuOpen}
                  aria-haspopup="listbox"
                  aria-labelledby="replacement-player-label replacement-player-value"
                  onClick={() => setReplacementMenuOpen((open) => !open)}
                >
                  <span id="replacement-player-value">
                    {selectedReplacement
                      ? `${selectedReplacement.player} — ${formatTeam(selectedReplacement.team)}, ${formatPrice(selectedReplacement.price_tenths)}`
                      : "No player selected"}
                  </span>
                  <span aria-hidden="true">▾</span>
                </button>
                {replacementMenuOpen && (
                  <div
                    className="replacement-combobox-menu"
                    onKeyDown={(event) => {
                      if (event.key === "Escape") {
                        setReplacementMenuOpen(false);
                      }
                    }}
                  >
                    <label>
                      Search players
                      <input
                        ref={replacementSearchRef}
                        type="search"
                        value={replacementSearch}
                        onChange={(event) => setReplacementSearch(event.target.value)}
                        placeholder="Player, club or position"
                      />
                    </label>
                    <div
                      className="replacement-options"
                      role="listbox"
                      aria-label="Replacement players"
                    >
                      <button
                        type="button"
                        role="option"
                        aria-selected={!replacedPlayerId}
                        onClick={() => selectReplacedPlayer("")}
                      >
                        No player selected
                      </button>
                      {replacementOptions.map((player) => (
                        <button
                          type="button"
                          role="option"
                          aria-selected={player.stable_player_id === replacedPlayerId}
                          key={player.stable_player_id}
                          onClick={() => selectReplacedPlayer(player.stable_player_id)}
                        >
                          <strong>{player.player}</strong>
                          <span>
                            {positionLabels[player.position as OfficialPosition]} ·{" "}
                            {formatTeam(player.team)} · {formatPrice(player.price_tenths)}
                          </span>
                        </button>
                      ))}
                      {replacementOptions.length === 0 && (
                        <p className="replacement-no-results" role="status">
                          No players match the search.
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {replacedPlayerId && result.replacedPlayer && (
            <p className="finder-context" id="replacement-position-note">
              Replacing <strong>{result.replacedPlayer.player}</strong>. Recommendations are limited
              to {positionLabels[result.position!].toLowerCase()}s from the same forecast.
            </p>
          )}

          <dl className="finder-trace" aria-label="Player Finder forecast identity">
            <div>
              <dt>
                <InformationLabel label="Forecast" className="finder-information-label">
                  The FPL gameweek these recommendations project.
                </InformationLabel>
              </dt>
              <dd>{gameweek ? `Gameweek ${gameweek}` : "Gameweek not available"}</dd>
            </div>
            <div>
              <dt>
                <InformationLabel label="Generated" className="finder-information-label">
                  When the current official forecast was generated, shown in UTC.
                </InformationLabel>
              </dt>
              <dd>{formatDate(generatedAt)}</dd>
            </div>
            <div>
              <dt>
                <InformationLabel label="Run ID" className="finder-information-label">
                  The traceable identifier for the exact official forecast run used.
                </InformationLabel>
              </dt>
              <dd>{runId || "Not available"}</dd>
            </div>
          </dl>
          <p className="finder-source-note">
            Recommendations use only the current official forecast shown on this dashboard.
          </p>

          {result.error ? (
            <p className="field-error" id="finder-budget-error" role="alert">
              {result.error}
            </p>
          ) : count === 0 ? (
            <div className="finder-empty" role="status">
              No eligible players matched this position and budget.
            </div>
          ) : (
            <>
              <div className="finder-results-heading">
                <p>
                  <strong>
                    {count} {count === 1 ? "recommendation" : "recommendations"}
                  </strong>
                </p>
                {count < 5 && (
                  <p>Fewer than five eligible players matched the selected filters.</p>
                )}
                {!result.replacedPlayer && (
                  <p>Expected-points difference is N/A until a player is selected.</p>
                )}
              </div>
              <div className="recommendation-grid" aria-label="Player Finder recommendations">
                {result.recommendations.map((recommendation, index) => {
                  const difference = recommendation.expectedPointsDifference;
                  const differenceClass =
                    difference === null || Math.abs(difference) < 0.005
                      ? "difference-neutral"
                      : difference > 0
                        ? "difference-positive"
                        : "difference-negative";
                  return (
                    <article
                      className="recommendation-card"
                      key={recommendation.player.stable_player_id}
                    >
                      <div className="recommendation-title">
                        <span className="recommendation-rank" aria-label={`Rank ${index + 1}`}>
                          {index + 1}
                        </span>
                        <div>
                          <h3>{recommendation.player.player}</h3>
                          <p>
                            {formatTeam(recommendation.player.team)} ·{" "}
                            {positionLabels[recommendation.player.position as OfficialPosition]}
                          </p>
                        </div>
                      </div>
                      <dl>
                        <div>
                          <dt>
                            <span className="finder-metric-label">
                              <InfoTooltip label="Official price">
                                The player's current official FPL price from this forecast artifact.
                              </InfoTooltip>
                              <span>Official price</span>
                            </span>
                          </dt>
                          <dd>{formatPrice(recommendation.player.price_tenths)}</dd>
                        </div>
                        <div>
                          <dt>
                            <span className="finder-metric-label">
                              <InfoTooltip label="Expected points">
                                The model's mean projected FPL points for this gameweek, not a
                                guaranteed result.
                              </InfoTooltip>
                              <span>Expected points</span>
                            </span>
                          </dt>
                          <dd>{formatNumber(recommendation.player.expected_points)}</dd>
                        </div>
                        <div>
                          <dt>
                            <span className="finder-metric-label">
                              <InfoTooltip label="Appearance">
                                The forecast probability that the player appears for any minutes.
                              </InfoTooltip>
                              <span>Appearance</span>
                            </span>
                          </dt>
                          <dd>{formatPercentage(recommendation.player.p_appearance)}</dd>
                        </div>
                        <div>
                          <dt>
                            <span className="finder-metric-label">
                              <InfoTooltip label="P(5+ points)">
                                The forecast probability that the player scores at least five FPL
                                points.
                              </InfoTooltip>
                              <span>P(5+ points)</span>
                            </span>
                          </dt>
                          <dd>{formatPercentage(recommendation.player.prob_points_ge_5)}</dd>
                        </div>
                        <div>
                          <dt>
                            <span className="finder-metric-label">
                              <InfoTooltip label="Expected points / £1.0m">
                                Expected points divided by the player's official price in millions,
                                calculated before display rounding.
                              </InfoTooltip>
                              <span>Expected points / £1.0m</span>
                            </span>
                          </dt>
                          <dd>{recommendation.expectedPointsPerMillion.toFixed(2)}</dd>
                        </div>
                        <div>
                          <dt>
                            <span className="finder-metric-label">
                              <InfoTooltip label="Expected-points difference">
                                This player's expected points minus the selected player's expected
                                points. It is unavailable when no player is selected.
                              </InfoTooltip>
                              <span>Expected-points difference</span>
                            </span>
                          </dt>
                          <dd className={differenceClass}>{formatDifference(difference)}</dd>
                        </div>
                      </dl>
                    </article>
                  );
                })}
              </div>
              <p className="finder-disclaimer">
                Projections are estimates, not guarantees. This finder does not account for complete
                squad legality, selling prices, bank balance, free transfers, hits or multi-gameweek
                planning.
              </p>
            </>
          )}
        </>
      )}
    </section>
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
