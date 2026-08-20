import { useEffect, useMemo, useState } from "react";
import type { FrontendData, JsonRecord } from "./data";
import {
  optimizerPlayersFromProjections,
  optimizeFixedSquad,
  POSITION_QUOTAS,
  type FixedSquadResult,
  type FplPosition,
  type OptimizerPlayer,
  validateSquad,
} from "./yourTeamOptimizer";
import {
  changedRoleSummary,
  recommendIndependentTransfers,
  recommendTransfers,
  type IndependentTransferRecommendationResult,
  type TransferRecommendationResult,
} from "./yourTeamTransfers";
import {
  resetYourTeam,
  restoreYourTeam,
  saveYourTeam,
  type ForecastIdentity,
} from "./yourTeamStorage";
import { formatDate, formatNumber, formatPercentage, formatPrice, formatTeam } from "./formatting";
import { InfoTooltip, InformationLabel } from "./InfoTooltip";

interface YourTeamPageProps {
  data: FrontendData;
}

interface CalculationBase {
  baseline: FixedSquadResult;
  elapsedMilliseconds: number;
}

export interface IndependentCalculation extends CalculationBase {
  mode: "independent";
  transfers: IndependentTransferRecommendationResult;
}

export interface CombinedCalculation extends CalculationBase {
  mode: "combined";
  transfers: TransferRecommendationResult;
}

export type Calculation = IndependentCalculation | CombinedCalculation;

const positions = Object.keys(POSITION_QUOTAS) as FplPosition[];

export function YourTeamPage({ data }: YourTeamPageProps) {
  const contract = useMemo(() => validateContract(data), [data]);
  const pool = contract.players;
  const identity = contract.identity;
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [sellingPrices, setSellingPrices] = useState<Record<string, number>>({});
  const [bankTenths, setBankTenths] = useState(0);
  const [freeTransfers, setFreeTransfers] = useState(1);
  const [combineRecommendations, setCombineRecommendations] = useState(false);
  const [search, setSearch] = useState("");
  const [message, setMessage] = useState("");
  const [calculation, setCalculation] = useState<Calculation | null>(null);
  const [calculating, setCalculating] = useState(false);

  useEffect(() => {
    if (!identity || !contract.valid) return;
    const storage = browserStorage();
    if (!storage) return;
    const saved = restoreYourTeam(storage, identity);
    if (saved) {
      setSelectedIds(saved.playerIds);
      setSellingPrices(saved.sellingPrices);
      setBankTenths(saved.bankTenths);
      setFreeTransfers(saved.freeTransfers);
      setCombineRecommendations(saved.combineRecommendations === true);
    }
  }, [contract.valid, identity]);

  useEffect(() => {
    if (!identity || !contract.valid) return;
    const storage = browserStorage();
    if (!storage) return;
    if (
      selectedIds.length === 0 &&
      Object.keys(sellingPrices).length === 0 &&
      bankTenths === 0 &&
      freeTransfers === 1 &&
      !combineRecommendations
    ) {
      resetYourTeam(storage);
      return;
    }
    saveYourTeam(storage, {
      schemaVersion: 1,
      forecastIdentity: identity,
      playerIds: selectedIds,
      sellingPrices,
      bankTenths,
      freeTransfers,
      combineRecommendations,
    });
  }, [bankTenths, combineRecommendations, contract.valid, freeTransfers, identity, selectedIds, sellingPrices]);

  const selected = selectedIds
    .map((id) => pool.find((player) => player.id === id))
    .filter((player): player is OptimizerPlayer => Boolean(player));
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const squadErrors = validateSquad(selected);
  const roleById = calculation ? optimizedRoles(calculation.baseline) : new Map<string, string>();
  const searchResults = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return [];
    return pool
      .filter(
        (player) =>
          !selectedSet.has(player.id) &&
          `${player.name} ${formatTeam(player.team)} ${player.position}`.toLowerCase().includes(query),
      )
      .slice(0, 10);
  }, [pool, search, selectedSet]);

  function invalidateCalculation() {
    setCalculation(null);
    setMessage("");
  }

  function addPlayer(player: OptimizerPlayer) {
    if (selectedSet.has(player.id)) return setMessage("That player is already selected.");
    if (selected.filter((item) => item.position === player.position).length >= POSITION_QUOTAS[player.position]) {
      return setMessage(`The squad already has ${POSITION_QUOTAS[player.position]} ${player.position} players.`);
    }
    if (selected.filter((item) => item.team === player.team).length >= 3) {
      return setMessage("A squad may contain no more than three players from one club.");
    }
    setSelectedIds((current) => [...current, player.id]);
    setSellingPrices((current) => ({ ...current, [player.id]: player.priceTenths }));
    setSearch("");
    invalidateCalculation();
  }

  function removePlayer(player: OptimizerPlayer) {
    setSelectedIds((current) => current.filter((id) => id !== player.id));
    setSellingPrices((current) => {
      const next = { ...current };
      delete next[player.id];
      return next;
    });
    invalidateCalculation();
  }

  async function calculate() {
    const inputErrors = [
      ...squadErrors,
      ...validateMoneyAndTransfers(selected, sellingPrices, bankTenths, freeTransfers),
    ];
    if (inputErrors.length) {
      setMessage(inputErrors.join(" "));
      return;
    }
    setMessage("");
    setCalculation(null);
    setCalculating(true);
    await allowCalculatingStateToPaint();
    const started = performance.now();
    try {
      const baseline = optimizeFixedSquad(selected);
      const args = {
        squad: selected,
        pool,
        sellingPrices,
        bankTenths,
        freeTransfers,
        baseline,
      };
      if (combineRecommendations) {
        setCalculation({
          mode: "combined",
          baseline,
          transfers: recommendTransfers(args),
          elapsedMilliseconds: performance.now() - started,
        });
      } else {
        setCalculation({
          mode: "independent",
          baseline,
          transfers: recommendIndependentTransfers(args),
          elapsedMilliseconds: performance.now() - started,
        });
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown calculation error.";
      setMessage(`Calculation failed: ${detail} No recommendation was produced.`);
    } finally {
      setCalculating(false);
    }
  }

  function reset() {
    const storage = browserStorage();
    if (storage) resetYourTeam(storage);
    setSelectedIds([]);
    setSellingPrices({});
    setBankTenths(0);
    setFreeTransfers(1);
    setCombineRecommendations(false);
    setSearch("");
    setCalculation(null);
    setMessage("Your saved squad and inputs were cleared from this browser.");
  }

  if (!contract.valid || !identity) {
    return (
      <main className="page-shell content your-team-content">
        <section className="load-state" role="alert">
          <p className="eyebrow">Unavailable</p>
          <h2>Your Team is not available</h2>
          <p>{contract.error}</p>
          <p>No lineup or transfer recommendation has been calculated.</p>
        </section>
      </main>
    );
  }

  const squadCost = selected.reduce((sum, player) => sum + player.priceTenths, 0);
  return (
    <main className="page-shell content your-team-content">
      <section aria-labelledby="your-team-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Private browser tool</p>
            <h2 id="your-team-heading">Your Team</h2>
          </div>
          <button type="button" className="secondary-action" onClick={reset}>Reset Your Team</button>
        </div>
        <p className="your-team-intro">
          Enter your 15-player squad manually. Your squad, selling prices, bank and free transfers
          stay in this browser&apos;s localStorage and are never sent anywhere.
        </p>
        <dl className="finder-trace" aria-label="Your Team forecast identity">
          <div><dt>Forecast</dt><dd>{identity.season}, Gameweek {identity.gameweek}</dd></div>
          <div><dt>Generated</dt><dd>{formatDate(data.freshness.generated_at)}</dd></div>
          <div><dt>Run ID</dt><dd>{identity.runId}</dd></div>
        </dl>
      </section>

      <section className="team-entry" aria-labelledby="team-entry-heading" aria-busy={calculating}>
        <div className="section-heading">
          <div><p className="eyebrow">15-player squad</p><h2 id="team-entry-heading">Team entry</h2></div>
          <p className="section-note">Squad cost {formatPrice(squadCost)} · {selected.length}/15 selected</p>
        </div>
        <div className="team-entry-controls">
          <label>
            Search current forecast players
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Player, club or position"
            />
          </label>
          <div className="team-entry-field">
            <span className="control-label"><label htmlFor="your-team-bank">Money in the bank (£m)</label> <InfoTooltip label="Money in the bank (£m)">{HELP.bank}</InfoTooltip></span>
            <input
              id="your-team-bank"
              type="number"
              min="0"
              step="0.1"
              value={(bankTenths / 10).toFixed(1)}
              onChange={(event) => { setBankTenths(Math.round(Number(event.target.value) * 10)); invalidateCalculation(); }}
            />
          </div>
          <div className="team-entry-field">
            <span className="control-label"><label htmlFor="your-team-free-transfers">Free transfers</label> <InfoTooltip label="Free transfers">{combineRecommendations ? HELP.freeTransfersCombined : HELP.freeTransfersIndependent}</InfoTooltip></span>
            <input
              id="your-team-free-transfers"
              type="number"
              min="0"
              max="5"
              step="1"
              value={freeTransfers}
              onChange={(event) => { setFreeTransfers(Number(event.target.value)); invalidateCalculation(); }}
            />
          </div>
        </div>
        <div className="transfer-mode-control">
          <label>
            <input
              type="checkbox"
              checked={combineRecommendations}
              disabled={calculating}
              aria-describedby="transfer-mode-description"
              onChange={(event) => {
                setCombineRecommendations(event.target.checked);
                invalidateCalculation();
              }}
            />
            <span>Combine recommendations into one plan</span>
          </label>
          <InfoTooltip label="Combine recommendations into one plan">
            {HELP.combineRecommendations}
          </InfoTooltip>
          <p id="transfer-mode-description">
            {combineRecommendations
              ? "Combined plan: transfers are optimized together, so one move may fund or enable another."
              : "Independent suggestions: every option is one separate transfer from your current squad."}
          </p>
        </div>
        {search && (
          <div className="player-search-results" role="list" aria-label="Player search results">
            {searchResults.map((player) => (
              <button
                type="button"
                role="listitem"
                aria-label={`Add ${player.name}, ${player.position}, ${formatTeam(player.team)}`}
                key={player.id}
                onClick={() => addPlayer(player)}
              >
                <strong>{player.name}</strong>
                <span>{formatTeam(player.team)} · {player.position} · {formatPrice(player.priceTenths)}</span>
              </button>
            ))}
            {searchResults.length === 0 && <p role="status">No available players match.</p>}
          </div>
        )}
        <div className="quota-strip" aria-label="Position quota progress">
          {positions.map((position) => (
            <span key={position}>{position} {selected.filter((player) => player.position === position).length}/{POSITION_QUOTAS[position]}</span>
          ))}
        </div>
        {selected.length > 0 && squadErrors.length > 0 && (
          <p className="field-error" role="status">{squadErrors.join(" ")}</p>
        )}
        {message && <p className="field-error" role="alert">{message}</p>}
        {selected.length > 0 && (
          <PlayerTable
            players={selected}
            sellingPrices={sellingPrices}
            roleById={roleById}
            onSellingPrice={(player, value) => {
              setSellingPrices((current) => ({ ...current, [player.id]: Math.round(value * 10) }));
              invalidateCalculation();
            }}
            onRemove={removePlayer}
          />
        )}
        <p className="field-help selling-price-help">
          Selling price defaults to the current official price. Correct it if FPL shows a different
          sale value; affordability uses your entered selling price plus bank.
        </p>
        <button type="button" className="primary-action" onClick={calculate} disabled={squadErrors.length > 0 || calculating}>
          {calculating ? "Calculating…" : "Optimize lineup and transfers"}
        </button>
        {calculating && (
          <p className="field-help" role="status" aria-live="polite">
            {combineRecommendations
              ? "Calculating the exact lineup and bounded combined plan…"
              : "Calculating exact independent single-transfer suggestions…"}
          </p>
        )}
      </section>

      {calculation && (
        <>
          <LineupResult result={calculation.baseline} pool={pool} />
          {calculation.mode === "combined"
            ? <TransferResults calculation={calculation} />
            : <IndependentTransferResults calculation={calculation} />}
        </>
      )}

      <section className="methodology" aria-labelledby="your-team-methodology">
        <p className="eyebrow">Scope and limitations</p>
        <h2 id="your-team-methodology">One frozen Gameweek only</h2>
        <p>
          The browser applies the Python-authoritative D2 fixed-squad decision method to this exact
          frozen forecast. It evaluates ordinary autosubs and captain fallback across 32,768
          independent appearance states. Transfer plans may use up to the entered number of free
          transfers. Independent suggestions are separate one-transfer comparisons; combined plans
          may connect several moves. Both cover this Gameweek only. It does not regenerate forecasts, plan future
          Gameweeks, or model chips, wildcards or Free Hit.
        </p>
        <p>Projections and recommendations are estimates, not guarantees.</p>
      </section>
    </main>
  );
}

function PlayerTable({
  players,
  sellingPrices,
  roleById,
  onSellingPrice,
  onRemove,
}: {
  players: OptimizerPlayer[];
  sellingPrices: Record<string, number>;
  roleById: Map<string, string>;
  onSellingPrice: (player: OptimizerPlayer, value: number) => void;
  onRemove: (player: OptimizerPlayer) => void;
}) {
  return (
    <div className="table-scroll your-team-table-scroll">
      <table className="your-team-table">
        <thead><tr>
          <th>Player</th><th>Club</th><th>Pos</th><th>Opponent</th><th>Official</th>
          <th><InformationLabel label="Selling price">{HELP.sellingPrice}</InformationLabel></th>
          <th><InformationLabel label="Expected points">{HELP.expectedPoints}</InformationLabel></th>
          <th><InformationLabel label="Expected minutes">{HELP.expectedMinutes}</InformationLabel></th>
          <th><InformationLabel label="Appearance probability">{HELP.appearanceProbability}</InformationLabel></th>
          <th><InformationLabel label="Start probability">{HELP.startProbability}</InformationLabel></th>
          <th><InformationLabel label="At least five points probability">{HELP.fivePointsProbability}</InformationLabel></th>
          <th><InformationLabel label="Optimized role">{HELP.optimizedRole}</InformationLabel></th>
          <th><span className="visually-hidden">Action</span></th>
        </tr></thead>
        <tbody>{players.map((player) => {
          const row = player.projection;
          return (
            <tr key={player.id}>
              <th scope="row" data-label="Player">{player.name}</th>
              <td data-label="Club">{formatTeam(player.team)}</td>
              <td data-label="Position">{player.position}</td>
              <td data-label="Opponent">{row.opponent_display || "No fixture"}</td>
              <td className="numeric" data-label="Official price">{formatPrice(player.priceTenths)}</td>
              <td data-label="Selling price"><MobileInfo label="Selling price">{HELP.sellingPrice}</MobileInfo><input aria-label={`${player.name} selling price`} type="number" min="0" step="0.1" value={((sellingPrices[player.id] ?? player.priceTenths) / 10).toFixed(1)} onChange={(event) => onSellingPrice(player, Number(event.target.value))} /></td>
              <td className="numeric" data-label="Expected points"><MobileInfo label="Expected points">{HELP.expectedPoints}</MobileInfo>{formatNumber(row.expected_points)}</td>
              <td className="numeric" data-label="Expected minutes"><MobileInfo label="Expected minutes">{HELP.expectedMinutes}</MobileInfo>{formatNumber(row.expected_minutes, 1)}</td>
              <td className="numeric" data-label="Appearance"><MobileInfo label="Appearance probability">{HELP.appearanceProbability}</MobileInfo>{formatPercentage(row.p_appearance)}</td>
              <td className="numeric" data-label="Start"><MobileInfo label="Start probability">{HELP.startProbability}</MobileInfo>{formatPercentage(row.p_start)}</td>
              <td className="numeric" data-label="At least five points"><MobileInfo label="At least five points probability">{HELP.fivePointsProbability}</MobileInfo>{formatPercentage(row.prob_points_ge_5)}</td>
              <td data-label="Optimized role"><MobileInfo label="Optimized role">{HELP.optimizedRole}</MobileInfo><span className="role-label">{roleById.get(player.id) ?? "Not optimized"}</span></td>
              <td data-label="Action"><button type="button" onClick={() => onRemove(player)}>Remove</button></td>
            </tr>
          );
        })}</tbody>
      </table>
    </div>
  );
}

function MobileInfo({ label, children }: { label: string; children: string }) {
  return <span className="mobile-metric-help"><InfoTooltip label={label}>{children}</InfoTooltip></span>;
}

function LineupResult({ result, pool }: { result: FixedSquadResult; pool: OptimizerPlayer[] }) {
  const name = (id: string) => pool.find((player) => player.id === id)?.name ?? id;
  const breakdown = result.breakdown;
  return (
    <section aria-labelledby="optimized-lineup-heading">
      <div className="section-heading"><div><p className="eyebrow">Exact expected-realized evaluation <InfoTooltip label="Exact D2 evaluation">{HELP.exactD2}</InfoTooltip></p><h2 id="optimized-lineup-heading">Optimized lineup</h2></div><p className="section-note">Formation {result.decision.formation}</p></div>
      <dl className="optimizer-summary">
        <Metric label="Nominal starting XI" help={HELP.nominalXi} value={breakdown.nominalStartingXiExpectedPoints} />
        <Metric label="Active starters" help={HELP.activeStarters} value={breakdown.expectedActiveStarterPoints} />
        <Metric label="Autosub contribution" help={HELP.autosubContribution} value={breakdown.expectedAutosubContribution} />
        <Metric label="Captain bonus" help={HELP.captainBonus} value={breakdown.expectedCaptainBonus} />
        <Metric label="Vice contingency" help={HELP.viceContingency} value={breakdown.expectedViceCaptainContingency} />
        <Metric label="Expected realized" help={HELP.expectedRealized} value={breakdown.expectedRealizedTotal} />
        <Metric label="Expected autosubs" help={HELP.expectedAutosubs} value={breakdown.expectedAutomaticSubstitutions} />
        <div><dt><InformationLabel label="Unreplaced risk">{HELP.unreplacedRisk}</InformationLabel></dt><dd>{formatPercentage(breakdown.probabilityUnreplacedStarter)}</dd></div>
      </dl>
      <div className="lineup-summary-cards">
        <article><h3>Starting XI</h3><p>{result.decision.lineup.map(name).join(", ")}</p></article>
        <article><h3>Captaincy</h3><p>Captain: {name(result.decision.captain)} · Vice-captain: {name(result.decision.viceCaptain)}</p></article>
        <article><h3>Bench</h3><p>Goalkeeper: {name(result.decision.bench[0])} · Outfield order: {result.decision.bench.slice(1).map(name).join(", ")}</p></article>
      </div>
    </section>
  );
}

export function IndependentTransferResults({ calculation }: { calculation: IndependentCalculation }) {
  const { baseline, transfers } = calculation;
  const groupLimit = transfers.freeTransfersAvailable === 0 ? 1 : transfers.freeTransfersAvailable;
  const searchDescription = transfers.shortlisted
    ? `${transfers.exactPlanEvaluations} exact D2 single-transfer evaluations from a deterministic diversity-aware shortlist of ${transfers.candidatePlanCount} legal candidates`
    : `${transfers.exactPlanEvaluations} exact D2 single-transfer evaluations across ${transfers.candidatePlanCount} legal candidates`;
  return (
    <section aria-labelledby="independent-transfer-heading">
      <div className="section-heading">
        <div><p className="eyebrow">Independent transfers · current Gameweek <InfoTooltip label="Independent recommendation">{HELP.independentRecommendation}</InfoTooltip></p><h2 id="independent-transfer-heading">Transfer recommendations</h2></div>
        <p className="section-note">{searchDescription} · {(calculation.elapsedMilliseconds / 1000).toFixed(2)}s</p>
      </div>
      <div className="notice" role="note">
        Each recommendation is independent and assumes no other listed transfer is made. Update your squad and recalculate after making a transfer.
      </div>
      <dl className="transfer-plan-summary independent-baseline" aria-label="Independent transfer baseline">
        <TransferMetric label="Baseline expected realized total" help={HELP.expectedRealized} value={formatNumber(transfers.expectedRealizedBefore, 2)} />
      </dl>
      {transfers.recommendNoTransfer ? (
        <div className="notice" role="status">
          <strong>No transfer recommended.</strong>{" "}
          {transfers.freeTransfersAvailable === 0
            ? "No paid single transfer has a positive net improvement after the four-point hit."
            : `Fewer than ${groupLimit} beneficial outgoing-player groups are available; none had a positive exact improvement.`}
        </div>
      ) : (
        <>
          <div className="transfer-groups">
            {transfers.groups.map((group, groupIndex) => (
              <article className="transfer-group-card" key={group.playerOut.id}>
                <div className="recommendation-title">
                  <span className="recommendation-rank">{groupIndex + 1}</span>
                  <div><h3>{group.playerOut.name} out</h3><p>{group.playerOut.position} · selling price {formatPrice(group.outgoingSellingPriceTenths)}</p></div>
                </div>
                <div className="transfer-options">
                  {group.options.map((option, optionIndex) => (
                    <article className="transfer-option-card" key={option.playerIn.id}>
                      <p className="option-rank">{optionIndex === 0 ? "Top retained option" : `Alternative ${optionIndex + 1}`}</p>
                      <h4>{option.playerIn.name}</h4>
                      <p>{formatTeam(option.playerIn.team)} · {formatPrice(option.incomingPriceTenths)}</p>
                      <dl>
                        <TransferMetric label="Resulting total" help={HELP.expectedRealized} value={formatNumber(option.expectedRealizedAfter, 2)} />
                        <TransferMetric label="Gross improvement" help={HELP.grossImprovement} value={signed(option.grossImprovement)} />
                        <TransferMetric label="Points hit" help={HELP.pointsHit} value={`−${option.pointsHit}`} />
                        <TransferMetric label="Net improvement" help={HELP.netImprovement} value={signed(option.netImprovement)} />
                        <TransferMetric label="Bank after transfer" help={HELP.bankAfterTransfer} value={formatPrice(option.bankRemainingTenths)} />
                      </dl>
                      <p className="transfer-changes">{changedRoleSummary(baseline.decision, option.result.decision).join(" · ") || "Formation, lineup roles and bench order unchanged"}</p>
                    </article>
                  ))}
                </div>
              </article>
            ))}
          </div>
          {transfers.groups.length < groupLimit && (
            <div className="notice" role="status">
              Only {transfers.groups.length} outgoing player{transfers.groups.length === 1 ? " has" : "s have"} a positive independent transfer; no negative-value group was added to fill the list.
            </div>
          )}
          {transfers.shortlisted && (
            <div className="notice" role="note">
              Candidates use a deterministic diversity-aware bounded shortlist <InfoTooltip label="Deterministic bounded shortlist">{HELP.boundedShortlist}</InfoTooltip>. Results are exact D2 evaluations of the retained single transfers, not a claim of the global optimum.
            </div>
          )}
        </>
      )}
    </section>
  );
}

export function TransferResults({ calculation }: { calculation: CombinedCalculation }) {
  const { baseline, transfers } = calculation;
  const searchDescription = transfers.shortlisted
    ? `${transfers.exactPlanEvaluations} exact D2 plan evaluations from a deterministic bounded shortlist of ${transfers.candidatePlanCount} legal candidate plans`
    : `${transfers.exactPlanEvaluations} exact D2 plan evaluations across ${transfers.candidatePlanCount} legal candidate plans`;
  return (
    <section aria-labelledby="transfer-heading">
      <div className="section-heading">
        <div><p className="eyebrow">Combined transfers · current Gameweek <InfoTooltip label="Combined plan">{HELP.combinedPlan}</InfoTooltip></p><h2 id="transfer-heading">Transfer recommendations</h2></div>
        <p className="section-note">{searchDescription} · {(calculation.elapsedMilliseconds / 1000).toFixed(2)}s</p>
      </div>
      {transfers.recommendNoTransfer ? (
        <div className="notice" role="status">
          <strong>No transfer recommended.</strong>{" "}
          {transfers.freeTransfersAvailable === 0
            ? "No paid move has a positive net expected-points improvement after the four-point hit."
            : `Use 0 of ${transfers.freeTransfersAvailable} free transfers and roll ${transfers.freeTransfersRolled}.`}
        </div>
      ) : (
        <>
          <div className="transfer-plan-decision" role="status">
            <strong>{transferUsageText(transfers)} <InfoTooltip label="Rolled free transfers">{HELP.rolledTransfers}</InfoTooltip></strong>
            <span>The first replacement in each group forms the primary combined plan <InfoTooltip label="Primary option">{HELP.primaryOption}</InfoTooltip>.</span>
          </div>
          <dl className="transfer-plan-summary" aria-label="Primary transfer plan summary">
            <TransferMetric label="Baseline" help={HELP.expectedRealized} value={formatNumber(transfers.expectedRealizedBefore, 2)} />
            <TransferMetric label="Resulting total" help={HELP.expectedRealized} value={formatNumber(transfers.expectedRealizedAfter, 2)} />
            <TransferMetric label="Gross improvement" help={HELP.grossImprovement} value={signed(transfers.grossImprovement)} />
            <TransferMetric label="Points hit" help={HELP.pointsHit} value={`−${transfers.pointsHit}`} />
            <TransferMetric label="Net improvement" help={HELP.netImprovement} value={signed(transfers.netImprovement)} />
            <TransferMetric label="Bank remaining" help={HELP.bankAfterTransfer} value={formatPrice(transfers.bankRemainingTenths)} />
          </dl>
          <p className="primary-plan-line">
            <strong>Primary plan:</strong>{" "}
            {transfers.primaryTransfers.map((transfer) => `${transfer.playerOut.name} → ${transfer.playerIn.name}`).join(" · ")}
          </p>
          <p className="transfer-changes">
            {transfers.result
              ? changedRoleSummary(baseline.decision, transfers.result.decision).join(" · ") || "Starting XI, captaincy and bench order unchanged"
              : "Starting XI, captaincy and bench order unchanged"}
          </p>
          <div className="transfer-groups">
            {transfers.groups.map((group, groupIndex) => (
              <article className="transfer-group-card" key={group.playerOut.id}>
                <div className="recommendation-title">
                  <span className="recommendation-rank">{groupIndex + 1}</span>
                  <div><h3>{group.playerOut.name} out</h3><p>{group.playerOut.position} · selling price {formatPrice(group.outgoingSellingPriceTenths)}</p></div>
                </div>
                <div className="transfer-options">
                  {group.options.map((option, optionIndex) => (
                    <article className="transfer-option-card" key={option.playerIn.id}>
                      <p className="option-rank">{optionIndex === 0 ? "Primary option" : `Alternative ${optionIndex + 1}`}</p>
                      <h4>{option.playerIn.name}</h4>
                      <p>{formatTeam(option.playerIn.team)} · {formatPrice(option.incomingPriceTenths)}</p>
                      <dl>
                        <TransferMetric label="Plan total" help={HELP.expectedRealized} value={formatNumber(option.expectedRealizedAfter, 2)} />
                        <TransferMetric label="Net improvement" help={HELP.netImprovement} value={signed(option.netImprovement)} />
                        <TransferMetric label="Bank remaining" help={HELP.bankAfterTransfer} value={formatPrice(option.bankRemainingTenths)} />
                      </dl>
                      <p className="transfer-changes">{changedRoleSummary(baseline.decision, option.result.decision).join(" · ") || "Lineup roles unchanged"}</p>
                    </article>
                  ))}
                </div>
              </article>
            ))}
          </div>
          <p className="field-help transfer-dependency-note">
            Alternatives are evaluated with every other primary transfer fixed. Options that would duplicate an incoming player, exceed the combined budget or break a squad rule are omitted.
          </p>
          {transfers.shortlisted && (
            <div className="notice" role="note">
              This is a deterministic bounded-shortlist plan <InfoTooltip label="Deterministic bounded shortlist">{HELP.boundedShortlist}</InfoTooltip>, not a claim of the global multi-transfer optimum.
            </div>
          )}
        </>
      )}
    </section>
  );
}

function transferUsageText(transfers: TransferRecommendationResult): string {
  if (transfers.freeTransfersAvailable === 0) return "Make 1 paid transfer after a four-point hit.";
  if (transfers.freeTransfersRolled === 0) {
    return `Use all ${transfers.transfersUsed} available free transfer${transfers.transfersUsed === 1 ? "" : "s"}.`;
  }
  return `Use ${transfers.transfersUsed} of ${transfers.freeTransfersAvailable} free transfers and roll ${transfers.freeTransfersRolled}.`;
}

function Metric({ label, help, value }: { label: string; help: string; value: number }) {
  return <div><dt><InformationLabel label={label}>{help}</InformationLabel></dt><dd>{formatNumber(value, 2)}</dd></div>;
}

function TransferMetric({ label, help, value }: { label: string; help: string; value: string }) {
  return <div><dt><InformationLabel label={label}>{help}</InformationLabel></dt><dd>{value}</dd></div>;
}

const HELP = {
  bank: "Money left in your FPL budget before any suggested transfer.",
  freeTransfersIndependent: "The maximum number of separate outgoing-player suggestion groups to show. Each suggestion uses one free transfer.",
  freeTransfersCombined: "The maximum number of connected transfers the planner may use. It can recommend fewer and roll the rest.",
  sellingPrice: "The amount FPL gives you when this player is sold. Edit it if it differs from the current market price.",
  combineRecommendations: "Off compares separate single transfers. On builds one connected plan where moves can fund or enable each other.",
  expectedPoints: "Forecast points for the Gameweek, including the chance the player does not appear.",
  expectedMinutes: "Forecast average minutes for the Gameweek, including zero-minute outcomes.",
  appearanceProbability: "Forecast chance of playing any minutes in the Gameweek.",
  startProbability: "Forecast chance of being named in the starting lineup.",
  fivePointsProbability: "Forecast chance of scoring at least five FPL points.",
  optimizedRole: "The role chosen by the exact lineup optimizer after accounting for autosubs and captain fallback.",
  expectedRealized: "Expected Gameweek points after appearance risk, legal autosubs and captain or vice-captain fallback.",
  nominalXi: "Expected points of the named starting XI before autosubs or captaincy are applied.",
  activeStarters: "Expected points contributed by starters who actually appear.",
  autosubContribution: "Expected points added by legal bench replacements for absent starters.",
  captainBonus: "Expected extra points from doubling the captain when the captain appears.",
  viceContingency: "Expected extra points from the vice-captain when the captain does not appear.",
  expectedAutosubs: "Average number of automatic substitutions across all appearance outcomes.",
  unreplacedRisk: "Chance at least one absent starter cannot be legally replaced from the bench.",
  grossImprovement: "Resulting expected realized total minus the unchanged-squad baseline, before any hit.",
  pointsHit: "FPL points deducted for a paid transfer. A zero-free-transfer suggestion includes a four-point hit.",
  netImprovement: "Gross improvement after subtracting any transfer hit.",
  bankAfterTransfer: "Original bank plus the outgoing selling price, minus the incoming price for this suggestion or plan.",
  exactD2: "The fixed-squad decision method evaluates the lineup, bench and captaincy across all 32,768 independent appearance states.",
  boundedShortlist: "A deterministic candidate limit keeps browser calculations practical; every retained plan receives exact D2 evaluation.",
  primaryOption: "This replacement is part of the connected plan. Alternatives keep all other primary moves fixed.",
  independentRecommendation: "A single transfer evaluated from your unchanged squad, without sharing money or club slots with other suggestions.",
  combinedPlan: "A connected set of transfers evaluated together, allowing one move to fund or enable another.",
  rolledTransfers: "Available free transfers the plan recommends not using this Gameweek.",
} as const;

function optimizedRoles(result: FixedSquadResult): Map<string, string> {
  const roles = new Map<string, string>();
  result.decision.lineup.forEach((id) => roles.set(id, "Starter"));
  roles.set(result.decision.captain, "Captain");
  roles.set(result.decision.viceCaptain, "Vice-captain");
  roles.set(result.decision.bench[0], "Bench goalkeeper");
  result.decision.bench.slice(1).forEach((id, index) => roles.set(id, `Bench ${index + 1}`));
  return roles;
}

function validateMoneyAndTransfers(
  players: OptimizerPlayer[],
  sellingPrices: Record<string, number>,
  bankTenths: number,
  freeTransfers: number,
): string[] {
  const errors: string[] = [];
  if (!Number.isInteger(bankTenths) || bankTenths < 0) errors.push("Bank must be a non-negative amount in £0.1m increments.");
  if (!Number.isInteger(freeTransfers) || freeTransfers < 0 || freeTransfers > 5) errors.push("Free transfers must be a whole number from 0 to 5.");
  for (const player of players) if (!Number.isInteger(sellingPrices[player.id]) || sellingPrices[player.id] <= 0) errors.push(`Enter a valid selling price for ${player.name}.`);
  return errors;
}

function validateContract(data: FrontendData): { valid: boolean; error: string; players: OptimizerPlayer[]; identity: ForecastIdentity | null } {
  const text = (record: JsonRecord, key: string) => {
    const value = record[key];
    return typeof value === "string" || typeof value === "number" ? String(value) : "";
  };
  const season = text(data.status, "target_season");
  const gameweek = text(data.status, "target_gameweek");
  const runId = text(data.status, "run_id");
  const manifestSeason = text(data.manifest, "target_season");
  const manifestGameweek = text(data.manifest, "target_gameweek");
  const manifestRunId = text(data.manifest, "run_id");
  if (
    text(data.status, "state") !== "SUCCEEDED" ||
    text(data.freshness, "source_mode") !== "official_current_season" ||
    data.freshness.stale !== false ||
    !season || !gameweek || !runId ||
    season !== manifestSeason || gameweek !== manifestGameweek || runId !== manifestRunId ||
    !data.freshness.generated_at
  ) return { valid: false, error: "The frozen official forecast is absent, stale or has inconsistent run identity.", players: [], identity: null };
  try {
    if (data.projections.some((row) => row.season !== season || String(row.gameweek) !== gameweek)) throw new Error("Projection identity mismatch");
    const players = optimizerPlayersFromProjections(data.projections);
    if (new Set(players.map((player) => player.id)).size !== players.length) throw new Error("Duplicate projection identity");
    return { valid: true, error: "", players, identity: { season, gameweek, runId, playerIds: players.map((player) => player.id) } };
  } catch {
    return { valid: false, error: "The frozen projection artifact is malformed or missing authoritative optimizer fields.", players: [], identity: null };
  }
}

function signed(value: number): string {
  if (Math.abs(value) < 0.005) return "±0.00";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}

function browserStorage(): Storage | null {
  try {
    return window.localStorage ?? null;
  } catch {
    return null;
  }
}

function allowCalculatingStateToPaint(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === "function") {
      // The second frame starts only after the first frame has painted the busy state.
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    } else {
      setTimeout(resolve, 0);
    }
  });
}
