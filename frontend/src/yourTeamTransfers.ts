import {
  optimizeInheritedCombinedFixedSquad,
  type FixedSquadResult,
  type LineupDecision,
  type OptimizerPlayer,
  validateSquad,
} from "./yourTeamOptimizer";

export interface TransferOption {
  playerOut: OptimizerPlayer;
  playerIn: OptimizerPlayer;
  incomingPriceTenths: number;
  outgoingSellingPriceTenths: number;
  bankRemainingTenths: number;
  expectedRealizedAfter: number;
  grossImprovement: number;
  pointsHit: number;
  netImprovement: number;
  result: FixedSquadResult;
  startingXiChanged: boolean;
  captainChanged: boolean;
  viceCaptainChanged: boolean;
  benchOrderChanged: boolean;
  primary: boolean;
}

export interface TransferRecommendationGroup {
  playerOut: OptimizerPlayer;
  outgoingSellingPriceTenths: number;
  options: TransferOption[];
}

export interface TransferRecommendationResult {
  groups: TransferRecommendationGroup[];
  primaryTransfers: TransferOption[];
  recommendNoTransfer: boolean;
  freeTransfersAvailable: number;
  transfersUsed: number;
  freeTransfersRolled: number;
  expectedRealizedBefore: number;
  expectedRealizedAfter: number;
  grossImprovement: number;
  pointsHit: number;
  netImprovement: number;
  bankRemainingTenths: number;
  result: FixedSquadResult | null;
  exactPlanEvaluations: number;
  candidatePlanCount: number;
  searchedDepths: number[];
  shortlisted: boolean;
}

interface CandidateTransfer {
  playerOut: OptimizerPlayer;
  playerIn: OptimizerPlayer;
  sellingPrice: number;
  proxyGain: number;
}

interface PlanState {
  transfers: CandidateTransfer[];
  proxyGain: number;
  bankRemaining: number;
  squad: OptimizerPlayer[];
  key: string;
}

interface EvaluatedPlan extends PlanState {
  result: FixedSquadResult;
  grossImprovement: number;
  pointsHit: number;
  netImprovement: number;
}

export const INCOMING_SHORTLIST_PER_OUTGOING = 6;
export const TRANSFER_BEAM_WIDTH = 120;
export const REPLACEMENT_OPTIONS_LIMIT = 3;
export const SINGLE_PLAN_EXACT_LIMIT = 5;
export const PRIMARY_COORDINATE_REFINEMENT_ROUNDS = 6;
const COMPARISON_TOLERANCE = 1e-10;

export function recommendTransfers({
  squad,
  pool,
  sellingPrices,
  bankTenths,
  freeTransfers,
  baseline,
}: {
  squad: readonly OptimizerPlayer[];
  pool: readonly OptimizerPlayer[];
  sellingPrices: Readonly<Record<string, number>>;
  bankTenths: number;
  freeTransfers: number;
  baseline: FixedSquadResult;
}): TransferRecommendationResult {
  validateInputs(squad, sellingPrices, bankTenths, freeTransfers);
  const ownedIds = new Set(squad.map((player) => player.id));
  const incomingShortlistWasApplied = squad.some(
    (playerOut) =>
      pool.filter(
        (playerIn) => !ownedIds.has(playerIn.id) && playerIn.position === playerOut.position,
      ).length > INCOMING_SHORTLIST_PER_OUTGOING,
  );
  const candidatesByOutgoing = buildCandidateShortlists(squad, pool, sellingPrices);
  const maximumTransfers = freeTransfers === 0 ? 1 : freeTransfers;
  const search = buildPlanShortlists(squad, candidatesByOutgoing, bankTenths, maximumTransfers);
  const exactByDepth = new Map<number, EvaluatedPlan>();
  let exactPlanEvaluations = 0;

  // The full official pool is searched with a deterministic proxy beam. The strongest legal plan
  // retained at each multi-transfer depth receives authoritative exact fixed-squad D2 refinement;
  // the one-transfer case preserves the prior five-candidate exact comparison. This keeps planning
  // practical without claiming a global combinatorial optimum.
  for (const [depth, states] of search.legalPlansByDepth) {
    const exactStates = maximumTransfers === 1 ? states.slice(0, SINGLE_PLAN_EXACT_LIMIT) : states.slice(0, 1);
    let bestAtDepth: EvaluatedPlan | undefined;
    for (const state of exactStates) {
      const result = evaluatePlan(state, baseline, freeTransfers);
      exactPlanEvaluations += 1;
      if (!bestAtDepth || compareEvaluatedPlans(result, bestAtDepth) < 0) bestAtDepth = result;
    }
    if (bestAtDepth) exactByDepth.set(depth, bestAtDepth);
  }

  const selected = selectImprovingDepth(exactByDepth, freeTransfers, maximumTransfers);
  if (!selected) {
    return emptyResult(baseline, bankTenths, freeTransfers, exactPlanEvaluations, {
      ...search,
      shortlisted: search.shortlisted || incomingShortlistWasApplied,
    });
  }

  let primaryPlan = selected;
  const evaluatedPlanCache = new Map<string, EvaluatedPlan>([[selected.key, selected]]);
  for (let round = 0; round < PRIMARY_COORDINATE_REFINEMENT_ROUNDS; round += 1) {
    let improved = false;
    const currentIncomingIds = new Set(primaryPlan.transfers.map((transfer) => transfer.playerIn.id));
    for (const primaryTransfer of primaryPlan.transfers) {
      let legalAlternativesEvaluated = 0;
      for (const alternative of candidatesByOutgoing.get(primaryTransfer.playerOut.id) ?? []) {
        if (
          alternative.playerIn.id === primaryTransfer.playerIn.id ||
          currentIncomingIds.has(alternative.playerIn.id)
        ) continue;
        const replacementTransfers = primaryPlan.transfers.map((transfer) =>
          transfer.playerOut.id === primaryTransfer.playerOut.id ? alternative : transfer,
        );
        const alternativeState = planState(squad, replacementTransfers, bankTenths);
        if (alternativeState.bankRemaining < 0 || validateSquad(alternativeState.squad).length) continue;
        if (legalAlternativesEvaluated >= REPLACEMENT_OPTIONS_LIMIT - 1) break;
        legalAlternativesEvaluated += 1;
        let evaluated = evaluatedPlanCache.get(alternativeState.key);
        if (!evaluated) {
          evaluated = evaluatePlan(alternativeState, baseline, freeTransfers);
          evaluatedPlanCache.set(alternativeState.key, evaluated);
          exactPlanEvaluations += 1;
        }
        if (compareEvaluatedPlans(evaluated, primaryPlan) < 0) {
          primaryPlan = evaluated;
          improved = true;
          break;
        }
      }
      if (improved) break;
    }
    if (!improved) break;
  }

  const primaryOptions = primaryPlan.transfers.map((transfer) =>
    optionFromPlan(transfer, primaryPlan, baseline, true),
  );
  const primaryIncomingIds = new Set(primaryPlan.transfers.map((transfer) => transfer.playerIn.id));
  const displayedIncomingIds = new Set(primaryIncomingIds);
  const groups: TransferRecommendationGroup[] = [];

  for (const primaryTransfer of primaryPlan.transfers) {
    const alternatives: TransferOption[] = [];
    const exactCandidates = (candidatesByOutgoing.get(primaryTransfer.playerOut.id) ?? [])
      .filter(
        (candidate) =>
          candidate.playerIn.id !== primaryTransfer.playerIn.id &&
          !primaryIncomingIds.has(candidate.playerIn.id) &&
          !displayedIncomingIds.has(candidate.playerIn.id),
      );
    for (const alternative of exactCandidates) {
      if (alternatives.length >= REPLACEMENT_OPTIONS_LIMIT - 1) break;
      const replacementTransfers = primaryPlan.transfers.map((transfer) =>
        transfer.playerOut.id === primaryTransfer.playerOut.id ? alternative : transfer,
      );
      const alternativeState = planState(squad, replacementTransfers, bankTenths);
      if (alternativeState.bankRemaining < 0 || validateSquad(alternativeState.squad).length) continue;
      let evaluated = evaluatedPlanCache.get(alternativeState.key);
      if (!evaluated) {
        evaluated = evaluatePlan(alternativeState, baseline, freeTransfers);
        evaluatedPlanCache.set(alternativeState.key, evaluated);
        exactPlanEvaluations += 1;
      }
      if (evaluated.netImprovement <= COMPARISON_TOLERANCE) continue;
      alternatives.push(optionFromPlan(alternative, evaluated, baseline, false));
      displayedIncomingIds.add(alternative.playerIn.id);
    }
    alternatives.sort(compareOptions);
    groups.push({
      playerOut: primaryTransfer.playerOut,
      outgoingSellingPriceTenths: primaryTransfer.sellingPrice,
      options: [
        optionFromPlan(primaryTransfer, primaryPlan, baseline, true),
        ...alternatives.slice(0, REPLACEMENT_OPTIONS_LIMIT - 1),
      ],
    });
  }

  return {
    groups,
    primaryTransfers: primaryOptions,
    recommendNoTransfer: false,
    freeTransfersAvailable: freeTransfers,
    transfersUsed: primaryPlan.transfers.length,
    freeTransfersRolled: Math.max(0, freeTransfers - primaryPlan.transfers.length),
    expectedRealizedBefore: baseline.breakdown.expectedRealizedTotal,
    expectedRealizedAfter: primaryPlan.result.breakdown.expectedRealizedTotal,
    grossImprovement: primaryPlan.grossImprovement,
    pointsHit: primaryPlan.pointsHit,
    netImprovement: primaryPlan.netImprovement,
    bankRemainingTenths: primaryPlan.bankRemaining,
    result: primaryPlan.result,
    exactPlanEvaluations,
    candidatePlanCount: search.candidatePlanCount,
    searchedDepths: [0, ...search.legalPlansByDepth.keys()],
    shortlisted: search.shortlisted || incomingShortlistWasApplied,
  };
}

export function applyPrimaryTransfers(
  squad: readonly OptimizerPlayer[],
  transfers: readonly Pick<TransferOption, "playerOut" | "playerIn">[],
): OptimizerPlayer[] {
  const replacements = new Map(transfers.map((transfer) => [transfer.playerOut.id, transfer.playerIn]));
  return squad.map((player) => replacements.get(player.id) ?? player);
}

export function changedRoleSummary(before: LineupDecision, after: LineupDecision): string[] {
  const changes: string[] = [];
  if (setKey(before.lineup) !== setKey(after.lineup)) changes.push("Starting XI changed");
  if (before.captain !== after.captain) changes.push("Captain changed");
  if (before.viceCaptain !== after.viceCaptain) changes.push("Vice-captain changed");
  if (before.bench.join(",") !== after.bench.join(",")) changes.push("Bench order changed");
  return changes;
}

function validateInputs(
  squad: readonly OptimizerPlayer[],
  sellingPrices: Readonly<Record<string, number>>,
  bankTenths: number,
  freeTransfers: number,
): void {
  const squadErrors = validateSquad(squad);
  if (squadErrors.length) throw new Error(squadErrors.join(" "));
  if (!Number.isInteger(bankTenths) || bankTenths < 0) {
    throw new Error("Bank must be a non-negative £0.1m amount.");
  }
  if (!Number.isInteger(freeTransfers) || freeTransfers < 0 || freeTransfers > 5) {
    throw new Error("Free transfers must be a whole number from 0 to 5.");
  }
  for (const player of squad) {
    const sellingPrice = sellingPrices[player.id];
    if (!Number.isInteger(sellingPrice) || sellingPrice <= 0) {
      throw new Error(`Selling price for ${player.name} must be a positive £0.1m amount.`);
    }
  }
}

function buildCandidateShortlists(
  squad: readonly OptimizerPlayer[],
  pool: readonly OptimizerPlayer[],
  sellingPrices: Readonly<Record<string, number>>,
): Map<string, CandidateTransfer[]> {
  const ownedIds = new Set(squad.map((player) => player.id));
  const shortlists = new Map<string, CandidateTransfer[]>();
  for (const playerOut of [...squad].sort(comparePlayerIds)) {
    const candidates = [...pool]
      .filter((playerIn) => !ownedIds.has(playerIn.id) && playerIn.position === playerOut.position)
      .map((playerIn) => ({
        playerOut,
        playerIn,
        sellingPrice: sellingPrices[playerOut.id],
        proxyGain: playerIn.expectedPoints - playerOut.expectedPoints,
      }))
      .sort(compareCandidateTransfers)
      .slice(0, INCOMING_SHORTLIST_PER_OUTGOING);
    shortlists.set(playerOut.id, candidates);
  }
  return shortlists;
}

function buildPlanShortlists(
  squad: readonly OptimizerPlayer[],
  candidatesByOutgoing: ReadonlyMap<string, readonly CandidateTransfer[]>,
  bankTenths: number,
  maximumTransfers: number,
): {
  legalPlansByDepth: Map<number, PlanState[]>;
  candidatePlanCount: number;
  shortlisted: boolean;
} {
  let beam: PlanState[] = [planState(squad, [], bankTenths)];
  const legalPlansByDepth = new Map<number, PlanState[]>();
  let candidatePlanCount = 0;
  let shortlisted = false;
  for (let depth = 1; depth <= maximumTransfers; depth += 1) {
    const expanded = new Map<string, PlanState>();
    for (const state of beam) {
      const lastOutgoingId = state.transfers.at(-1)?.playerOut.id ?? "";
      const usedIncoming = new Set(state.transfers.map((transfer) => transfer.playerIn.id));
      for (const [outgoingId, candidates] of candidatesByOutgoing) {
        if (outgoingId <= lastOutgoingId) continue;
        for (const candidate of candidates) {
          if (usedIncoming.has(candidate.playerIn.id)) continue;
          const next = planState(squad, [...state.transfers, candidate], bankTenths);
          expanded.set(next.key, next);
        }
      }
    }
    const ordered = [...expanded.values()].sort(comparePlanStates);
    const legal = ordered.filter(
      (state) => state.bankRemaining >= 0 && validateSquad(state.squad).length === 0,
    );
    candidatePlanCount += legal.length;
    legalPlansByDepth.set(depth, legal.slice(0, TRANSFER_BEAM_WIDTH));
    shortlisted ||= ordered.length > TRANSFER_BEAM_WIDTH;

    // Preserve strong legal states while retaining some temporarily over-budget or over-club-limit
    // states that a later outgoing transfer can repair.
    const nextBeam = new Map<string, PlanState>();
    for (const state of legal.slice(0, Math.floor(TRANSFER_BEAM_WIDTH / 2))) {
      nextBeam.set(state.key, state);
    }
    for (const state of ordered) {
      if (nextBeam.size >= TRANSFER_BEAM_WIDTH) break;
      nextBeam.set(state.key, state);
    }
    beam = [...nextBeam.values()].sort(comparePlanStates);
  }
  return { legalPlansByDepth, candidatePlanCount, shortlisted };
}

function selectImprovingDepth(
  exactByDepth: ReadonlyMap<number, EvaluatedPlan>,
  freeTransfers: number,
  maximumTransfers: number,
): EvaluatedPlan | null {
  let selected: EvaluatedPlan | null = null;
  for (let depth = 1; depth <= maximumTransfers; depth += 1) {
    const candidate = exactByDepth.get(depth);
    if (!candidate) continue;
    if (freeTransfers === 0) return candidate.netImprovement > COMPARISON_TOLERANCE ? candidate : null;
    if (!selected || compareEvaluatedPlans(candidate, selected) < 0) selected = candidate;
  }
  return selected && selected.netImprovement > COMPARISON_TOLERANCE ? selected : null;
}

function evaluatePlan(
  state: PlanState,
  baseline: FixedSquadResult,
  freeTransfers: number,
): EvaluatedPlan {
  const replacements = new Map(
    state.transfers.map((transfer) => [transfer.playerOut.id, transfer.playerIn.id]),
  );
  const result = optimizeInheritedCombinedFixedSquad(state.squad, baseline.decision, replacements);
  return evaluatedPlan(state, result, baseline, freeTransfers);
}

function evaluatedPlan(
  state: PlanState,
  result: FixedSquadResult,
  baseline: FixedSquadResult,
  freeTransfers: number,
): EvaluatedPlan {
  const grossImprovement = result.breakdown.expectedRealizedTotal - baseline.breakdown.expectedRealizedTotal;
  const pointsHit = freeTransfers === 0 && state.transfers.length > 0 ? 4 : 0;
  return {
    ...state,
    result,
    grossImprovement,
    pointsHit,
    netImprovement: grossImprovement - pointsHit,
  };
}

function planState(
  originalSquad: readonly OptimizerPlayer[],
  transfers: readonly CandidateTransfer[],
  bankTenths: number,
): PlanState {
  const ordered = [...transfers].sort((left, right) =>
    compareStrings(left.playerOut.id, right.playerOut.id),
  );
  const replacements = new Map(ordered.map((transfer) => [transfer.playerOut.id, transfer.playerIn]));
  const squad = originalSquad.map((player) => replacements.get(player.id) ?? player);
  const bankRemaining = ordered.reduce(
    (bank, transfer) => bank + transfer.sellingPrice - transfer.playerIn.priceTenths,
    bankTenths,
  );
  return {
    transfers: ordered,
    proxyGain: ordered.reduce((total, transfer) => total + transfer.proxyGain, 0),
    bankRemaining,
    squad,
    key: ordered.map((transfer) => `${transfer.playerOut.id}>${transfer.playerIn.id}`).join("|"),
  };
}

function optionFromPlan(
  transfer: CandidateTransfer,
  plan: EvaluatedPlan,
  baseline: FixedSquadResult,
  primary: boolean,
): TransferOption {
  return {
    playerOut: transfer.playerOut,
    playerIn: transfer.playerIn,
    incomingPriceTenths: transfer.playerIn.priceTenths,
    outgoingSellingPriceTenths: transfer.sellingPrice,
    bankRemainingTenths: plan.bankRemaining,
    expectedRealizedAfter: plan.result.breakdown.expectedRealizedTotal,
    grossImprovement: plan.grossImprovement,
    pointsHit: plan.pointsHit,
    netImprovement: plan.netImprovement,
    result: plan.result,
    startingXiChanged: setKey(plan.result.decision.lineup) !== setKey(baseline.decision.lineup),
    captainChanged: plan.result.decision.captain !== baseline.decision.captain,
    viceCaptainChanged: plan.result.decision.viceCaptain !== baseline.decision.viceCaptain,
    benchOrderChanged: plan.result.decision.bench.join(",") !== baseline.decision.bench.join(","),
    primary,
  };
}

function emptyResult(
  baseline: FixedSquadResult,
  bankTenths: number,
  freeTransfers: number,
  exactPlanEvaluations: number,
  search: {
    candidatePlanCount: number;
    shortlisted: boolean;
    legalPlansByDepth?: ReadonlyMap<number, readonly PlanState[]>;
  },
): TransferRecommendationResult {
  return {
    groups: [],
    primaryTransfers: [],
    recommendNoTransfer: true,
    freeTransfersAvailable: freeTransfers,
    transfersUsed: 0,
    freeTransfersRolled: freeTransfers,
    expectedRealizedBefore: baseline.breakdown.expectedRealizedTotal,
    expectedRealizedAfter: baseline.breakdown.expectedRealizedTotal,
    grossImprovement: 0,
    pointsHit: 0,
    netImprovement: 0,
    bankRemainingTenths: bankTenths,
    result: null,
    exactPlanEvaluations,
    candidatePlanCount: search.candidatePlanCount,
    searchedDepths: [0, ...(search.legalPlansByDepth?.keys() ?? [])],
    shortlisted: search.shortlisted,
  };
}

function compareCandidateTransfers(left: CandidateTransfer, right: CandidateTransfer): number {
  const gain = right.proxyGain - left.proxyGain;
  if (Math.abs(gain) > COMPARISON_TOLERANCE) return gain;
  const price = left.playerIn.priceTenths - right.playerIn.priceTenths;
  if (price !== 0) return price;
  return compareStrings(
    `${left.playerOut.id}|${left.playerIn.id}`,
    `${right.playerOut.id}|${right.playerIn.id}`,
  );
}

function comparePlanStates(left: PlanState, right: PlanState): number {
  const gain = right.proxyGain - left.proxyGain;
  if (Math.abs(gain) > COMPARISON_TOLERANCE) return gain;
  const bank = right.bankRemaining - left.bankRemaining;
  if (bank !== 0) return bank;
  return compareStrings(left.key, right.key);
}

function compareOptions(left: TransferOption, right: TransferOption): number {
  const values = [
    right.netImprovement - left.netImprovement,
    right.grossImprovement - left.grossImprovement,
    right.expectedRealizedAfter - left.expectedRealizedAfter,
    right.bankRemainingTenths - left.bankRemainingTenths,
  ];
  for (const value of values) if (Math.abs(value) > COMPARISON_TOLERANCE) return value;
  return compareStrings(left.playerIn.id, right.playerIn.id);
}

function compareEvaluatedPlans(left: EvaluatedPlan, right: EvaluatedPlan): number {
  const values = [
    right.netImprovement - left.netImprovement,
    right.grossImprovement - left.grossImprovement,
    right.result.breakdown.expectedRealizedTotal - left.result.breakdown.expectedRealizedTotal,
    right.bankRemaining - left.bankRemaining,
  ];
  for (const value of values) if (Math.abs(value) > COMPARISON_TOLERANCE) return value;
  const transferCount = left.transfers.length - right.transfers.length;
  if (transferCount !== 0) return transferCount;
  return compareStrings(left.key, right.key);
}

function setKey(values: readonly string[]): string {
  return [...values].sort().join(",");
}

function comparePlayerIds(left: OptimizerPlayer, right: OptimizerPlayer): number {
  return compareStrings(left.id, right.id);
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
