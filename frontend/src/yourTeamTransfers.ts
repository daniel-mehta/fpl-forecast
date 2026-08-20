import {
  optimizeInheritedFixedSquad,
  type FixedSquadResult,
  type LineupDecision,
  type OptimizerPlayer,
  validateSquad,
} from "./yourTeamOptimizer";

export interface TransferRecommendation {
  playerOut: OptimizerPlayer;
  playerIn: OptimizerPlayer;
  incomingPriceTenths: number;
  outgoingSellingPriceTenths: number;
  bankRemainingTenths: number;
  expectedRealizedBefore: number;
  expectedRealizedAfter: number;
  grossImprovement: number;
  pointsHit: number;
  netImprovement: number;
  result: FixedSquadResult;
  startingXiChanged: boolean;
  captainChanged: boolean;
  viceCaptainChanged: boolean;
  benchOrderChanged: boolean;
}

export interface TransferRecommendationResult {
  recommendations: TransferRecommendation[];
  recommendNoTransfer: boolean;
  legalCandidatesEvaluated: number;
  legalCandidateCount: number;
  shortlisted: boolean;
}

export const TRANSFER_SHORTLIST_LIMIT = 5;

export function recommendSingleTransfers({
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
  const squadErrors = validateSquad(squad);
  if (squadErrors.length) throw new Error(squadErrors.join(" "));
  if (!Number.isInteger(bankTenths) || bankTenths < 0) throw new Error("Bank must be a non-negative £0.1m amount.");
  if (!Number.isInteger(freeTransfers) || freeTransfers < 0 || freeTransfers > 5) {
    throw new Error("Free transfers must be a whole number from 0 to 5.");
  }
  const ownedIds = new Set(squad.map((player) => player.id));
  const recommendations: TransferRecommendation[] = [];
  const legalCandidates: {
    playerOut: OptimizerPlayer;
    playerIn: OptimizerPlayer;
    sellingPrice: number;
    bankRemaining: number;
    proxyGain: number;
  }[] = [];
  for (const playerOut of [...squad].sort(comparePlayerIds)) {
    const sellingPrice = sellingPrices[playerOut.id];
    if (!Number.isInteger(sellingPrice) || sellingPrice <= 0) {
      throw new Error(`Selling price for ${playerOut.name} must be a positive £0.1m amount.`);
    }
    for (const playerIn of [...pool].sort(comparePlayerIds)) {
      if (ownedIds.has(playerIn.id) || playerIn.position !== playerOut.position) continue;
      const bankRemaining = bankTenths + sellingPrice - playerIn.priceTenths;
      if (bankRemaining < 0) continue;
      const nextSquad = squad.map((player) => (player.id === playerOut.id ? playerIn : player));
      if (validateSquad(nextSquad).length) continue;
      legalCandidates.push({
        playerOut,
        playerIn,
        sellingPrice,
        bankRemaining,
        proxyGain:
          playerIn.appearanceProbability * playerIn.conditionalPoints -
          playerOut.appearanceProbability * playerOut.conditionalPoints,
      });
    }
  }
  // Exact refinement is the expensive step. The deterministic shortlist ranks the complete legal
  // pool by the same authoritative unconditional player value used by D2's approximate seed. It
  // retains the strongest legal one-player gains before exact 32,768-state re-optimization.
  legalCandidates.sort(
    (left, right) =>
      right.proxyGain - left.proxyGain ||
      compareStrings(`${left.playerOut.id}|${left.playerIn.id}`, `${right.playerOut.id}|${right.playerIn.id}`),
  );
  for (const { playerOut, playerIn, sellingPrice, bankRemaining } of legalCandidates.slice(
    0,
    TRANSFER_SHORTLIST_LIMIT,
  )) {
      const nextSquad = squad.map((player) => (player.id === playerOut.id ? playerIn : player));
      const result = optimizeInheritedFixedSquad(
        nextSquad,
        baseline.decision,
        playerOut.id,
        playerIn.id,
      );
      const grossImprovement = result.breakdown.expectedRealizedTotal - baseline.breakdown.expectedRealizedTotal;
      const pointsHit = freeTransfers > 0 ? 0 : 4;
      const netImprovement = grossImprovement - pointsHit;
      recommendations.push({
        playerOut,
        playerIn,
        incomingPriceTenths: playerIn.priceTenths,
        outgoingSellingPriceTenths: sellingPrice,
        bankRemainingTenths: bankRemaining,
        expectedRealizedBefore: baseline.breakdown.expectedRealizedTotal,
        expectedRealizedAfter: result.breakdown.expectedRealizedTotal,
        grossImprovement,
        pointsHit,
        netImprovement,
        result,
        startingXiChanged: setKey(result.decision.lineup) !== setKey(baseline.decision.lineup),
        captainChanged: result.decision.captain !== baseline.decision.captain,
        viceCaptainChanged: result.decision.viceCaptain !== baseline.decision.viceCaptain,
        benchOrderChanged: result.decision.bench.join(",") !== baseline.decision.bench.join(","),
      });
  }
  recommendations.sort(compareRecommendations);
  const top = recommendations.slice(0, 5);
  return {
    recommendations: top,
    recommendNoTransfer: top.length === 0 || top[0].netImprovement <= 0,
    legalCandidatesEvaluated: recommendations.length,
    legalCandidateCount: legalCandidates.length,
    shortlisted: legalCandidates.length > TRANSFER_SHORTLIST_LIMIT,
  };
}

export function changedRoleSummary(before: LineupDecision, after: LineupDecision): string[] {
  const changes: string[] = [];
  if (setKey(before.lineup) !== setKey(after.lineup)) changes.push("Starting XI changed");
  if (before.captain !== after.captain) changes.push("Captain changed");
  if (before.viceCaptain !== after.viceCaptain) changes.push("Vice-captain changed");
  if (before.bench.join(",") !== after.bench.join(",")) changes.push("Bench order changed");
  return changes;
}

function compareRecommendations(left: TransferRecommendation, right: TransferRecommendation): number {
  const numeric = [
    right.netImprovement - left.netImprovement,
    right.grossImprovement - left.grossImprovement,
    right.expectedRealizedAfter - left.expectedRealizedAfter,
    right.bankRemainingTenths - left.bankRemainingTenths,
  ];
  for (const value of numeric) if (Math.abs(value) > 1e-10) return value;
  return compareStrings(`${left.playerOut.id}|${left.playerIn.id}`, `${right.playerOut.id}|${right.playerIn.id}`);
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
