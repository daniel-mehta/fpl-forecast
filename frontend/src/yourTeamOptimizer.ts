import type { ProjectionRow } from "./data";

export type FplPosition = "GKP" | "DEF" | "MID" | "FWD";

export interface OptimizerPlayer {
  id: string;
  name: string;
  team: string;
  position: FplPosition;
  priceTenths: number;
  expectedPoints: number;
  conditionalPoints: number;
  appearanceProbability: number;
  projection: ProjectionRow;
}

export interface LineupDecision {
  lineup: string[];
  captain: string;
  viceCaptain: string;
  bench: string[];
  formation: string;
  objective: number;
}

export interface ExpectedRealizedBreakdown {
  nominalStartingXiExpectedPoints: number;
  expectedActiveStarterPoints: number;
  expectedAutosubContribution: number;
  expectedCaptainBonus: number;
  expectedViceCaptainContingency: number;
  expectedRealizedTotal: number;
  probabilityAllStartersAppear: number;
  expectedAutomaticSubstitutions: number;
  probabilityUnreplacedStarter: number;
  scenarioCount: 32768;
  probabilityMass: number;
  analyticMethod: "exact_32768_state_independent_appearance_enumeration";
}

export interface FixedSquadResult {
  decision: LineupDecision;
  breakdown: ExpectedRealizedBreakdown;
  exactEvaluations: number;
  iterations: number;
}

export const POSITION_QUOTAS: Record<FplPosition, number> = {
  GKP: 2,
  DEF: 5,
  MID: 5,
  FWD: 3,
};

const MIN_STARTERS: Record<FplPosition, number> = { GKP: 1, DEF: 3, MID: 2, FWD: 1 };
const MAX_STARTERS: Record<FplPosition, number> = { GKP: 1, DEF: 5, MID: 5, FWD: 3 };
const POSITIONS = Object.keys(POSITION_QUOTAS) as FplPosition[];
interface AutosubOutcome {
  usedMask: number;
  unreplaced: number;
}

const autosubStructureCache = new Map<string, Map<string, AutosubOutcome[]>>();
const missingDistributionCache = new Map<string, Map<string, number>>();
const benchProbabilityCache = new Map<string, Float64Array>();
const AUTOSUB_STRUCTURE_CACHE_LIMIT = 512;
const PROBABILITY_CACHE_LIMIT = 2048;

export function optimizerPlayersFromProjections(rows: ProjectionRow[]): OptimizerPlayer[] {
  return rows.map((row) => {
    const values = {
      priceTenths: Number(row.price_tenths),
      expectedPoints: Number(row.expected_points),
      conditionalPoints: Number(row.expected_points_given_appearance),
      appearanceProbability: Number(row.p_appearance),
    };
    if (
      !row.stable_player_id ||
      !row.player ||
      !row.team ||
      !POSITIONS.includes(row.position as FplPosition) ||
      !Number.isInteger(values.priceTenths) ||
      values.priceTenths < 0 ||
      !Number.isFinite(values.expectedPoints) ||
      !Number.isFinite(values.conditionalPoints) ||
      !Number.isFinite(values.appearanceProbability) ||
      values.appearanceProbability < 0 ||
      values.appearanceProbability > 1
    ) {
      throw new Error(
        `The frozen projection for ${row.player || row.stable_player_id || "an unknown player"} is missing a required optimizer field.`,
      );
    }
    return {
      id: row.stable_player_id,
      name: row.player,
      team: row.team,
      position: row.position as FplPosition,
      ...values,
      projection: row,
    };
  });
}

export function validateSquad(players: readonly OptimizerPlayer[]): string[] {
  const errors: string[] = [];
  if (players.length !== 15) errors.push("Select exactly 15 players.");
  if (new Set(players.map((player) => player.id)).size !== players.length) {
    errors.push("A player cannot be selected more than once.");
  }
  for (const position of POSITIONS) {
    const count = players.filter((player) => player.position === position).length;
    if (count !== POSITION_QUOTAS[position]) {
      errors.push(`Select exactly ${POSITION_QUOTAS[position]} ${position} players (currently ${count}).`);
    }
  }
  const clubs = new Map<string, number>();
  for (const player of players) clubs.set(player.team, (clubs.get(player.team) ?? 0) + 1);
  if ([...clubs.values()].some((count) => count > 3)) {
    errors.push("A squad may contain no more than three players from one club.");
  }
  return errors;
}

// Port of expected_realized.py::optimize_lineup_expected_realized(shortlist=1)
// followed by expected_realized.py::refine_fixed_squad_lineup. Python remains authoritative.
export function optimizeFixedSquad(players: readonly OptimizerPlayer[]): FixedSquadResult {
  const errors = validateSquad(players);
  if (errors.length) throw new Error(errors.join(" "));
  const frame = playerMap(players);
  const initial = initialExpectedRealizedDecision(frame);
  return refineFixedSquad(players, initial);
}

// Port of milp.py::_inherited_expected_realized_solution plus the final fixed-squad refinement.
export function optimizeInheritedFixedSquad(
  players: readonly OptimizerPlayer[],
  reference: LineupDecision,
  outgoingId: string,
  incomingId: string,
): FixedSquadResult {
  const errors = validateSquad(players);
  if (errors.length) throw new Error(errors.join(" "));
  const frame = playerMap(players);
  const lineup = reference.lineup.map((id) => (id === outgoingId ? incomingId : id));
  const bench = reference.bench.map((id) => (id === outgoingId ? incomingId : id));
  if (!isLegalLineup(lineup, frame) || bench[0] === undefined || frame.get(bench[0])?.position !== "GKP") {
    return optimizeFixedSquad(players);
  }
  const [captain, viceCaptain] = bestCaptainPair(lineup, frame);
  const initial: LineupDecision = {
    lineup,
    captain,
    viceCaptain,
    bench,
    formation: formation(lineup, frame),
    objective: 0,
  };
  initial.objective = evaluateExpectedRealized(players, initial).expectedRealizedTotal;
  return refineFixedSquad(players, initial);
}

// Multi-swap extension of milp.py::_inherited_expected_realized_solution. It applies a complete
// simultaneous replacement map to the reference lineup and bench, then runs the same authoritative
// fixed-squad D2 refinement. Python remains authoritative for the underlying lineup semantics.
export function optimizeInheritedCombinedFixedSquad(
  players: readonly OptimizerPlayer[],
  reference: LineupDecision,
  replacements: ReadonlyMap<string, string>,
): FixedSquadResult {
  const errors = validateSquad(players);
  if (errors.length) throw new Error(errors.join(" "));
  const frame = playerMap(players);
  const lineup = reference.lineup.map((id) => replacements.get(id) ?? id);
  const bench = reference.bench.map((id) => replacements.get(id) ?? id);
  if (
    new Set([...lineup, ...bench]).size !== 15 ||
    !isLegalLineup(lineup, frame) ||
    bench[0] === undefined ||
    frame.get(bench[0])?.position !== "GKP"
  ) {
    return optimizeFixedSquad(players);
  }
  const [captain, viceCaptain] = bestCaptainPair(lineup, frame);
  const initial: LineupDecision = {
    lineup,
    captain,
    viceCaptain,
    bench,
    formation: formation(lineup, frame),
    objective: 0,
  };
  initial.objective = evaluateExpectedRealized(players, initial).expectedRealizedTotal;
  return refineFixedSquad(players, initial);
}

// Exact sum over the same 2^15 independent appearance states used by
// expected_realized.py::evaluate_expected_realized_points. The browser groups starter states
// before applying four bench masks; no sampling or xP/p(appearance) reconstruction is used.
export function evaluateExpectedRealized(
  players: readonly OptimizerPlayer[],
  decision: LineupDecision,
): ExpectedRealizedBreakdown {
  const frame = playerMap(players);
  validateDecision(decision, frame);
  const starters = decision.lineup.map((id) => frame.get(id)!);
  const bench = decision.bench.map((id) => frame.get(id)!);
  const expectedActiveStarterPoints = starters.reduce(
    (sum, player) => sum + player.appearanceProbability * player.conditionalPoints,
    0,
  );
  const captain = frame.get(decision.captain)!;
  const vice = frame.get(decision.viceCaptain)!;
  const expectedCaptainBonus = captain.appearanceProbability * captain.conditionalPoints;
  const expectedViceCaptainContingency =
    (1 - captain.appearanceProbability) * vice.appearanceProbability * vice.conditionalPoints;

  let expectedAutosubContribution = 0;
  let expectedAutomaticSubstitutions = 0;
  let probabilityUnreplacedStarter = 0;
  const probabilityAllStartersAppear = starters.reduce(
    (probability, player) => probability * player.appearanceProbability,
    1,
  );
  let probabilityMass = 0;
  const benchStateCount = 1 << bench.length;
  const benchProbabilities = cachedStateProbabilities(bench);
  const missingSequences = cachedMissingSequenceDistribution(starters);
  const starterPositions = starters.map((player) => player.position);
  const autosubOutcomes = autosubOutcomesForStructure(starterPositions, bench);
  for (const [missingKey, starterProbability] of missingSequences) {
    if (starterProbability === 0) continue;
    const outcomes = cachedAutosubOutcomes(autosubOutcomes, missingKey, starterPositions, bench);
    for (let benchMask = 0; benchMask < benchStateCount; benchMask += 1) {
      const stateProbability = starterProbability * benchProbabilities[benchMask];
      if (stateProbability === 0) continue;
      probabilityMass += stateProbability;
      const outcome = outcomes[benchMask];
      const usedMask = outcome.usedMask;
      let autosubPoints = 0;
      for (let index = 0; index < bench.length; index += 1) {
        if (usedMask & (1 << index)) autosubPoints += bench[index].conditionalPoints;
      }
      expectedAutosubContribution += stateProbability * autosubPoints;
      expectedAutomaticSubstitutions += stateProbability * bitCount(usedMask);
      if (outcome.unreplaced > 0) probabilityUnreplacedStarter += stateProbability;
    }
  }
  if (Math.abs(probabilityMass - 1) > 1e-12) {
    throw new Error(`Independent appearance probability mass is ${probabilityMass}; expected 1.`);
  }
  const expectedRealizedTotal =
    expectedActiveStarterPoints +
    expectedAutosubContribution +
    expectedCaptainBonus +
    expectedViceCaptainContingency;
  return {
    nominalStartingXiExpectedPoints: expectedActiveStarterPoints,
    expectedActiveStarterPoints,
    expectedAutosubContribution,
    expectedCaptainBonus,
    expectedViceCaptainContingency,
    expectedRealizedTotal,
    probabilityAllStartersAppear,
    expectedAutomaticSubstitutions,
    probabilityUnreplacedStarter,
    scenarioCount: 32768,
    probabilityMass,
    analyticMethod: "exact_32768_state_independent_appearance_enumeration",
  };
}

function initialExpectedRealizedDecision(frame: Map<string, OptimizerPlayer>): LineupDecision {
  const ids = [...frame.keys()];
  let best:
    | { score: number; lineup: string[]; bench: string[]; captain: string; viceCaptain: string }
    | undefined;
  for (const lineup of combinations(ids, 11)) {
    if (!isLegalLineup(lineup, frame)) continue;
    const remaining = ids.filter((id) => !lineup.includes(id));
    const goalkeeper = remaining.filter((id) => frame.get(id)!.position === "GKP");
    const outfield = remaining.filter((id) => frame.get(id)!.position !== "GKP");
    if (goalkeeper.length !== 1 || outfield.length !== 3) continue;
    const [captain, viceCaptain] = bestCaptainPair(lineup, frame);
    for (const order of permutations(outfield)) {
      const bench = [goalkeeper[0], ...order];
      const candidate = {
        score: approximateRealizedScore(lineup, bench, captain, viceCaptain, frame),
        lineup: [...lineup],
        bench,
        captain,
        viceCaptain,
      };
      if (!best || compareApproximate(candidate, best) > 0) best = candidate;
    }
  }
  if (!best) throw new Error("No legal expected-realized lineup can be formed from squad.");
  const decision: LineupDecision = {
    lineup: best.lineup,
    bench: best.bench,
    captain: best.captain,
    viceCaptain: best.viceCaptain,
    formation: formation(best.lineup, frame),
    objective: 0,
  };
  decision.objective = evaluateExpectedRealized([...frame.values()], decision).expectedRealizedTotal;
  return decision;
}

function refineFixedSquad(
  players: readonly OptimizerPlayer[],
  initial: LineupDecision,
): FixedSquadResult {
  const frame = playerMap(players);
  const evaluationCache = new Map<string, ExpectedRealizedBreakdown>();
  const evaluate = (decision: LineupDecision) => {
    const key = `${decision.lineup.join(",")}|${decision.bench.join(",")}|${decision.captain}|${decision.viceCaptain}`;
    const cached = evaluationCache.get(key);
    if (cached) return { breakdown: cached, evaluated: false };
    const breakdown = evaluateExpectedRealized(players, decision);
    evaluationCache.set(key, breakdown);
    return { breakdown, evaluated: true };
  };
  let best = cloneDecision(initial);
  const initialEvaluation = evaluate(best);
  let bestBreakdown = initialEvaluation.breakdown;
  best.objective = bestBreakdown.expectedRealizedTotal;
  let exactEvaluations = initialEvaluation.evaluated ? 1 : 0;
  let iterations = 0;
  while (true) {
    iterations += 1;
    let iterationBest = best;
    let iterationBreakdown = bestBreakdown;
    for (const candidate of fixedSquadCandidates(best, frame)) {
      const [captain, viceCaptain] = bestCaptainPair(candidate.lineup, frame);
      const decision: LineupDecision = {
        ...candidate,
        captain,
        viceCaptain,
        formation: formation(candidate.lineup, frame),
        objective: 0,
      };
      const evaluation = evaluate(decision);
      const breakdown = evaluation.breakdown;
      if (evaluation.evaluated) exactEvaluations += 1;
      decision.objective = breakdown.expectedRealizedTotal;
      if (compareDecisions(decision, breakdown, iterationBest, iterationBreakdown, frame) > 0) {
        iterationBest = decision;
        iterationBreakdown = breakdown;
      }
    }
    if (compareDecisions(iterationBest, iterationBreakdown, best, bestBreakdown, frame) <= 0) break;
    best = iterationBest;
    bestBreakdown = iterationBreakdown;
  }
  return { decision: best, breakdown: bestBreakdown, exactEvaluations, iterations };
}

function fixedSquadCandidates(
  decision: LineupDecision,
  frame: Map<string, OptimizerPlayer>,
): { lineup: string[]; bench: string[] }[] {
  const lineups = new Map<string, string[]>();
  const put = (lineup: string[]) => lineups.set([...lineup].sort().join(","), lineup);
  put([...decision.lineup]);
  const startingGoalkeeper = decision.lineup.find((id) => frame.get(id)!.position === "GKP")!;
  const benchGoalkeeper = decision.bench[0];
  put(decision.lineup.map((id) => (id === startingGoalkeeper ? benchGoalkeeper : id)));
  const outfieldStarters = decision.lineup.filter((id) => frame.get(id)!.position !== "GKP");
  const outfieldBench = decision.bench.filter((id) => frame.get(id)!.position !== "GKP");
  for (const starter of outfieldStarters) {
    for (const substitute of outfieldBench) {
      const trial = decision.lineup.map((id) => (id === starter ? substitute : id));
      if (isLegalLineup(trial, frame)) put(trial);
    }
  }
  const candidates: { lineup: string[]; bench: string[] }[] = [];
  const orderedLineups = [...lineups.values()].sort((left, right) =>
    compareStrings([...left].sort().join(","), [...right].sort().join(",")),
  );
  for (const lineup of orderedLineups) {
    const remaining = [...frame.keys()].filter((id) => !lineup.includes(id));
    const goalkeeper = remaining.filter((id) => frame.get(id)!.position === "GKP").sort();
    const outfield = remaining.filter((id) => frame.get(id)!.position !== "GKP").sort();
    for (const order of permutations(outfield)) candidates.push({ lineup: [...lineup], bench: [goalkeeper[0], ...order] });
  }
  return candidates;
}

function autosubOutcomesForStructure(
  starterPositions: FplPosition[],
  bench: OptimizerPlayer[],
): Map<string, AutosubOutcome[]> {
  const key = `${starterPositions.join(",")}|${bench.map((player) => player.position).join(",")}`;
  let outcomes = autosubStructureCache.get(key);
  if (!outcomes) {
    outcomes = new Map();
    evictOldestIfFull(autosubStructureCache, AUTOSUB_STRUCTURE_CACHE_LIMIT);
    autosubStructureCache.set(key, outcomes);
  }
  return outcomes;
}

function cachedAutosubOutcomes(
  cache: Map<string, AutosubOutcome[]>,
  missingKey: string,
  starterPositions: FplPosition[],
  bench: OptimizerPlayer[],
): AutosubOutcome[] {
  const cached = cache.get(missingKey);
  if (cached) return cached;
  const missing = missingKey ? (missingKey.split(",") as FplPosition[]) : [];
  const outcomes = Array.from({ length: 1 << bench.length }, (_, benchMask) =>
    autosubOutcome(missing, starterPositions, bench, benchMask),
  );
  cache.set(missingKey, outcomes);
  return outcomes;
}

function autosubOutcome(
  missing: FplPosition[],
  starterPositions: FplPosition[],
  bench: OptimizerPlayer[],
  benchMask: number,
): { usedMask: number; unreplaced: number } {
  // Python starts with all nominal starters in the active list and removes each absent starter in
  // lineup order. Later absent starters therefore still constrain formation legality until their
  // own turn, which is material to bench-order parity.
  const active = [...starterPositions];
  const availableBench = bench.map((player, index) => ({ player, index }));
  let usedMask = 0;
  let unreplaced = 0;
  for (const position of missing) {
    let replacementIndex = -1;
    for (let candidateIndex = 0; candidateIndex < availableBench.length; candidateIndex += 1) {
      const { player: candidate, index: originalBenchIndex } = availableBench[candidateIndex];
      if (!(benchMask & (1 << originalBenchIndex))) continue;
      if (position === "GKP" ? candidate.position !== "GKP" : candidate.position === "GKP") continue;
      const trial = [...active];
      trial.splice(trial.indexOf(position), 1);
      trial.push(candidate.position);
      if (isLegalPositions(trial)) {
        replacementIndex = candidateIndex;
        break;
      }
    }
    if (replacementIndex < 0) {
      unreplaced += 1;
      continue;
    }
    const replacement = availableBench[replacementIndex];
    active.splice(active.indexOf(position), 1);
    active.push(replacement.player.position);
    availableBench.splice(replacementIndex, 1);
    usedMask |= 1 << replacement.index;
  }
  return { usedMask, unreplaced };
}

function missingSequenceDistribution(starters: OptimizerPlayer[]): Map<string, number> {
  let distribution = new Map<string, number>([["", 1]]);
  for (const player of starters) {
    const next = new Map<string, number>();
    for (const [key, probability] of distribution) {
      next.set(key, (next.get(key) ?? 0) + probability * player.appearanceProbability);
      const missingKey = key ? `${key},${player.position}` : player.position;
      next.set(missingKey, (next.get(missingKey) ?? 0) + probability * (1 - player.appearanceProbability));
    }
    distribution = next;
  }
  return distribution;
}

function cachedMissingSequenceDistribution(starters: OptimizerPlayer[]): Map<string, number> {
  const key = starters
    .map((player) => `${player.id}:${player.position}:${player.appearanceProbability}`)
    .join("|");
  let distribution = missingDistributionCache.get(key);
  if (!distribution) {
    distribution = missingSequenceDistribution(starters);
    evictOldestIfFull(missingDistributionCache, PROBABILITY_CACHE_LIMIT);
    missingDistributionCache.set(key, distribution);
  }
  return distribution;
}

function stateProbabilities(players: OptimizerPlayer[]): Float64Array {
  let probabilities = new Float64Array([1]);
  for (const player of players) {
    const next = new Float64Array(probabilities.length * 2);
    for (let mask = 0; mask < probabilities.length; mask += 1) {
      next[mask] = probabilities[mask] * (1 - player.appearanceProbability);
      next[mask + probabilities.length] = probabilities[mask] * player.appearanceProbability;
    }
    probabilities = next;
  }
  return probabilities;
}

function cachedStateProbabilities(players: OptimizerPlayer[]): Float64Array {
  const key = players.map((player) => `${player.id}:${player.appearanceProbability}`).join("|");
  let probabilities = benchProbabilityCache.get(key);
  if (!probabilities) {
    probabilities = stateProbabilities(players);
    evictOldestIfFull(benchProbabilityCache, PROBABILITY_CACHE_LIMIT);
    benchProbabilityCache.set(key, probabilities);
  }
  return probabilities;
}

function bitCount(value: number): number {
  let count = 0;
  for (let current = value; current; current >>= 1) count += current & 1;
  return count;
}

function evictOldestIfFull<K, V>(cache: Map<K, V>, limit: number): void {
  if (cache.size < limit) return;
  const oldest = cache.keys().next().value as K | undefined;
  if (oldest !== undefined) cache.delete(oldest);
}

function bestCaptainPair(lineup: string[], frame: Map<string, OptimizerPlayer>): [string, string] {
  let best: [number, string, string] | undefined;
  for (const captainId of lineup) {
    const captain = frame.get(captainId)!;
    const captainValue = captain.appearanceProbability * captain.conditionalPoints;
    for (const viceId of lineup) {
      if (captainId === viceId) continue;
      const vice = frame.get(viceId)!;
      const value = captainValue + (1 - captain.appearanceProbability) * vice.appearanceProbability * vice.conditionalPoints;
      const key: [number, string, string] = [value, captainId, viceId];
      if (!best || compareTuple(key, best) > 0) best = key;
    }
  }
  if (!best) throw new Error("Could not choose captain and vice-captain.");
  return [best[1], best[2]];
}

function approximateRealizedScore(
  lineup: string[],
  bench: string[],
  captainId: string,
  viceId: string,
  frame: Map<string, OptimizerPlayer>,
): number {
  const unconditional = (id: string) => {
    const player = frame.get(id)!;
    return player.appearanceProbability * player.conditionalPoints;
  };
  const starterPoints = lineup.reduce((sum, id) => sum + unconditional(id), 0);
  const missingExpectation = lineup.reduce((sum, id) => sum + 1 - frame.get(id)!.appearanceProbability, 0);
  let benchValue = 0;
  bench.forEach((id, index) => {
    const player = frame.get(id)!;
    if (player.position === "GKP") {
      const goalkeeperMissing = lineup
        .filter((starter) => frame.get(starter)!.position === "GKP")
        .reduce((sum, starter) => sum + 1 - frame.get(starter)!.appearanceProbability, 0);
      benchValue += goalkeeperMissing * unconditional(id);
    } else if (canEverSubstitute(id, lineup, frame)) {
      benchValue += (missingExpectation * Math.max(0, 1 - 0.25 * index) * unconditional(id)) / 3;
    }
  });
  const captain = frame.get(captainId)!;
  return starterPoints + benchValue + unconditional(captainId) + (1 - captain.appearanceProbability) * unconditional(viceId);
}

function compareApproximate(
  left: { score: number; lineup: string[]; bench: string[] },
  right: { score: number; lineup: string[]; bench: string[] },
): number {
  return compareTuple(
    [round(left.score, 10), [...left.lineup].sort().join(","), left.bench.join(",")],
    [round(right.score, 10), [...right.lineup].sort().join(","), right.bench.join(",")],
  );
}

function compareDecisions(
  left: LineupDecision,
  leftBreakdown: ExpectedRealizedBreakdown,
  right: LineupDecision,
  rightBreakdown: ExpectedRealizedBreakdown,
  frame: Map<string, OptimizerPlayer>,
): number {
  const points = (ids: string[]) => ids.reduce((sum, id) => sum + frame.get(id)!.expectedPoints, 0);
  const key = (decision: LineupDecision, breakdown: ExpectedRealizedBreakdown) => [
    round(breakdown.expectedRealizedTotal, 8),
    round(points(decision.lineup), 8),
    round(points(decision.bench), 8),
    round(breakdown.expectedViceCaptainContingency, 8),
    -round(breakdown.probabilityUnreplacedStarter, 8),
    1000 - [...frame.values()].reduce((sum, player) => sum + player.priceTenths, 0),
    `${[...decision.lineup].sort().join(",")}|${decision.bench.join(",")}|${decision.captain}|${decision.viceCaptain}`,
  ];
  return compareTuple(key(left, leftBreakdown), key(right, rightBreakdown));
}

function validateDecision(decision: LineupDecision, frame: Map<string, OptimizerPlayer>) {
  if (
    decision.lineup.length !== 11 ||
    decision.bench.length !== 4 ||
    new Set([...decision.lineup, ...decision.bench]).size !== 15 ||
    !isLegalLineup(decision.lineup, frame) ||
    frame.get(decision.bench[0])?.position !== "GKP" ||
    !decision.lineup.includes(decision.captain) ||
    !decision.lineup.includes(decision.viceCaptain) ||
    decision.captain === decision.viceCaptain
  ) {
    throw new Error("Decision is not a legal partition of the 15-player squad.");
  }
}

function playerMap(players: readonly OptimizerPlayer[]): Map<string, OptimizerPlayer> {
  return new Map([...players].sort((a, b) => compareStrings(a.id, b.id)).map((player) => [player.id, player]));
}

function isLegalLineup(lineup: string[], frame: Map<string, OptimizerPlayer>): boolean {
  return lineup.length === 11 && isLegalPositions(lineup.map((id) => frame.get(id)!.position));
}

function isLegalPositions(positions: FplPosition[]): boolean {
  const counts = Object.fromEntries(POSITIONS.map((position) => [position, 0])) as Record<FplPosition, number>;
  positions.forEach((position) => (counts[position] += 1));
  return POSITIONS.every(
    (position) => counts[position] >= MIN_STARTERS[position] && counts[position] <= MAX_STARTERS[position],
  );
}

function canEverSubstitute(id: string, lineup: string[], frame: Map<string, OptimizerPlayer>): boolean {
  return lineup.some((starter) => {
    if (frame.get(starter)!.position === "GKP") return false;
    return isLegalLineup(lineup.map((value) => (value === starter ? id : value)), frame);
  });
}

function formation(lineup: string[], frame: Map<string, OptimizerPlayer>): string {
  const count = (position: FplPosition) => lineup.filter((id) => frame.get(id)!.position === position).length;
  return `${count("DEF")}-${count("MID")}-${count("FWD")}`;
}

function combinations<T>(values: T[], count: number): T[][] {
  const result: T[][] = [];
  const visit = (start: number, selected: T[]) => {
    if (selected.length === count) {
      result.push([...selected]);
      return;
    }
    for (let index = start; index <= values.length - (count - selected.length); index += 1) {
      selected.push(values[index]);
      visit(index + 1, selected);
      selected.pop();
    }
  };
  visit(0, []);
  return result;
}

function permutations<T>(values: T[]): T[][] {
  if (values.length <= 1) return [[...values]];
  return values.flatMap((value, index) =>
    permutations([...values.slice(0, index), ...values.slice(index + 1)]).map((tail) => [value, ...tail]),
  );
}

function compareTuple(left: (number | string)[], right: (number | string)[]): number {
  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const a = left[index];
    const b = right[index];
    if (a === b) continue;
    if (typeof a === "number" && typeof b === "number") return a > b ? 1 : -1;
    return compareStrings(String(a), String(b));
  }
  return 0;
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round((value + Number.EPSILON) * factor) / factor;
}

function cloneDecision(decision: LineupDecision): LineupDecision {
  return { ...decision, lineup: [...decision.lineup], bench: [...decision.bench] };
}
