import { describe, expect, it } from "vitest";
import { legalSquad, projectionPool } from "./yourTeamTestFixtures";
import {
  optimizerPlayersFromProjections,
  optimizeFixedSquad,
  validateSquad,
  type OptimizerPlayer,
} from "./yourTeamOptimizer";
import {
  applyPrimaryTransfers,
  recommendTransfers,
  REPLACEMENT_OPTIONS_LIMIT,
} from "./yourTeamTransfers";

function setup(extras = 3) {
  const rows = projectionPool(extras);
  const pool = optimizerPlayersFromProjections(rows);
  const squad = legalSquad(rows);
  const baseline = optimizeFixedSquad(squad);
  const sellingPrices = Object.fromEntries(squad.map((player) => [player.id, player.priceTenths]));
  return { pool, squad, baseline, sellingPrices };
}

function withStrongReplacements(input: ReturnType<typeof setup>, count = 5) {
  const ids = ["GKP_02", "DEF_05", "DEF_06", "MID_05", "FWD_03"].slice(0, count);
  return input.pool.map((player) => {
    const index = ids.indexOf(player.id);
    return index < 0
      ? player
      : {
          ...player,
          expectedPoints: 25 - index,
          conditionalPoints: 25 - index,
          appearanceProbability: 1,
          priceTenths: 45,
          team: `strong_team_${index}`,
        };
  });
}

function primaryPairs(result: ReturnType<typeof recommendTransfers>) {
  return result.primaryTransfers.map((transfer) => `${transfer.playerOut.id}>${transfer.playerIn.id}`);
}

function replacementPlan(
  primary: ReturnType<typeof recommendTransfers>["primaryTransfers"],
  outgoingId: string,
  incoming: OptimizerPlayer,
) {
  return primary.map((transfer) =>
    transfer.playerOut.id === outgoingId ? { ...transfer, playerIn: incoming } : transfer,
  );
}

describe("combined transfer recommendations", () => {
  it("uses distinct outgoing and incoming players for five free transfers", () => {
    const input = setup();
    const result = recommendTransfers({ ...input, pool: withStrongReplacements(input), bankTenths: 30, freeTransfers: 5 });
    expect(result.transfersUsed).toBe(5);
    expect(result.groups).toHaveLength(5);
    const outgoing = result.primaryTransfers.map((item) => item.playerOut.id);
    const incoming = result.primaryTransfers.map((item) => item.playerIn.id);
    const displayedIncoming = result.groups.flatMap((group) =>
      group.options.map((option) => option.playerIn.id),
    );
    expect(new Set(outgoing).size).toBe(5);
    expect(new Set(incoming).size).toBe(5);
    expect(new Set(displayedIncoming).size).toBe(displayedIncoming.length);
    expect(result.groups.every((group) => group.options.length <= REPLACEMENT_OPTIONS_LIMIT)).toBe(true);
  }, 30_000);

  it("makes every primary and conditional alternative legal and affordable", () => {
    const input = setup();
    const result = recommendTransfers({ ...input, pool: withStrongReplacements(input), bankTenths: 30, freeTransfers: 5 });
    expect(validateSquad(applyPrimaryTransfers(input.squad, result.primaryTransfers))).toEqual([]);
    expect(result.bankRemainingTenths).toBeGreaterThanOrEqual(0);
    for (const group of result.groups) {
      expect(group.options[0].primary).toBe(true);
      for (const option of group.options) {
        const plan = replacementPlan(result.primaryTransfers, group.playerOut.id, option.playerIn);
        expect(validateSquad(applyPrimaryTransfers(input.squad, plan))).toEqual([]);
        expect(new Set(plan.map((item) => item.playerIn.id)).size).toBe(plan.length);
        expect(option.bankRemainingTenths).toBeGreaterThanOrEqual(0);
      }
      for (const alternative of group.options.slice(1)) {
        expect(group.options[0].netImprovement).toBeGreaterThanOrEqual(alternative.netImprovement - 1e-10);
      }
    }
  }, 30_000);

  it("selects the best exact depth and reports unused transfers as rolled", () => {
    const input = setup();
    const result = recommendTransfers({ ...input, pool: withStrongReplacements(input, 2), bankTenths: 20, freeTransfers: 5 });
    expect(result.transfersUsed).toBe(2);
    expect(result.freeTransfersRolled).toBe(3);
    expect(result.grossImprovement).toBeGreaterThan(0);
    expect(result.searchedDepths).toEqual([0, 1, 2, 3, 4, 5]);
  }, 30_000);

  it("accepts a temporary downgrade when it funds a positive two-transfer plan", () => {
    const input = setup(1);
    const ownedIds = new Set(input.squad.map((player) => player.id));
    const pool = input.pool.map((player) => {
      if (ownedIds.has(player.id)) return player;
      if (player.id === "MID_05") {
        return {
          ...player,
          expectedPoints: 0,
          conditionalPoints: 0,
          appearanceProbability: 1,
          priceTenths: 30,
          team: "funding_mid",
        };
      }
      if (player.id === "FWD_03") {
        return {
          ...player,
          expectedPoints: 25,
          conditionalPoints: 25,
          appearanceProbability: 1,
          priceTenths: 95,
          team: "funded_fwd",
        };
      }
      return { ...player, expectedPoints: 0, conditionalPoints: 0, priceTenths: 200 };
    });
    const sellingPrices = { ...input.sellingPrices, MID_04: 80 };
    const oneTransfer = recommendTransfers({
      ...input,
      pool,
      sellingPrices,
      bankTenths: 0,
      freeTransfers: 1,
    });
    const twoTransfers = recommendTransfers({
      ...input,
      pool,
      sellingPrices,
      bankTenths: 0,
      freeTransfers: 2,
    });

    expect(oneTransfer.recommendNoTransfer).toBe(true);
    expect(oneTransfer.grossImprovement).toBe(0);
    expect(twoTransfers.searchedDepths).toEqual([0, 1, 2]);
    expect(twoTransfers.transfersUsed).toBe(2);
    expect(primaryPairs(twoTransfers)).toEqual(["FWD_02>FWD_03", "MID_04>MID_05"]);
    expect(twoTransfers.bankRemainingTenths).toBe(2);
    expect(twoTransfers.netImprovement).toBeGreaterThan(0);
  }, 30_000);

  it("enforces combined bank and club limits while allowing one move to fund another", () => {
    const input = setup();
    const ownedClub = input.squad[0].team;
    const pool = input.pool.map((player) => {
      if (player.id === "DEF_05") return { ...player, expectedPoints: 25, conditionalPoints: 25, appearanceProbability: 1, priceTenths: 60, team: "funding_a" };
      if (player.id === "MID_05") return { ...player, expectedPoints: 24, conditionalPoints: 24, appearanceProbability: 1, priceTenths: 30, team: "funding_b" };
      if (!input.squad.some((owned) => owned.id === player.id)) return { ...player, team: ownedClub };
      return player;
    });
    const result = recommendTransfers({ ...input, pool, bankTenths: 0, freeTransfers: 2 });
    expect(result.transfersUsed).toBe(2);
    expect(result.bankRemainingTenths).toBeGreaterThanOrEqual(0);
    expect(validateSquad(applyPrimaryTransfers(input.squad, result.primaryTransfers))).toEqual([]);
  }, 30_000);

  it("uses editable selling prices for affordability", () => {
    const input = setup(1);
    const expensivePool = input.pool.map((player) =>
      player.id === "FWD_03"
        ? { ...player, expectedPoints: 30, conditionalPoints: 30, appearanceProbability: 1, priceTenths: 60 }
        : player,
    );
    const poor = recommendTransfers({ ...input, pool: expensivePool, bankTenths: 0, freeTransfers: 1 });
    const outgoing = input.squad.find((player) => player.id === "FWD_02")!;
    const rich = recommendTransfers({
      ...input,
      pool: expensivePool,
      sellingPrices: { ...input.sellingPrices, [outgoing.id]: 60 },
      bankTenths: 0,
      freeTransfers: 1,
    });
    expect(poor.primaryTransfers.some((item) => item.playerIn.id === "FWD_03")).toBe(false);
    expect(rich.primaryTransfers.some((item) => item.playerIn.id === "FWD_03")).toBe(true);
  }, 30_000);

  it("re-optimizes formation, bench order and captaincy", () => {
    const input = setup(1);
    const squad = input.squad.map((player) => player.id === "FWD_02" ? { ...player, expectedPoints: 0, conditionalPoints: 0, appearanceProbability: 1 } : player);
    const pool = input.pool.map((player) =>
      player.id === "FWD_03"
        ? { ...player, expectedPoints: 50, conditionalPoints: 50, appearanceProbability: 1, priceTenths: 45 }
        : squad.find((owned) => owned.id === player.id) ?? player,
    );
    const baseline = optimizeFixedSquad(squad);
    const result = recommendTransfers({ ...input, squad, baseline, pool, bankTenths: 20, freeTransfers: 1 });
    const primary = result.primaryTransfers[0];
    expect(primary.playerIn.id).toBe("FWD_03");
    expect(primary.captainChanged).toBe(true);
    expect(primary.benchOrderChanged).toBe(true);
    expect(primary.result.decision.formation).not.toBe(baseline.decision.formation);
  }, 30_000);

  it("is deterministic under tied candidate values", () => {
    const input = setup();
    const owned = new Set(input.squad.map((player) => player.id));
    const tiedPool = input.pool.map((player) =>
      owned.has(player.id)
        ? { ...player, expectedPoints: 1, conditionalPoints: 1, appearanceProbability: 1 }
        : { ...player, expectedPoints: 10, conditionalPoints: 10, appearanceProbability: 1, priceTenths: 45 },
    );
    const tiedSquad = tiedPool.filter((player) => owned.has(player.id));
    const baseline = optimizeFixedSquad(tiedSquad);
    const args = { squad: tiedSquad, pool: tiedPool, baseline, sellingPrices: input.sellingPrices, bankTenths: 20, freeTransfers: 5 };
    const first = recommendTransfers(args);
    const second = recommendTransfers({ ...args, pool: [...tiedPool].reverse() });
    expect(primaryPairs(second)).toEqual(primaryPairs(first));
    expect(second.groups.map((group) => group.options.map((option) => option.playerIn.id))).toEqual(first.groups.map((group) => group.options.map((option) => option.playerIn.id)));
  }, 30_000);

  it("keeps one-free-transfer behavior to one outgoing player and three choices", () => {
    const input = setup();
    const result = recommendTransfers({ ...input, pool: withStrongReplacements(input), bankTenths: 20, freeTransfers: 1 });
    expect(result.transfersUsed).toBe(1);
    expect(result.groups).toHaveLength(1);
    expect(result.groups[0].options.length).toBeLessThanOrEqual(3);
    expect(new Set(result.groups[0].options.map((item) => item.playerOut.id))).toEqual(
      new Set([result.groups[0].playerOut.id]),
    );
    expect(result.pointsHit).toBe(0);
  }, 30_000);

  it("only recommends a zero-free-transfer move after a positive four-point net gain", () => {
    const input = setup(1);
    const highPool = input.pool.map((player) =>
      player.id === "FWD_03"
        ? { ...player, expectedPoints: 50, conditionalPoints: 50, appearanceProbability: 1, priceTenths: 45 }
        : player,
    );
    const paid = recommendTransfers({ ...input, pool: highPool, bankTenths: 20, freeTransfers: 0 });
    expect(paid.transfersUsed).toBe(1);
    expect(paid.pointsHit).toBe(4);
    expect(paid.netImprovement).toBeCloseTo(paid.grossImprovement - 4, 10);
    expect(paid.netImprovement).toBeGreaterThan(0);

    const noPaid = recommendTransfers({ ...input, bankTenths: 20, freeTransfers: 0 });
    expect(noPaid.recommendNoTransfer).toBe(true);
    expect(noPaid.transfersUsed).toBe(0);
  }, 30_000);

  it("keeps bounded search practical for an official-sized 554-player pool", () => {
    const rows = projectionPool(135).slice(0, 554);
    const rawPool = optimizerPlayersFromProjections(rows);
    const squad = legalSquad(rows);
    const owned = new Set(squad.map((player) => player.id));
    const strongIds = new Set(["GKP_135", "DEF_135", "DEF_136", "MID_135", "FWD_135"]);
    const pool = rawPool.map((player) => {
      if (strongIds.has(player.id)) return { ...player, expectedPoints: 30, conditionalPoints: 30, appearanceProbability: 1, priceTenths: 45, team: `strong_${player.id}` };
      return owned.has(player.id) ? player : { ...player, priceTenths: 45 };
    });
    const baseline = optimizeFixedSquad(squad);
    const sellingPrices = Object.fromEntries(squad.map((player) => [player.id, player.priceTenths]));
    const started = performance.now();
    const result = recommendTransfers({ squad, pool, baseline, sellingPrices, bankTenths: 20, freeTransfers: 5 });
    const elapsed = performance.now() - started;
    expect(pool).toHaveLength(554);
    expect(result.shortlisted).toBe(true);
    expect(result.transfersUsed).toBeGreaterThan(0);
    expect(result.transfersUsed).toBeLessThanOrEqual(5);
    expect(result.groups.every((group) => group.options.length <= 3)).toBe(true);
    expect(result.searchedDepths).toEqual([0, 1, 2, 3, 4, 5]);
    expect(elapsed).toBeLessThan(5_000);
  }, 35_000);
});
