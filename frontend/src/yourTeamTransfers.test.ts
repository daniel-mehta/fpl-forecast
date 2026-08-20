import { describe, expect, it } from "vitest";
import { legalSquad, projectionPool } from "./yourTeamTestFixtures";
import { optimizerPlayersFromProjections, optimizeFixedSquad } from "./yourTeamOptimizer";
import { recommendSingleTransfers } from "./yourTeamTransfers";

function setup(extras = 2) {
  const rows = projectionPool(extras);
  const pool = optimizerPlayersFromProjections(rows);
  const squad = legalSquad(rows);
  const baseline = optimizeFixedSquad(squad);
  const sellingPrices = Object.fromEntries(squad.map((player) => [player.id, player.priceTenths]));
  return { pool, squad, baseline, sellingPrices };
}

describe("single-transfer recommendations", () => {
  it("returns legal same-position affordable transfers and exactly five results", () => {
    const input = setup(3);
    const result = recommendSingleTransfers({ ...input, bankTenths: 20, freeTransfers: 1 });
    expect(result.recommendations).toHaveLength(5);
    for (const recommendation of result.recommendations) {
      expect(recommendation.playerIn.position).toBe(recommendation.playerOut.position);
      expect(recommendation.bankRemainingTenths).toBeGreaterThanOrEqual(0);
      expect(recommendation.pointsHit).toBe(0);
    }
    expect(result.recommendations.some((item) => item.benchOrderChanged)).toBe(true);
  }, 20_000);

  it("rejects insufficient bank and club-limit violations before exact evaluation", () => {
    const input = setup(1);
    const ownedClub = input.squad[0].team;
    const pool = input.pool.map((player) =>
      input.squad.some((owned) => owned.id === player.id)
        ? player
        : { ...player, priceTenths: 1_000, team: ownedClub },
    );
    const result = recommendSingleTransfers({ ...input, pool, bankTenths: 0, freeTransfers: 1 });
    expect(result.recommendations).toHaveLength(0);
    expect(result.legalCandidateCount).toBe(0);
  }, 20_000);

  it("uses editable selling prices for affordability", () => {
    const input = setup(1);
    const expensive = input.pool.find((player) => !input.squad.some((owned) => owned.id === player.id))!;
    const outgoing = input.squad.find((player) => player.position === expensive.position)!;
    expensive.priceTenths = outgoing.priceTenths + 15;
    const poor = recommendSingleTransfers({ ...input, bankTenths: 0, freeTransfers: 1 });
    const richSelling = { ...input.sellingPrices, [outgoing.id]: outgoing.priceTenths + 15 };
    const rich = recommendSingleTransfers({ ...input, sellingPrices: richSelling, bankTenths: 0, freeTransfers: 1 });
    expect(poor.recommendations.some((item) => item.playerIn.id === expensive.id && item.playerOut.id === outgoing.id)).toBe(false);
    expect(rich.recommendations.some((item) => item.playerIn.id === expensive.id && item.playerOut.id === outgoing.id)).toBe(true);
  }, 20_000);

  it("applies a four-point hit only when no free transfer is available", () => {
    const input = setup(1);
    const free = recommendSingleTransfers({ ...input, bankTenths: 20, freeTransfers: 1 });
    const hit = recommendSingleTransfers({ ...input, bankTenths: 20, freeTransfers: 0 });
    expect(free.recommendations.every((item) => item.pointsHit === 0)).toBe(true);
    expect(hit.recommendations.every((item) => item.pointsHit === 4)).toBe(true);
    expect(hit.recommendations[0].netImprovement).toBeCloseTo(hit.recommendations[0].grossImprovement - 4, 10);
  }, 20_000);

  it("returns fewer than five when fewer legal candidates exist", () => {
    const input = setup(0);
    const result = recommendSingleTransfers({ ...input, bankTenths: 0, freeTransfers: 1 });
    expect(result.recommendations).toHaveLength(0);
  });

  it("is deterministic under tied candidates and recommends no transfer for negative net value", () => {
    const input = setup(1);
    const tiedPool = input.pool.map((player) => ({ ...player, expectedPoints: 1, conditionalPoints: 1, appearanceProbability: 1 }));
    const tiedSquad = tiedPool.filter((player) => input.squad.some((owned) => owned.id === player.id));
    const baseline = optimizeFixedSquad(tiedSquad);
    const args = { squad: tiedSquad, pool: tiedPool, baseline, sellingPrices: input.sellingPrices, bankTenths: 20, freeTransfers: 0 };
    const first = recommendSingleTransfers(args);
    const second = recommendSingleTransfers({ ...args, pool: [...tiedPool].reverse() });
    expect(second.recommendations.map((item) => `${item.playerOut.id}|${item.playerIn.id}`)).toEqual(
      first.recommendations.map((item) => `${item.playerOut.id}|${item.playerIn.id}`),
    );
    expect(first.recommendNoTransfer).toBe(true);
  }, 20_000);

  it("re-optimization can change formation and captain", () => {
    const input = setup(1);
    const squad = input.squad.map((player) =>
      player.id === "FWD_02"
        ? { ...player, expectedPoints: 0, conditionalPoints: 0, appearanceProbability: 1 }
        : player,
    );
    const pool = input.pool.map((player) =>
      player.id === "FWD_03"
        ? { ...player, expectedPoints: 50, conditionalPoints: 50, appearanceProbability: 1, priceTenths: 45 }
        : squad.find((owned) => owned.id === player.id) ?? player,
    );
    const baseline = optimizeFixedSquad(squad);
    const result = recommendSingleTransfers({ ...input, squad, baseline, pool, bankTenths: 20, freeTransfers: 1 });
    const strongest = result.recommendations[0];
    expect(strongest.playerIn.id).toBe("FWD_03");
    expect(strongest.captainChanged).toBe(true);
    expect(strongest.result.decision.formation).not.toBe(baseline.decision.formation);
  }, 20_000);

  it("keeps exact shortlisting practical for an official-sized 554-player pool", () => {
    const rows = projectionPool(135).slice(0, 554);
    const rawPool = optimizerPlayersFromProjections(rows);
    const squad = legalSquad(rows);
    const owned = new Set(squad.map((player) => player.id));
    const pool = rawPool.map((player) => (owned.has(player.id) ? player : { ...player, priceTenths: 45 }));
    const baseline = optimizeFixedSquad(squad);
    const sellingPrices = Object.fromEntries(squad.map((player) => [player.id, player.priceTenths]));
    const started = performance.now();
    const result = recommendSingleTransfers({ squad, pool, baseline, sellingPrices, bankTenths: 20, freeTransfers: 1 });
    const elapsed = performance.now() - started;
    expect(pool).toHaveLength(554);
    expect(result.legalCandidateCount).toBeGreaterThan(100);
    expect(result.legalCandidatesEvaluated).toBe(5);
    expect(result.recommendations).toHaveLength(5);
    expect(elapsed).toBeLessThan(15_000);
  }, 20_000);
});
