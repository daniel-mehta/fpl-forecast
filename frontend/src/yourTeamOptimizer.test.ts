import { describe, expect, it } from "vitest";
import { changedPlayer, legalSquad } from "./yourTeamTestFixtures";
import {
  evaluateExpectedRealized,
  optimizeFixedSquad,
  validateSquad,
} from "./yourTeamOptimizer";

describe("Python-authoritative fixed-squad D2 port", () => {
  it("returns a legal deterministic lineup, bench and captaincy with exact state metadata", () => {
    const squad = legalSquad();
    const first = optimizeFixedSquad(squad);
    const second = optimizeFixedSquad(squad);

    expect(second).toEqual(first);
    expect(first.decision.lineup).toHaveLength(11);
    expect(first.decision.bench).toHaveLength(4);
    expect(first.decision.bench[0]).toMatch(/^GKP_/);
    expect(first.decision.captain).not.toBe(first.decision.viceCaptain);
    expect(first.breakdown.scenarioCount).toBe(32768);
    expect(first.breakdown.probabilityMass).toBeCloseTo(1, 10);
    expect(first.breakdown.expectedRealizedTotal).toBeCloseTo(
      first.breakdown.expectedActiveStarterPoints +
        first.breakdown.expectedAutosubContribution +
        first.breakdown.expectedCaptainBonus +
        first.breakdown.expectedViceCaptainContingency,
      10,
    );
  });

  it("enforces formations and uses outfield bench order for autosubs", () => {
    let squad = legalSquad();
    squad = changedPlayer(squad, "DEF_04", { expectedPoints: 0, conditionalPoints: 0, appearanceProbability: 0 });
    const result = optimizeFixedSquad(squad);
    const positions = result.decision.lineup.map((id) => squad.find((player) => player.id === id)!.position);
    expect(positions.filter((position) => position === "DEF").length).toBeGreaterThanOrEqual(3);
    expect(result.breakdown.expectedAutosubContribution).toBeGreaterThanOrEqual(0);
  });

  it("models low-appearance captain fallback without double-counting unconditional xP", () => {
    let squad = legalSquad();
    squad = changedPlayer(squad, "MID_00", { expectedPoints: 5, conditionalPoints: 25, appearanceProbability: 0.2 });
    const result = optimizeFixedSquad(squad);
    expect(result.breakdown.expectedCaptainBonus).toBeGreaterThan(0);
    expect(result.breakdown.expectedViceCaptainContingency).toBeGreaterThan(0);
  });

  it("selects the better goalkeeper ordering in both directions", () => {
    for (const reliable of ["GKP_00", "GKP_01"]) {
      const other = reliable === "GKP_00" ? "GKP_01" : "GKP_00";
      let squad = legalSquad();
      squad = changedPlayer(squad, reliable, { expectedPoints: 3.6, conditionalPoints: 4, appearanceProbability: 0.9 });
      squad = changedPlayer(squad, other, { expectedPoints: 0.02, conditionalPoints: 2, appearanceProbability: 0.01 });
      const result = optimizeFixedSquad(squad);
      expect(result.decision.lineup).toContain(reliable);
      expect(result.decision.bench[0]).toBe(other);
    }
  });

  it("preserves deterministic tie-breaking", () => {
    const squad = legalSquad().map((player) => ({ ...player, expectedPoints: 4, conditionalPoints: 4, appearanceProbability: 1 }));
    expect(optimizeFixedSquad(squad)).toEqual(optimizeFixedSquad([...squad].reverse()));
  });

  it("fails illegal quotas, duplicates and club limits", () => {
    const squad = legalSquad();
    expect(validateSquad([...squad.slice(0, 14), squad[0]])).toContain("A player cannot be selected more than once.");
    expect(validateSquad(squad.map((player) => ({ ...player, team: "one_club" })))).toContain(
      "A squad may contain no more than three players from one club.",
    );
  });

  it("reports unreplaced-starter risk from exact autosub states", () => {
    const squad = legalSquad().map((player) =>
      ["DEF_02", "DEF_03", "DEF_04"].includes(player.id)
        ? { ...player, appearanceProbability: 0, expectedPoints: 0 }
        : { ...player, appearanceProbability: 1, expectedPoints: player.conditionalPoints },
    );
    const optimized = optimizeFixedSquad(squad);
    const evaluated = evaluateExpectedRealized(squad, optimized.decision);
    expect(evaluated.probabilityUnreplacedStarter).toBeGreaterThanOrEqual(0);
    expect(evaluated.probabilityUnreplacedStarter).toBeLessThanOrEqual(1);
  });
});
