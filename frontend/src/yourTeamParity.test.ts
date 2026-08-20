import { describe, expect, it } from "vitest";
import fixture from "./fixtures/your_team_parity.json";
import { optimizeFixedSquad, type OptimizerPlayer } from "./yourTeamOptimizer";

type NumericFields = Pick<
  OptimizerPlayer,
  "expectedPoints" | "conditionalPoints" | "appearanceProbability"
>;

describe("Python and TypeScript D2 parity fixtures", () => {
  for (const parityCase of fixture.cases) {
    it(parityCase.name, () => {
      const all = (parityCase as { all?: Record<string, number> }).all ?? {};
      const overrides = parityCase.overrides as Record<string, Record<string, number>>;
      const players = fixture.base.map((row) => {
        const values = { ...row, ...all, ...(overrides[row.player_uid] ?? {}) };
        return {
          id: values.player_uid,
          name: values.player_name,
          team: values.player_team_uid,
          position: values.fpl_position,
          priceTenths: values.price_tenths,
          expectedPoints: values.expected_points,
          conditionalPoints: values.expected_points_given_appearance,
          appearanceProbability: values.p_appearance,
          projection: {} as OptimizerPlayer["projection"],
        } as OptimizerPlayer & NumericFields;
      });
      const result = optimizeFixedSquad(players);
      const expected = parityCase.expected;
      expect(result.decision.lineup).toEqual(expected.lineup);
      expect(result.decision.formation).toBe(expected.formation);
      expect(result.decision.bench[0]).toBe(expected.bench_goalkeeper);
      expect(result.decision.bench.slice(1)).toEqual(expected.outfield_bench);
      expect(result.decision.captain).toBe(expected.captain);
      expect(result.decision.viceCaptain).toBe(expected.vice_captain);
      expect(result.breakdown.expectedRealizedTotal).toBeCloseTo(expected.expected_realized_total, 8);
      expect(result.breakdown.expectedAutosubContribution).toBeCloseTo(expected.expected_autosub_contribution, 8);
      expect(result.breakdown.expectedCaptainBonus).toBeCloseTo(expected.expected_captain_bonus, 8);
      expect(result.breakdown.expectedViceCaptainContingency).toBeCloseTo(expected.expected_vice_captain_contingency, 8);
      expect(result.breakdown.expectedAutomaticSubstitutions).toBeCloseTo(expected.expected_automatic_substitutions, 8);
      expect(result.breakdown.probabilityUnreplacedStarter).toBeCloseTo(expected.probability_unreplaced_starter, 8);
    }, 20_000);
  }
});
