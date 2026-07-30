import { describe, expect, it } from "vitest";
import type { ProjectionRow } from "./data";
import { findPlayers } from "./playerFinder";

function player(
  id: string,
  position: string,
  priceTenths: number,
  expectedPoints: number,
  appearance = 0.8,
  fivePlus = 0.25,
): ProjectionRow {
  return {
    stable_player_id: id,
    player: id,
    team: "team_arsenal",
    position,
    price_tenths: String(priceTenths),
    expected_points: String(expectedPoints),
    expected_minutes: "70",
    p_appearance: String(appearance),
    p_start: "0.7",
    prob_points_ge_5: String(fivePlus),
    status: "a",
    model_variant: "X2_TEAM_CONSTRAINED_SIM_M7",
  };
}

const rows = [
  player("gkp", "GKP", 50, 9),
  player("mid-1", "MID", 50, 8),
  player("mid-2", "MID", 60, 7),
  player("mid-3", "MID", 70, 6),
  player("mid-4", "MID", 80, 5),
  player("mid-5", "MID", 90, 4),
  player("mid-6", "MID", 100, 3),
];

describe("findPlayers", () => {
  it("returns no more than five players in the selected position", () => {
    const result = findPlayers(rows, "MID", "10.0");
    expect(result.error).toBeNull();
    expect(result.recommendations).toHaveLength(5);
    expect(result.recommendations.every(({ player: row }) => row.position === "MID")).toBe(true);
  });

  it("includes the exact budget boundary and excludes players above it", () => {
    const result = findPlayers(rows, "MID", "6.0");
    expect(result.recommendations.map(({ player: row }) => row.stable_player_id)).toEqual([
      "mid-1",
      "mid-2",
    ]);
  });

  it("excludes the replaced player and enforces that player's position", () => {
    const result = findPlayers(rows, "FWD", "10.0", "mid-1");
    expect(result.position).toBe("MID");
    expect(result.recommendations.map(({ player: row }) => row.stable_player_id)).not.toContain(
      "mid-1",
    );
    expect(result.recommendations.every(({ player: row }) => row.position === "MID")).toBe(true);
  });

  it("orders primarily by expected points", () => {
    const result = findPlayers(
      [player("cheap", "DEF", 40, 4), player("expensive", "DEF", 70, 6)],
      "DEF",
      "8.0",
    );
    expect(result.recommendations.map(({ player: row }) => row.stable_player_id)).toEqual([
      "expensive",
      "cheap",
    ]);
  });

  it("applies deterministic appearance, P(5+), price and identifier tie-breakers", () => {
    const tied = [
      player("z-id", "FWD", 60, 5, 0.9, 0.4),
      player("higher-appearance", "FWD", 80, 5, 0.95, 0.1),
      player("higher-five-plus", "FWD", 80, 5, 0.9, 0.5),
      player("a-id", "FWD", 60, 5, 0.9, 0.4),
      player("lower-price", "FWD", 50, 5, 0.9, 0.4),
    ];
    expect(
      findPlayers(tied, "FWD", "10.0").recommendations.map(
        ({ player: row }) => row.stable_player_id,
      ),
    ).toEqual(["higher-appearance", "higher-five-plus", "lower-price", "a-id", "z-id"]);
  });

  it("does not depend on incoming artifact row order", () => {
    const forward = findPlayers(rows, "MID", "10.0").recommendations.map(
      ({ player: row }) => row.stable_player_id,
    );
    const reversed = findPlayers(rows.slice().reverse(), "MID", "10.0").recommendations.map(
      ({ player: row }) => row.stable_player_id,
    );
    expect(reversed).toEqual(forward);
  });

  it("calculates value and replacement difference from full-precision values", () => {
    const source = player("source", "DEF", 50, 4.1234);
    const candidate = player("candidate", "DEF", 75, 6.9876);
    const result = findPlayers([source, candidate], "DEF", "7.5", "source");
    expect(result.recommendations[0].expectedPointsPerMillion).toBeCloseTo(6.9876 / 7.5, 12);
    expect(result.recommendations[0].expectedPointsDifference).toBeCloseTo(2.8642, 12);
  });

  it("uses an unavailable difference when no player is selected", () => {
    const result = findPlayers(rows, "GKP", "5.0");
    expect(result.recommendations[0].expectedPointsDifference).toBeNull();
  });

  it.each(["", " ", "text", "-1", "0", "7.55", " 7.5 "])(
    "rejects invalid budget %j clearly",
    (budget) => {
      const result = findPlayers(rows, "MID", budget);
      expect(result.recommendations).toEqual([]);
      expect(result.error).toMatch(/valid budget/i);
    },
  );

  it("handles invalid positions and unavailable replaced players clearly", () => {
    expect(findPlayers(rows, "ALL", "8.0").error).toBe("Choose a valid FPL position.");
    expect(findPlayers(rows, "MID", "8.0", "missing").error).toMatch(
      /unavailable in the current official forecast/i,
    );
  });

  it("returns no eligible players or fewer than five without error", () => {
    expect(findPlayers(rows, "FWD", "4.0").recommendations).toHaveLength(0);
    expect(findPlayers(rows, "GKP", "5.0").recommendations).toHaveLength(1);
  });

  it("excludes records invalid under the frontend forecast contract", () => {
    const invalid = player("invalid", "MID", 50, 20);
    invalid.p_appearance = "not-a-probability";
    const result = findPlayers([invalid, player("valid", "MID", 50, 5)], "MID", "5.0");
    expect(result.recommendations.map(({ player: row }) => row.stable_player_id)).toEqual([
      "valid",
    ]);
  });
});
