import { describe, expect, it } from "vitest";
import type { ProjectionRow, SquadRow } from "./data";
import {
  filterProjections,
  groupSquad,
  latestOfficialRetrievalTimestamp,
  normalizePriceRange,
  paginateRows,
  squadRoleLabel,
  timestampLine,
} from "./dashboard";

function projection(
  player: string,
  team: string,
  position: string,
  price: number,
): ProjectionRow {
  return {
    stable_player_id: player,
    player,
    team,
    position,
    price_tenths: String(price),
    expected_points: "3",
    expected_minutes: "70",
    p_appearance: "0.9",
    p_start: "0.8",
    prob_points_ge_5: "0.2",
    status: "a",
    model_variant: "X2",
  };
}

const projections = [
  projection("Alpha", "team_arsenal", "GKP", 55),
  projection("Bravo", "team_liverpool", "MID", 125),
  projection("Charlie", "team_arsenal", "MID", 45),
];

describe("projection filters", () => {
  it("supports minimum, maximum and range filtering", () => {
    expect(
      filterProjections(projections, {
        search: "",
        position: "ALL",
        minPriceTenths: 55,
        maxPriceTenths: null,
      }).map((row) => row.player),
    ).toEqual(["Alpha", "Bravo"]);
    expect(
      filterProjections(projections, {
        search: "",
        position: "ALL",
        minPriceTenths: null,
        maxPriceTenths: 55,
      }).map((row) => row.player),
    ).toEqual(["Alpha", "Charlie"]);
    expect(
      filterProjections(projections, {
        search: "",
        position: "ALL",
        minPriceTenths: 50,
        maxPriceTenths: 60,
      }).map((row) => row.player),
    ).toEqual(["Alpha"]);
  });

  it("combines search, position and price", () => {
    expect(
      filterProjections(projections, {
        search: "arsenal",
        position: "GKP",
        minPriceTenths: 50,
        maxPriceTenths: 60,
      }).map((row) => row.player),
    ).toEqual(["Alpha"]);
  });

  it("prevents invalid ranges in either direction", () => {
    expect(
      normalizePriceRange("min", 80, {
        minPriceTenths: null,
        maxPriceTenths: 60,
      }),
    ).toEqual({ minPriceTenths: 80, maxPriceTenths: 80 });
    expect(
      normalizePriceRange("max", 40, {
        minPriceTenths: 50,
        maxPriceTenths: null,
      }),
    ).toEqual({ minPriceTenths: 40, maxPriceTenths: 40 });
  });
});

describe("projection pagination", () => {
  const rows = Array.from({ length: 60 }, (_, index) => `Player ${index + 1}`);

  it("returns non-overlapping pages and the correct final partial range", () => {
    const first = paginateRows(rows, 1, 25);
    const second = paginateRows(rows, 2, 25);
    const final = paginateRows(rows, 3, 25);

    expect(first.rows).toHaveLength(25);
    expect(first.rows[0]).toBe("Player 1");
    expect(first.rows[24]).toBe("Player 25");
    expect(first).toMatchObject({
      page: 1,
      totalRows: 60,
      totalPages: 3,
      rangeStart: 1,
      rangeEnd: 25,
    });
    expect(second.rows).toHaveLength(25);
    expect(second.rows[0]).toBe("Player 26");
    expect(second.rows).not.toEqual(expect.arrayContaining(first.rows));
    expect(final.rows).toHaveLength(10);
    expect(final).toMatchObject({ rangeStart: 51, rangeEnd: 60 });
  });

  it("clamps page requests, reports empty results, and rejects invalid sizes", () => {
    expect(paginateRows(rows, 99, 25).page).toBe(3);
    expect(paginateRows([], 3, 25)).toMatchObject({
      rows: [],
      page: 1,
      totalRows: 0,
      totalPages: 0,
      rangeStart: 0,
      rangeEnd: 0,
    });
    expect(() => paginateRows(rows, 1, 0)).toThrow(/positive integer/i);
    expect(() => paginateRows(rows, 1, 2.5)).toThrow(/positive integer/i);
  });
});

describe("publication and squad helpers", () => {
  it("selects the latest official retrieval time and renders it in UTC", () => {
    const freshness = {
      official_snapshots: {
        bootstrap: { retrieved_at: "2026-07-24T19:02:00Z" },
        fixtures: { retrieved_at: "2026-07-24T19:03:00Z" },
      },
    };
    const latest = latestOfficialRetrievalTimestamp(freshness);
    expect(latest).toBe("2026-07-24T19:03:00Z");
    expect(timestampLine("Official data retrieved", latest)).toBe(
      "Official data retrieved Jul 24, 2026, 19:03 UTC",
    );
    expect(timestampLine("Official data retrieved", latest)).not.toContain(
      ["Local", "time"].join(" "),
    );
    expect(timestampLine("Forecast generated", undefined)).toBeNull();
    expect(timestampLine("Forecast generated", "invalid")).toBeNull();
  });

  it("groups starters before the ordered bench and provides clear role labels", () => {
    const rows = [
      { player_uid: "starter", selected_role: "captain", bench_order: "" },
      { player_uid: "bench-2", selected_role: "squad", bench_order: "3" },
      { player_uid: "gkp", selected_role: "squad", bench_order: "1" },
      { player_uid: "bench-1", selected_role: "squad", bench_order: "2" },
    ] as SquadRow[];
    const grouped = groupSquad(rows);
    expect(grouped.starters.map((row) => row.player_uid)).toEqual(["starter"]);
    expect(grouped.bench.map((row) => row.player_uid)).toEqual(["gkp", "bench-1", "bench-2"]);
    expect(squadRoleLabel(grouped.bench[0])).toBe("Bench goalkeeper");
    expect(squadRoleLabel(grouped.bench[1])).toBe("Bench 1");
    expect(squadRoleLabel(grouped.starters[0])).toBe("Captain");
  });
});
