import { describe, expect, it } from "vitest";
import { restoreYourTeam, saveYourTeam, YOUR_TEAM_STORAGE_KEY, type ForecastIdentity } from "./yourTeamStorage";

const identity: ForecastIdentity = { season: "2026-27", gameweek: "1", runId: "run-a", playerIds: ["a", "b"] };

function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() { return values.size; },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    removeItem: (key) => { values.delete(key); },
    setItem: (key, value) => { values.set(key, value); },
  };
}

describe("Your Team persistence", () => {
  it("restores saved inputs for the identical frozen player identity", () => {
    const storage = memoryStorage();
    saveYourTeam(storage, { schemaVersion: 1, forecastIdentity: identity, playerIds: ["a"], sellingPrices: { a: 50 }, bankTenths: 3, freeTransfers: 2 });
    expect(restoreYourTeam(storage, identity)?.bankTenths).toBe(3);
  });

  it("invalidates saved data after season, gameweek, run or player identity changes", () => {
    const storage = memoryStorage();
    for (const changed of [
      { ...identity, season: "2027-28" },
      { ...identity, gameweek: "2" },
      { ...identity, runId: "run-b" },
      { ...identity, playerIds: ["a", "c"] },
    ]) {
      saveYourTeam(storage, { schemaVersion: 1, forecastIdentity: identity, playerIds: ["a"], sellingPrices: { a: 50 }, bankTenths: 0, freeTransfers: 1 });
      expect(restoreYourTeam(storage, changed)).toBeNull();
      expect(storage.getItem(YOUR_TEAM_STORAGE_KEY)).toBeNull();
    }
  });
});
