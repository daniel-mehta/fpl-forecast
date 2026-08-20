import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { FrontendData } from "./data";
import { TransferResults, YourTeamPage, type Calculation } from "./YourTeam";
import { legalSquad, projectionPool } from "./yourTeamTestFixtures";
import { optimizerPlayersFromProjections, optimizeFixedSquad } from "./yourTeamOptimizer";
import { YOUR_TEAM_STORAGE_KEY } from "./yourTeamStorage";
import type { TransferOption, TransferRecommendationResult } from "./yourTeamTransfers";

function data(overrides: Partial<FrontendData> = {}): FrontendData {
  return {
    status: { state: "SUCCEEDED", target_season: "2026-27", target_gameweek: 1, run_id: "run-a" },
    projections: projectionPool(1),
    squad: [],
    lineup: [],
    comparison: [],
    freshness: { source_mode: "official_current_season", generated_at: "2026-08-20T10:00:00Z", stale: false },
    manifest: { target_season: "2026-27", target_gameweek: 1, run_id: "run-a" },
    ...overrides,
  };
}

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

function combinedCalculation(): Calculation {
  const players = optimizerPlayersFromProjections(projectionPool(1));
  const squad = legalSquad(projectionPool(1));
  const baseline = optimizeFixedSquad(squad);
  const pairs = [
    [squad.find((player) => player.id === "GKP_01")!, players.find((player) => player.id === "GKP_02")!],
    [squad.find((player) => player.id === "DEF_04")!, players.find((player) => player.id === "DEF_05")!],
  ] as const;
  const options: TransferOption[] = pairs.map(([playerOut, playerIn]) => ({
    playerOut,
    playerIn,
    incomingPriceTenths: playerIn.priceTenths,
    outgoingSellingPriceTenths: playerOut.priceTenths,
    bankRemainingTenths: 10,
    expectedRealizedAfter: baseline.breakdown.expectedRealizedTotal + 2,
    grossImprovement: 2,
    pointsHit: 0,
    netImprovement: 2,
    result: baseline,
    startingXiChanged: true,
    captainChanged: false,
    viceCaptainChanged: false,
    benchOrderChanged: true,
    primary: true,
  }));
  const transfers: TransferRecommendationResult = {
    groups: options.map((option) => ({ playerOut: option.playerOut, outgoingSellingPriceTenths: option.outgoingSellingPriceTenths, options: [option] })),
    primaryTransfers: options,
    recommendNoTransfer: false,
    freeTransfersAvailable: 5,
    transfersUsed: 2,
    freeTransfersRolled: 3,
    expectedRealizedBefore: baseline.breakdown.expectedRealizedTotal,
    expectedRealizedAfter: baseline.breakdown.expectedRealizedTotal + 2,
    grossImprovement: 2,
    pointsHit: 0,
    netImprovement: 2,
    bankRemainingTenths: 10,
    result: baseline,
    exactPlanEvaluations: 5,
    candidatePlanCount: 120,
    searchedDepths: [0, 1, 2, 3, 4, 5],
    shortlisted: true,
  };
  return { baseline, transfers, elapsedMilliseconds: 1200 };
}

describe("Your Team page", () => {
  it("fails closed when the frozen optimizer contract is missing", () => {
    const malformed = data({ projections: projectionPool(0).map((row) => ({ ...row, expected_points_given_appearance: undefined })) });
    render(<YourTeamPage data={malformed} />);
    expect(screen.getByRole("alert")).toHaveTextContent("missing authoritative optimizer fields");
    expect(screen.queryByRole("button", { name: "Optimize lineup and transfers" })).not.toBeInTheDocument();
  });

  it("supports searchable manual selection and displays all required player metrics", async () => {
    const user = userEvent.setup();
    render(<YourTeamPage data={data()} />);
    await user.type(screen.getByLabelText("Search current forecast players"), "GKP Player 0");
    const results = screen.getByRole("list", { name: "Player search results" });
    await user.click(within(results).getByRole("listitem"));
    const row = screen.getByRole("rowheader", { name: "GKP Player 0" }).closest("tr")!;
    expect(within(row).getByText("ARS (H)")).toBeInTheDocument();
    expect(within(row).getByLabelText("GKP Player 0 selling price")).toHaveValue(4.5);
    expect(row.querySelector('[data-label="Expected points"]')).toBeInTheDocument();
    expect(row.querySelector('[data-label="Expected minutes"]')).toBeInTheDocument();
    expect(row.querySelector('[data-label="Appearance"]')).toBeInTheDocument();
    expect(row.querySelector('[data-label="Start"]')).toBeInTheDocument();
    expect(row.querySelector('[data-label="At least five points"]')).toBeInTheDocument();
    expect(screen.getByText("GKP 1/2")).toBeInTheDocument();
  });

  it("identifies the exact season, Gameweek, run and local-only storage promise", () => {
    render(<YourTeamPage data={data()} />);
    expect(screen.getByText("2026-27, Gameweek 1")).toBeInTheDocument();
    expect(screen.getByText("run-a")).toBeInTheDocument();
    expect(screen.getByText(/never sent anywhere/)).toBeInTheDocument();
  });

  it("persists the entered free-transfer count", async () => {
    const storage = memoryStorage();
    const original = Object.getOwnPropertyDescriptor(window, "localStorage");
    Object.defineProperty(window, "localStorage", { configurable: true, value: storage });
    const user = userEvent.setup();
    const first = render(<YourTeamPage data={data()} />);
    await user.clear(screen.getByLabelText("Free transfers"));
    await user.type(screen.getByLabelText("Free transfers"), "5");
    await waitFor(() => expect(storage.length).toBe(1));
    first.unmount();
    render(<YourTeamPage data={data()} />);
    await waitFor(() => expect(screen.getByLabelText("Free transfers")).toHaveValue(5));
    if (original) Object.defineProperty(window, "localStorage", original);
  });

  it("paints a calculating state before running exact optimization", async () => {
    const storage = memoryStorage();
    const rows = projectionPool(1);
    const squad = legalSquad(rows);
    storage.setItem(YOUR_TEAM_STORAGE_KEY, JSON.stringify({
      schemaVersion: 1,
      forecastIdentity: {
        season: "2026-27",
        gameweek: "1",
        runId: "run-a",
        playerIds: rows.map((row) => row.stable_player_id),
      },
      playerIds: squad.map((player) => player.id),
      sellingPrices: Object.fromEntries(squad.map((player) => [player.id, player.priceTenths])),
      bankTenths: 0,
      freeTransfers: 1,
    }));
    const original = Object.getOwnPropertyDescriptor(window, "localStorage");
    Object.defineProperty(window, "localStorage", { configurable: true, value: storage });
    render(<YourTeamPage data={data({ projections: rows })} />);
    await waitFor(() => expect(screen.getByText(/15\/15 selected/)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Optimize lineup and transfers" }));
    expect(screen.getByRole("button", { name: "Calculating…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("Calculating the exact lineup");
    await waitFor(() => expect(screen.getByRole("heading", { name: "Optimized lineup" })).toBeInTheDocument(), {
      timeout: 10_000,
    });
    if (original) Object.defineProperty(window, "localStorage", original);
  }, 15_000);

  it.each([1280, 390])("renders the combined plan and distinct outgoing groups at %ipx", (width) => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: width });
    render(<TransferResults calculation={combinedCalculation()} />);
    expect(screen.getByText("Use 2 of 5 free transfers and roll 3.")).toBeInTheDocument();
    expect(screen.getAllByText("Primary option")).toHaveLength(2);
    expect(screen.getByRole("heading", { name: "GKP Player 1 out" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "DEF Player 4 out" })).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("bounded-shortlist plan");
  });
});
