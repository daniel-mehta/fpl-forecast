import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { FrontendData } from "./data";
import { YourTeamPage } from "./YourTeam";
import { projectionPool } from "./yourTeamTestFixtures";

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
});
