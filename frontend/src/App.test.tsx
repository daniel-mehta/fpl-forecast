import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "./App";
import type { FrontendData, ProjectionRow, SquadRow } from "./data";

function projection(
  player: string,
  team: string,
  position: string,
  price: number,
  points: number,
): ProjectionRow {
  return {
    stable_player_id: player,
    player,
    team,
    position,
    price_tenths: String(price),
    opponent_display: "CHE (H)",
    expected_points: String(points),
    expected_minutes: "75",
    p_appearance: "0.9",
    p_start: "0.8",
    prob_points_ge_5: "0.3",
    status: "a",
    model_variant: "X2_TEAM_CONSTRAINED_SIM_M7",
  };
}

function squadRow(
  id: string,
  role: string,
  benchOrder = "",
  position = "MID",
): SquadRow {
  return {
    player_uid: id,
    player_name: id,
    player_team_uid: "team_arsenal",
    fpl_position: position,
    price_tenths: "50",
    opponent_display: "CHE (H)",
    expected_points: "3.4",
    selected_role: role,
    bench_order: benchOrder,
  };
}

function fixtureData(): FrontendData {
  const squad = [
    squadRow("Captain", "captain"),
    squadRow("Vice", "vice_captain"),
    ...Array.from({ length: 9 }, (_, index) => squadRow(`Starter ${index + 1}`, "starter")),
    squadRow("Bench goalkeeper player", "squad", "1", "GKP"),
    squadRow("First bench player", "squad", "2", "DEF"),
    squadRow("Second bench player", "squad", "3", "MID"),
    squadRow("Third bench player", "squad", "4", "FWD"),
  ];
  return {
    status: {
      state: "SUCCEEDED",
      run_id: "official-run-123",
      target_season: "2026-27",
      target_gameweek: 1,
    },
    projections: [
      projection("Alpha", "team_arsenal", "GKP", 55, 4),
      projection("Bravo", "team_liverpool", "MID", 125, 8),
      projection("Charlie", "team_arsenal", "MID", 45, 2),
    ],
    squad,
    lineup: [
      {
        captain: "Captain",
        vice_captain: "Vice",
        lineup: "",
        bench: "",
        formation: "3-5-2",
        cost_tenths: "985",
        bank_tenths: "15",
        optimizer_variant: "D2_EXPECTED_REALIZED_POINTS",
        expected_realized_total: "50.76",
        expected_autosub_contribution: "2.45",
        expected_captain_bonus: "5.03",
        expected_vice_captain_contingency: "0.24",
        expected_automatic_substitutions: "0.88",
        evaluated_squads: "6135",
        probability_unreplaced_starter: "0.16",
        solver_status: "heuristic_feasible",
        termination_reason: "configured_iteration_bound_reached",
        lineup_refinement_status: "single_change_local_optimum",
      },
    ],
    comparison: [],
    freshness: {
      generated_at: "2026-07-24T19:05:00Z",
      source_mode: "official_current_season",
      stale: false,
      official_snapshots: {
        bootstrap: { retrieved_at: "2026-07-24T19:02:00Z" },
      },
    },
    manifest: {},
  };
}

function projectionFixture(count = 60): FrontendData {
  const data = fixtureData();
  data.projections = Array.from({ length: count }, (_, index) =>
    projection(
      `Projection ${String(index + 1).padStart(3, "0")}`,
      index % 2 === 0 ? "team_arsenal" : "team_liverpool",
      index % 4 === 0 ? "GKP" : index % 4 === 1 ? "DEF" : index % 4 === 2 ? "MID" : "FWD",
      40 + (index % 10) * 5,
      count - index,
    ),
  );
  return data;
}

describe("dashboard interactions", () => {
  it("paginates sorted projections with working navigation and a final partial page", async () => {
    const user = userEvent.setup();
    render(<App initialData={projectionFixture()} />);
    const section = screen.getByRole("heading", { name: "Player projections" }).closest("section");
    expect(section).not.toBeNull();
    const projectionsTable = within(section!).getByRole("table");

    expect(within(section!).getByText("Showing 1–25 of 60")).toBeInTheDocument();
    expect(within(projectionsTable).getAllByRole("rowheader")).toHaveLength(25);
    expect(within(projectionsTable).getByRole("rowheader", { name: "Projection 001" })).toBeInTheDocument();
    expect(within(projectionsTable).queryByRole("rowheader", { name: "Projection 026" })).not.toBeInTheDocument();

    const topNavigation = within(section!).getByRole("navigation", {
      name: "Top projection pagination",
    });
    expect(within(topNavigation).getByRole("button", { name: "Previous" })).toBeDisabled();
    await user.click(within(topNavigation).getByRole("button", { name: "Next" }));
    expect(within(section!).getByText("Showing 26–50 of 60")).toBeInTheDocument();
    expect(within(projectionsTable).queryByRole("rowheader", { name: "Projection 001" })).not.toBeInTheDocument();
    expect(within(projectionsTable).getByRole("rowheader", { name: "Projection 026" })).toBeInTheDocument();

    await user.selectOptions(within(topNavigation).getByLabelText("Top projection page"), "3");
    expect(within(section!).getByText("Showing 51–60 of 60")).toBeInTheDocument();
    expect(within(projectionsTable).getAllByRole("rowheader")).toHaveLength(10);
    expect(within(topNavigation).getByRole("button", { name: "Next" })).toBeDisabled();
  });

  it("supports predefined and custom page sizes while retaining the last valid size", async () => {
    const user = userEvent.setup();
    render(<App initialData={projectionFixture()} />);
    const section = screen.getByRole("heading", { name: "Player projections" }).closest("section");
    expect(section).not.toBeNull();
    const pageSize = within(section!).getByLabelText("Players per page");

    await user.selectOptions(pageSize, "10");
    expect(within(section!).getByText("Showing 1–10 of 60")).toBeInTheDocument();
    await user.selectOptions(pageSize, "custom");
    const custom = within(section!).getByLabelText("Custom page size");
    const apply = within(section!).getByRole("button", { name: "Apply" });

    for (const invalidValue of ["abc", "0", "-1", "1.5", "555"]) {
      await user.clear(custom);
      await user.type(custom, invalidValue);
      await user.click(apply);
      expect(within(section!).getByRole("alert")).toHaveTextContent(
        "Enter a whole number from 1 to 554.",
      );
      expect(within(section!).getByText("Showing 1–10 of 60")).toBeInTheDocument();
    }
    await user.clear(custom);
    await user.click(apply);
    expect(within(section!).getByRole("alert")).toBeInTheDocument();
    expect(within(section!).getByText("Showing 1–10 of 60")).toBeInTheDocument();

    await user.type(custom, "1");
    await user.click(apply);
    expect(within(section!).getByText("Showing 1–1 of 60")).toBeInTheDocument();
    await user.clear(custom);
    await user.type(custom, "554");
    await user.click(apply);
    expect(within(section!).getByText("Showing 1–60 of 60")).toBeInTheDocument();
  });

  it("paginates after filtering and sorting, resets pages, and exposes mobile card labels", async () => {
    const user = userEvent.setup();
    render(<App initialData={projectionFixture()} />);
    const section = screen.getByRole("heading", { name: "Player projections" }).closest("section");
    expect(section).not.toBeNull();
    const topNavigation = within(section!).getByRole("navigation", {
      name: "Top projection pagination",
    });

    await user.click(within(topNavigation).getByRole("button", { name: "Next" }));
    expect(within(topNavigation).getByLabelText("Top projection page")).toHaveValue("2");
    await user.click(within(section!).getByRole("button", { name: /Expected points:/ }));
    expect(within(topNavigation).getByLabelText("Top projection page")).toHaveValue("1");
    expect(within(section!).getByRole("rowheader", { name: "Projection 060" })).toBeInTheDocument();

    await user.click(within(topNavigation).getByRole("button", { name: "Next" }));
    await user.type(within(section!).getByLabelText("Search players or teams"), "Projection 001");
    expect(within(section!).getByText("Showing 1–1 of 1")).toBeInTheDocument();
    expect(within(topNavigation).getByLabelText("Top projection page")).toHaveValue("1");
    await user.click(within(section!).getByRole("button", { name: "Reset filters" }));
    expect(within(section!).getByText("Showing 1–25 of 60")).toBeInTheDocument();

    await user.click(within(topNavigation).getByRole("button", { name: "Next" }));
    await user.selectOptions(within(section!).getByLabelText("Position"), "GKP");
    expect(within(topNavigation).getByLabelText("Top projection page")).toHaveValue("1");
    expect(within(section!).getByText("Showing 1–15 of 15")).toBeInTheDocument();
    await user.click(within(section!).getByRole("button", { name: "Reset filters" }));

    await user.click(within(topNavigation).getByRole("button", { name: "Next" }));
    await user.selectOptions(within(section!).getByLabelText("Minimum price"), "80");
    expect(within(topNavigation).getByLabelText("Top projection page")).toHaveValue("1");
    expect(within(section!).getByText("Showing 1–12 of 12")).toBeInTheDocument();
    await user.click(within(section!).getByRole("button", { name: "Reset filters" }));

    const firstRow = within(section!).getByRole("rowheader", { name: "Projection 060" }).closest("tr");
    expect(firstRow).not.toBeNull();
    expect(firstRow!.querySelector('[data-label="Player"]')).toBeInTheDocument();
    expect(firstRow!.querySelector('[data-label="At least 5 points"]')).toBeInTheDocument();
  });

  it("shows a clear empty state when projection filters match no players", async () => {
    const user = userEvent.setup();
    render(<App initialData={projectionFixture()} />);
    const section = screen.getByRole("heading", { name: "Player projections" }).closest("section");
    expect(section).not.toBeNull();

    await user.type(within(section!).getByLabelText("Search players or teams"), "No such player");
    expect(within(section!).getByText("Showing 0 of 0")).toBeInTheDocument();
    expect(within(section!).getByRole("status")).toHaveTextContent(
      "No players match the current projection filters.",
    );
    expect(within(section!).queryByRole("table")).not.toBeInTheDocument();
    expect(
      within(section!).getAllByRole("button", { name: "Previous" }).every((button) =>
        button.hasAttribute("disabled"),
      ),
    ).toBe(true);
  });

  it("combines price, position and search filters, updates count, and resets", async () => {
    const user = userEvent.setup();
    render(<App initialData={fixtureData()} />);

    expect(screen.getByText("3 matching players")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Minimum price"), "55");
    expect(screen.getByText("2 matching players")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Maximum price"), "55");
    expect(screen.getByText("1 matching player")).toBeInTheDocument();
    const projectionsSection = screen
      .getByRole("heading", { name: "Player projections" })
      .closest("section");
    expect(projectionsSection).not.toBeNull();
    await user.selectOptions(within(projectionsSection!).getByLabelText("Position"), "GKP");
    await user.type(screen.getByLabelText("Search players or teams"), "arsenal");
    expect(screen.getByRole("rowheader", { name: "Alpha" })).toBeInTheDocument();
    expect(screen.queryByRole("rowheader", { name: "Bravo" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Reset filters" }));
    expect(screen.getByText("3 matching players")).toBeInTheDocument();
    expect(screen.getByLabelText("Minimum price")).toHaveValue("");
    expect(screen.getByLabelText("Maximum price")).toHaveValue("");
  });

  it("opens tooltips by click and keyboard, closes on Escape, and names controls", async () => {
    const user = userEvent.setup();
    render(<App initialData={fixtureData()} />);
    const expectedRealized = screen.getByRole("button", {
      name: "More information about Expected realized",
    });
    await user.click(expectedRealized);
    expect(screen.getByRole("tooltip")).toHaveTextContent(
      "Projected points after accounting for appearances",
    );
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    const autosub = screen.getByRole("button", {
      name: "More information about Autosub value",
    });
    await user.tab();
    expect(autosub).toHaveFocus();
    expect(screen.getByRole("tooltip")).toHaveTextContent("Expected points contributed by bench players");
    await user.click(
      screen.getByRole("button", { name: "More information about Captain bonus" }),
    );
    expect(screen.getAllByRole("tooltip")).toHaveLength(1);
    expect(screen.getByRole("tooltip")).toHaveTextContent("doubling the captain's score");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shows understandable timestamps, grouped squad roles, cost and bank", () => {
    render(<App initialData={fixtureData()} />);
    expect(
      screen.getByText("Official data retrieved Jul 24, 2026, 19:02 UTC"),
    ).toBeInTheDocument();
    expect(screen.getByText("Forecast generated Jul 24, 2026, 19:05 UTC")).toBeInTheDocument();
    expect(screen.queryByText(/Local\s+time/)).not.toBeInTheDocument();
    expect(
      screen.getByText("Official data validated and forecast published successfully."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Current publication")).not.toBeInTheDocument();

    const startingGroup = screen.getByRole("heading", { name: "Starting XI" }).closest("section");
    const benchGroup = screen.getByRole("heading", { name: "Bench" }).closest("section");
    expect(startingGroup).not.toBeNull();
    expect(benchGroup).not.toBeNull();
    expect(within(startingGroup!).getByRole("cell", { name: "Captain" })).toBeInTheDocument();
    expect(within(benchGroup!).getByText("Bench goalkeeper")).toBeInTheDocument();
    expect(within(benchGroup!).getByText("Bench 1")).toBeInTheDocument();
    expect(within(benchGroup!).getByText("Bench 2")).toBeInTheDocument();
    expect(within(benchGroup!).getByText("Bench 3")).toBeInTheDocument();
    expect(screen.getByText("£98.5m")).toBeInTheDocument();
    expect(screen.getByText("£1.5m")).toBeInTheDocument();
    expect(screen.getByLabelText("Optimizer expected-realized diagnostics")).toHaveClass(
      "optimizer-summary",
    );
    expect(screen.queryByText(/DEMO DATA/)).not.toBeInTheDocument();
  });

  it("explains the status summary with accessible tooltips", async () => {
    const user = userEvent.setup();
    render(<App initialData={fixtureData()} />);
    const status = screen.getByRole("heading", { name: "Status summary" }).closest("section");
    expect(status).not.toBeNull();

    for (const label of ["Operational state", "Data freshness", "Forecast status"]) {
      expect(
        within(status!).getByRole("button", { name: `More information about ${label}` }),
      ).toBeInTheDocument();
    }

    await user.click(
      within(status!).getByRole("button", {
        name: "More information about Forecast status",
      }),
    );
    expect(screen.getByRole("tooltip")).toHaveTextContent(
      "not a measure of forecast accuracy",
    );
  });

  it("falls back accurately when publication timestamps are missing", () => {
    const data = fixtureData();
    data.freshness = { source_mode: "official_current_season", stale: false };
    render(<App initialData={data} />);
    expect(screen.getByText("Timestamp not available")).toBeInTheDocument();
  });

  it("falls back accurately when publication timestamps are invalid", () => {
    const data = fixtureData();
    data.freshness = {
      generated_at: "not-a-timestamp",
      source_mode: "official_current_season",
      stale: false,
      official_snapshots: {
        bootstrap: { retrieved_at: "also-invalid" },
      },
    };
    render(<App initialData={data} />);
    expect(screen.getByText("Timestamp not available")).toBeInTheDocument();
    expect(screen.queryByText(/UTC/)).not.toBeInTheDocument();
  });

  it("shows Player Finder results with official forecast identity and preserves dashboard views", () => {
    render(<App initialData={fixtureData()} />);
    const finder = screen.getByRole("heading", { name: "Player Finder" }).closest("section");
    expect(finder).not.toBeNull();
    expect(within(finder!).getByText("Gameweek 1")).toBeInTheDocument();
    expect(within(finder!).getByText("Jul 24, 2026, 19:05 UTC")).toBeInTheDocument();
    expect(within(finder!).getByText("official-run-123")).toBeInTheDocument();
    expect(
      within(finder!).getByText(/recommendations use only the current official forecast/i),
    ).toBeInTheDocument();
    expect(within(finder!).getAllByText("N/A").length).toBeGreaterThan(0);
    expect(
      within(finder!).getByText(/difference is N\/A until a player is selected/i),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Recommended squad" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Player projections" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Model comparison" })).toBeInTheDocument();
  });

  it("explains Player Finder controls and forecast identity with accessible tooltips", async () => {
    const user = userEvent.setup();
    render(<App initialData={fixtureData()} />);
    const finder = screen.getByRole("heading", { name: "Player Finder" }).closest("section");
    expect(finder).not.toBeNull();

    for (const label of [
      "Position",
      "Maximum budget (£m)",
      "Optional player being replaced",
      "Forecast",
      "Generated",
      "Run ID",
    ]) {
      expect(
        within(finder!).getByRole("button", { name: `More information about ${label}` }),
      ).toBeInTheDocument();
    }

    await user.click(
      within(finder!).getByRole("button", {
        name: "More information about Maximum budget (£m)",
      }),
    );
    expect(screen.getByRole("tooltip")).toHaveTextContent(
      "Selecting a player fills their price",
    );
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("explains every Player Finder recommendation metric", async () => {
    const user = userEvent.setup();
    render(<App initialData={fixtureData()} />);
    const card = screen.getByRole("heading", { name: "Bravo" }).closest("article");
    expect(card).not.toBeNull();
    const metricLabels = card!.querySelectorAll(".finder-metric-label");
    expect(metricLabels).toHaveLength(6);
    metricLabels.forEach((label) => {
      expect(label.firstElementChild).toHaveClass("info-tooltip");
    });

    for (const label of [
      "Official price",
      "Expected points",
      "Appearance",
      "P(5+ points)",
      "Expected points / £1.0m",
      "Expected-points difference",
    ]) {
      expect(
        within(card!).getByRole("button", { name: `More information about ${label}` }),
      ).toBeInTheDocument();
    }

    await user.click(
      within(card!).getByRole("button", {
        name: "More information about Expected points / £1.0m",
      }),
    );
    expect(screen.getByRole("tooltip")).toHaveTextContent(
      "calculated before display rounding",
    );
  });

  it("validates budgets inline and shows no-match and fewer-than-five states", async () => {
    const user = userEvent.setup();
    render(<App initialData={fixtureData()} />);
    const finder = screen.getByRole("heading", { name: "Player Finder" }).closest("section");
    expect(finder).not.toBeNull();
    const budget = within(finder!).getByRole("textbox", { name: "Maximum budget (£m)" });

    await user.clear(budget);
    expect(within(finder!).getByRole("alert")).toHaveTextContent(/valid budget/i);
    expect(budget).toHaveAttribute("aria-invalid", "true");

    await user.type(budget, "4.0");
    expect(within(finder!).getByRole("status")).toHaveTextContent(/no eligible players/i);

    await user.clear(budget);
    await user.type(budget, "5.0");
    expect(within(finder!).getByText(/fewer than five eligible players/i)).toBeInTheDocument();
  });

  it("locks replacements to the selected player's position and displays signed differences", async () => {
    const user = userEvent.setup();
    render(<App initialData={fixtureData()} />);
    const finder = screen.getByRole("heading", { name: "Player Finder" }).closest("section");
    expect(finder).not.toBeNull();

    expect(within(finder!).queryByLabelText("Search players")).not.toBeInTheDocument();
    await user.click(
      within(finder!).getByRole("button", {
        name: "Optional player being replaced No player selected",
      }),
    );
    const replacementSearch = within(finder!).getByLabelText("Search players");
    expect(replacementSearch).toHaveFocus();
    await user.type(replacementSearch, "Charlie");
    const replacementList = within(finder!).getByRole("listbox", {
      name: "Replacement players",
    });
    expect(within(replacementList).getByRole("option", { name: /Charlie/ })).toBeInTheDocument();
    expect(within(replacementList).queryByRole("option", { name: /Bravo/ })).not.toBeInTheDocument();
    await user.click(within(replacementList).getByRole("option", { name: /Charlie/ }));

    const finderPosition = within(finder!).getByLabelText("Position");
    const budget = within(finder!).getByRole("textbox", { name: "Maximum budget (£m)" });
    expect(finderPosition).toBeDisabled();
    expect(finderPosition).toHaveValue("MID");
    expect(budget).toHaveValue("4.5");
    await user.clear(budget);
    await user.type(budget, "15.0");
    expect(budget).toHaveValue("15.0");
    expect(within(finder!).getByText(/recommendations are limited to midfielders/i)).toBeInTheDocument();
    expect(within(finder!).queryByRole("heading", { name: "Charlie" })).not.toBeInTheDocument();
    expect(within(finder!).getByText("+6.00")).toBeInTheDocument();
  });

  it("fails closed when the loaded data are not an official current forecast", () => {
    const data = fixtureData();
    data.freshness.source_mode = "mock";
    render(<App initialData={data} />);
    const finder = screen.getByRole("heading", { name: "Player Finder" }).closest("section");
    expect(finder).not.toBeNull();
    expect(within(finder!).getByRole("status")).toHaveTextContent(
      /unavailable until a current official forecast/i,
    );
  });
});
