import type { ProjectionRow } from "./data";

export const OFFICIAL_POSITIONS = ["GKP", "DEF", "MID", "FWD"] as const;

export type OfficialPosition = (typeof OFFICIAL_POSITIONS)[number];

export interface PlayerFinderRecommendation {
  player: ProjectionRow;
  expectedPointsPerMillion: number;
  expectedPointsDifference: number | null;
}

export interface PlayerFinderResult {
  recommendations: PlayerFinderRecommendation[];
  position: OfficialPosition | null;
  replacedPlayer: ProjectionRow | null;
  error: string | null;
}

function finiteNumber(value: string): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function nonEmptyString(value: string): boolean {
  return typeof value === "string" && value.trim().length > 0;
}

export function isOfficialPosition(value: string): value is OfficialPosition {
  return OFFICIAL_POSITIONS.includes(value as OfficialPosition);
}

export function isValidForecastPlayer(row: ProjectionRow): boolean {
  const priceTenths = finiteNumber(row.price_tenths);
  const expectedPoints = finiteNumber(row.expected_points);
  const expectedMinutes = finiteNumber(row.expected_minutes);
  const appearance = finiteNumber(row.p_appearance);
  const start = finiteNumber(row.p_start);
  const pointsFivePlus = finiteNumber(row.prob_points_ge_5);
  return (
    nonEmptyString(row.stable_player_id) &&
    nonEmptyString(row.player) &&
    nonEmptyString(row.team) &&
    isOfficialPosition(row.position) &&
    priceTenths !== null &&
    Number.isInteger(priceTenths) &&
    priceTenths > 0 &&
    expectedPoints !== null &&
    expectedPoints >= 0 &&
    expectedMinutes !== null &&
    expectedMinutes >= 0 &&
    expectedMinutes <= 90 &&
    appearance !== null &&
    appearance >= 0 &&
    appearance <= 1 &&
    start !== null &&
    start >= 0 &&
    start <= appearance &&
    pointsFivePlus !== null &&
    pointsFivePlus >= 0 &&
    pointsFivePlus <= 1
  );
}

export function validForecastPlayers(rows: ProjectionRow[]): ProjectionRow[] {
  return rows
    .filter(isValidForecastPlayer)
    .slice()
    .sort((left, right) => {
      const nameDifference = left.player.localeCompare(right.player, undefined, {
        sensitivity: "base",
      });
      return nameDifference || left.stable_player_id.localeCompare(right.stable_player_id);
    });
}

function parseBudgetMillions(value: string): number | null {
  if (!/^\d+(?:\.\d)?$/.test(value)) {
    return null;
  }
  const millions = Number(value);
  return Number.isFinite(millions) && millions > 0 ? millions : null;
}

function comparePlayers(left: ProjectionRow, right: ProjectionRow): number {
  return (
    Number(right.expected_points) - Number(left.expected_points) ||
    Number(right.p_appearance) - Number(left.p_appearance) ||
    Number(right.prob_points_ge_5) - Number(left.prob_points_ge_5) ||
    Number(left.price_tenths) - Number(right.price_tenths) ||
    left.stable_player_id.localeCompare(right.stable_player_id, undefined, {
      sensitivity: "base",
    }) ||
    left.player.localeCompare(right.player, undefined, { sensitivity: "base" })
  );
}

export function findPlayers(
  rows: ProjectionRow[],
  position: string,
  maximumBudgetMillions: string,
  replacedPlayerId?: string,
): PlayerFinderResult {
  const budgetMillions = parseBudgetMillions(maximumBudgetMillions);
  if (budgetMillions === null) {
    return {
      recommendations: [],
      position: isOfficialPosition(position) ? position : null,
      replacedPlayer: null,
      error: "Enter a valid budget greater than £0.0m, using at most one decimal place.",
    };
  }

  const validRows = validForecastPlayers(rows);
  const replacedPlayer = replacedPlayerId
    ? (validRows.find((row) => row.stable_player_id === replacedPlayerId) ?? null)
    : null;
  if (replacedPlayerId && !replacedPlayer) {
    return {
      recommendations: [],
      position: isOfficialPosition(position) ? position : null,
      replacedPlayer: null,
      error: "The selected player is unavailable in the current official forecast.",
    };
  }

  const effectivePosition = replacedPlayer?.position ?? position;
  if (!isOfficialPosition(effectivePosition)) {
    return {
      recommendations: [],
      position: null,
      replacedPlayer,
      error: "Choose a valid FPL position.",
    };
  }

  const maximumPriceTenths = budgetMillions * 10;
  const selectedExpectedPoints = replacedPlayer ? Number(replacedPlayer.expected_points) : null;
  const recommendations = validRows
    .filter(
      (row) =>
        row.position === effectivePosition &&
        Number(row.price_tenths) <= maximumPriceTenths &&
        row.stable_player_id !== replacedPlayer?.stable_player_id,
    )
    .sort(comparePlayers)
    .slice(0, 5)
    .map((player) => {
      const expectedPoints = Number(player.expected_points);
      return {
        player,
        expectedPointsPerMillion: expectedPoints / (Number(player.price_tenths) / 10),
        expectedPointsDifference:
          selectedExpectedPoints === null ? null : expectedPoints - selectedExpectedPoints,
      };
    });

  return {
    recommendations,
    position: effectivePosition,
    replacedPlayer,
    error: null,
  };
}
