import type { JsonRecord, ProjectionRow, SquadRow } from "./data";
import { formatDate, formatTeam } from "./formatting";

export interface ProjectionFilters {
  search: string;
  position: string;
  minPriceTenths: number | null;
  maxPriceTenths: number | null;
}

export interface PriceRange {
  minPriceTenths: number | null;
  maxPriceTenths: number | null;
}

export function filterProjections(
  rows: ProjectionRow[],
  filters: ProjectionFilters,
): ProjectionRow[] {
  const query = filters.search.trim().toLowerCase();
  return rows.filter((row) => {
    const price = Number(row.price_tenths);
    const haystack =
      `${row.player} ${formatTeam(row.team)} ${row.opponent_display ?? ""}`.toLowerCase();
    return (
      (filters.position === "ALL" || row.position === filters.position) &&
      (!query || haystack.includes(query)) &&
      (filters.minPriceTenths === null || price >= filters.minPriceTenths) &&
      (filters.maxPriceTenths === null || price <= filters.maxPriceTenths)
    );
  });
}

export function normalizePriceRange(
  changed: "min" | "max",
  nextValue: number | null,
  current: PriceRange,
): PriceRange {
  if (changed === "min") {
    return {
      minPriceTenths: nextValue,
      maxPriceTenths:
        nextValue !== null &&
        current.maxPriceTenths !== null &&
        nextValue > current.maxPriceTenths
          ? nextValue
          : current.maxPriceTenths,
    };
  }
  return {
    minPriceTenths:
      nextValue !== null &&
      current.minPriceTenths !== null &&
      nextValue < current.minPriceTenths
        ? nextValue
        : current.minPriceTenths,
    maxPriceTenths: nextValue,
  };
}

export function availablePrices(rows: ProjectionRow[]): number[] {
  return [...new Set(rows.map((row) => Number(row.price_tenths)).filter(Number.isFinite))].sort(
    (left, right) => left - right,
  );
}

export function groupSquad(rows: SquadRow[]): {
  starters: SquadRow[];
  bench: SquadRow[];
} {
  const bench = rows
    .filter((row) => Number(row.bench_order) > 0 || row.selected_role === "squad")
    .sort((left, right) => Number(left.bench_order) - Number(right.bench_order));
  const benchIds = new Set(bench.map((row) => row.player_uid));
  return {
    starters: rows.filter((row) => !benchIds.has(row.player_uid)),
    bench,
  };
}

export function squadRoleLabel(player: SquadRow): string {
  const order = Number(player.bench_order);
  if (Number.isFinite(order) && order > 0) {
    return order === 1 ? "Bench goalkeeper" : `Bench ${order - 1}`;
  }
  if (player.selected_role === "captain") {
    return "Captain";
  }
  if (player.selected_role === "vice_captain") {
    return "Vice captain";
  }
  return "Starter";
}

export function latestOfficialRetrievalTimestamp(freshness: JsonRecord): string | null {
  const snapshots = freshness.official_snapshots;
  if (!snapshots || typeof snapshots !== "object" || Array.isArray(snapshots)) {
    return null;
  }
  const timestamps = Object.values(snapshots)
    .map((snapshot) => {
      if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) {
        return null;
      }
      const value = (snapshot as JsonRecord).retrieved_at;
      return typeof value === "string" && !Number.isNaN(new Date(value).valueOf()) ? value : null;
    })
    .filter((value): value is string => value !== null);
  if (timestamps.length === 0) {
    return null;
  }
  return timestamps.sort(
    (left, right) => new Date(right).valueOf() - new Date(left).valueOf(),
  )[0];
}

export function timestampLine(label: string, value: unknown): string | null {
  const formatted = formatDate(value);
  return formatted === "Not available" ? null : `${label} ${formatted}`;
}
