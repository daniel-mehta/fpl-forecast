export const YOUR_TEAM_STORAGE_KEY = "fpl-forecast:your-team:v1";

export interface ForecastIdentity {
  season: string;
  gameweek: string;
  runId: string;
  playerIds: string[];
}

export interface SavedYourTeam {
  schemaVersion: 1;
  forecastIdentity: ForecastIdentity;
  playerIds: string[];
  sellingPrices: Record<string, number>;
  bankTenths: number;
  freeTransfers: number;
}

export function saveYourTeam(storage: Storage, value: SavedYourTeam): void {
  storage.setItem(YOUR_TEAM_STORAGE_KEY, JSON.stringify(value));
}

export function restoreYourTeam(
  storage: Storage,
  identity: ForecastIdentity,
): SavedYourTeam | null {
  const serialized = storage.getItem(YOUR_TEAM_STORAGE_KEY);
  if (!serialized) return null;
  try {
    const saved = JSON.parse(serialized) as SavedYourTeam;
    if (
      saved.schemaVersion !== 1 ||
      saved.forecastIdentity.season !== identity.season ||
      saved.forecastIdentity.gameweek !== identity.gameweek ||
      saved.forecastIdentity.runId !== identity.runId ||
      setKey(saved.forecastIdentity.playerIds) !== setKey(identity.playerIds) ||
      !Array.isArray(saved.playerIds) ||
      !saved.sellingPrices ||
      !Number.isInteger(saved.bankTenths) ||
      !Number.isInteger(saved.freeTransfers)
    ) {
      storage.removeItem(YOUR_TEAM_STORAGE_KEY);
      return null;
    }
    return saved;
  } catch {
    storage.removeItem(YOUR_TEAM_STORAGE_KEY);
    return null;
  }
}

export function resetYourTeam(storage: Storage): void {
  storage.removeItem(YOUR_TEAM_STORAGE_KEY);
}

function setKey(values: readonly string[]): string {
  return [...values].sort().join(",");
}
