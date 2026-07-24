import Papa from "papaparse";

export type JsonRecord = Record<string, unknown>;

export interface ProjectionRow {
  stable_player_id: string;
  player: string;
  team: string;
  position: string;
  price_tenths: string;
  fixture_count?: string;
  opponent_display?: string;
  opponent_short_names?: string;
  opponent_official_names?: string;
  home_away_sequence?: string;
  expected_points: string;
  expected_minutes: string;
  p_appearance: string;
  p_start: string;
  prob_points_ge_5: string;
  status: string;
  model_variant: string;
}

export interface SquadRow {
  player_uid: string;
  player_name: string;
  player_team_uid: string;
  fpl_position: string;
  price_tenths: string;
  fixture_count?: string;
  opponent_display?: string;
  opponent_short_names?: string;
  opponent_official_names?: string;
  home_away_sequence?: string;
  expected_points: string;
  selected_role: string;
  bench_order: string;
}

export interface LineupRow {
  captain: string;
  vice_captain: string;
  lineup: string;
  bench: string;
  formation: string;
  optimizer_variant?: string;
  expected_nominal_starting_xi_points?: string;
  expected_autosub_contribution?: string;
  expected_captain_bonus?: string;
  expected_vice_captain_contingency?: string;
  expected_realized_total?: string;
  expected_automatic_substitutions?: string;
  probability_unreplaced_starter?: string;
  solver_name?: string;
  solver_status: string;
  optimality_scope?: string;
  evaluated_squads?: string;
  analytic_method?: string;
}

export interface FrontendData {
  status: JsonRecord;
  projections: ProjectionRow[];
  squad: SquadRow[];
  lineup: LineupRow[];
  comparison: Record<string, string>[];
  freshness: JsonRecord;
  manifest: JsonRecord;
}

function assetUrl(filename: string): string {
  return `${import.meta.env.BASE_URL}data/${filename}`;
}

async function loadJson(filename: string): Promise<JsonRecord> {
  const response = await fetch(assetUrl(filename));
  if (!response.ok) {
    throw new Error(`Unable to load ${filename} (${response.status}). Run npm run sync-data.`);
  }
  return (await response.json()) as JsonRecord;
}

async function loadCsv<T>(filename: string): Promise<T[]> {
  const response = await fetch(assetUrl(filename));
  if (!response.ok) {
    throw new Error(`Unable to load ${filename} (${response.status}). Run npm run sync-data.`);
  }
  const parsed = Papa.parse<T>(await response.text(), {
    header: true,
    skipEmptyLines: true,
  });
  if (parsed.errors.length > 0) {
    throw new Error(`Unable to parse ${filename}: ${parsed.errors[0].message}`);
  }
  return parsed.data;
}

export async function loadFrontendData(): Promise<FrontendData> {
  const [status, projections, squad, lineup, comparison, freshness, manifest] = await Promise.all([
    loadJson("operational_status.json"),
    loadCsv<ProjectionRow>("player_gameweek_projections.csv"),
    loadCsv<SquadRow>("optimized_squad.csv"),
    loadCsv<LineupRow>("optimized_lineup.csv"),
    loadCsv<Record<string, string>>("model_comparison.csv"),
    loadJson("data_freshness.json"),
    loadJson("run_manifest.json"),
  ]);
  return { status, projections, squad, lineup, comparison, freshness, manifest };
}
