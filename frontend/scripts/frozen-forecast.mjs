import { createHash } from "node:crypto";
import { cp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";

export const FRONTEND_SCHEMA_VERSION = "phase9_frontend_v1";
export const BUNDLE_SCHEMA_VERSION = "frozen_official_forecast_v1";
export const FORECAST_FILES = [
  "operational_status.json",
  "player_gameweek_projections.csv",
  "optimized_squad.csv",
  "optimized_lineup.csv",
  "model_comparison.csv",
  "data_freshness.json",
  "run_manifest.json",
];
export const BUNDLE_MANIFEST = "frozen_forecast_manifest.json";

function fail(message) {
  throw new Error(`No valid frozen official forecast bundle was available. Frontend deployment was cancelled. ${message}`);
}

function sha256(contents) {
  return createHash("sha256").update(contents).digest("hex");
}

async function readJson(directory, filename) {
  let contents;
  try {
    contents = await readFile(path.join(directory, filename), "utf8");
  } catch {
    fail(`Required file ${filename} is missing.`);
  }
  try {
    return JSON.parse(contents);
  } catch {
    fail(`Required JSON file ${filename} is invalid.`);
  }
}

function value(record, name) {
  return record && typeof record === "object" ? record[name] : undefined;
}

function requireEqual(label, ...values) {
  if (values.some((item) => item === undefined || item === null || item === "") || new Set(values).size !== 1) {
    fail(`${label} is missing or inconsistent.`);
  }
}

async function inventory(directory) {
  const files = {};
  for (const filename of FORECAST_FILES) {
    let contents;
    try {
      contents = await readFile(path.join(directory, filename));
    } catch {
      fail(`Required file ${filename} is missing.`);
    }
    files[filename] = { bytes: contents.length, sha256: sha256(contents) };
  }
  return files;
}

async function validateCsv(directory, filename, requiredColumns = []) {
  let contents;
  try {
    contents = await readFile(path.join(directory, filename), "utf8");
  } catch {
    fail(`Required file ${filename} is missing.`);
  }
  const [header, ...rows] = contents.trim().split(/\r?\n/);
  const columns = new Set((header ?? "").split(","));
  if (!header || rows.length === 0 || requiredColumns.some((column) => !columns.has(column))) {
    fail(`${filename} is empty or does not have the required public columns.`);
  }
  const schemaColumn = header.split(",").indexOf("schema_version");
  if (schemaColumn >= 0 && rows.some((row) => row.split(",")[schemaColumn] !== FRONTEND_SCHEMA_VERSION)) {
    fail(`${filename} has an unsupported frontend schema.`);
  }
  return rows.length;
}

export async function validateFrozenBundle(directory, { requireManifest = true } = {}) {
  const status = await readJson(directory, "operational_status.json");
  const freshness = await readJson(directory, "data_freshness.json");
  const runManifest = await readJson(directory, "run_manifest.json");
  for (const [label, record] of [["operational status", status], ["freshness", freshness]]) {
    if (value(record, "schema_version") !== FRONTEND_SCHEMA_VERSION) {
      fail(`${label} has an unsupported frontend schema.`);
    }
  }
  if (value(runManifest, "frontend_schema_version") !== FRONTEND_SCHEMA_VERSION) {
    fail("Run manifest has an unsupported frontend schema.");
  }
  requireEqual("Forecast run ID", value(status, "run_id"), value(runManifest, "run_id"));
  requireEqual("Forecast season", value(status, "target_season"), value(runManifest, "target_season"));
  requireEqual("Forecast gameweek", value(status, "target_gameweek"), value(runManifest, "target_gameweek"));
  if (value(status, "state") !== "SUCCEEDED" || !value(runManifest, "completed_at") || !value(freshness, "generated_at")) {
    fail("Forecast publication state or timestamp is missing.");
  }
  const simulator = value(value(runManifest, "model_lineage"), "xpoints_simulator");
  if (!value(simulator, "version") || !value(simulator, "model_contract_version")) {
    fail("Forecast simulator identity is missing.");
  }
  const projectionRows = await validateCsv(directory, "player_gameweek_projections.csv", [
    "schema_version",
    "stable_player_id",
    "gameweek",
    "expected_points_given_appearance",
  ]);
  await validateCsv(directory, "optimized_squad.csv");
  await validateCsv(directory, "optimized_lineup.csv");
  await validateCsv(directory, "model_comparison.csv");

  const files = await inventory(directory);
  const identity = {
    run_id: value(runManifest, "run_id"),
    season: value(runManifest, "target_season"),
    gameweek: value(runManifest, "target_gameweek"),
    published_at: value(runManifest, "completed_at"),
    generated_at: value(freshness, "generated_at"),
    simulator_version: value(simulator, "version"),
    output_contract_version: value(simulator, "model_contract_version"),
    player_count: Number(value(value(runManifest, "model_lineage"), "current_player_count")),
  };
  if (!Number.isInteger(identity.player_count) || identity.player_count < 1) {
    fail("Forecast player count is missing or invalid.");
  }
  if (projectionRows !== identity.player_count) fail("Forecast player count does not match the public projections.");

  if (requireManifest) {
    const manifest = await readJson(directory, BUNDLE_MANIFEST);
    if (value(manifest, "schema_version") !== BUNDLE_SCHEMA_VERSION) fail("Frozen bundle manifest version is unsupported.");
    for (const [key, item] of Object.entries(identity)) requireEqual(`Frozen manifest ${key}`, item, value(value(manifest, "forecast_identity"), key));
    for (const [filename, expected] of Object.entries(files)) {
      const actual = value(value(manifest, "files"), filename);
      if (!actual || actual.sha256 !== expected.sha256 || actual.bytes !== expected.bytes) fail(`Checksum mismatch for ${filename}.`);
    }
  }
  return { files, identity };
}

export async function freezeBundle(source, destination) {
  const { files, identity } = await validateFrozenBundle(source, { requireManifest: false });
  await mkdir(destination, { recursive: true });
  for (const filename of FORECAST_FILES) await cp(path.join(source, filename), path.join(destination, filename));
  await writeFile(path.join(destination, BUNDLE_MANIFEST), `${JSON.stringify({ schema_version: BUNDLE_SCHEMA_VERSION, forecast_identity: identity, files }, null, 2)}\n`);
  return { files, identity };
}

export async function prepareBundle(source, destination) {
  await validateFrozenBundle(source);
  await mkdir(destination, { recursive: true });
  for (const entry of await readdir(destination)) {
    if (entry !== "README.md") await cp(path.join(source, entry), path.join(destination, entry), { recursive: true, force: true });
  }
  for (const filename of FORECAST_FILES) await cp(path.join(source, filename), path.join(destination, filename));
  await cp(path.join(source, BUNDLE_MANIFEST), path.join(destination, BUNDLE_MANIFEST));
}
