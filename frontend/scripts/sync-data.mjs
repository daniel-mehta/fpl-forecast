import { copyFile, mkdir, readFile, readdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDirectory, "../..");
const operationalRoot = path.join(projectRoot, "outputs", "operational");
const runsRoot = path.join(operationalRoot, "runs");
const pointerPath = path.join(operationalRoot, "latest_successful.json");
const destination = path.join(projectRoot, "frontend", "public", "data");
const schemaVersion = "phase8_frontend_v1";
const artifacts = [
  "operational_status.json",
  "player_gameweek_projections.csv",
  "optimized_squad.csv",
  "optimized_lineup.csv",
  "model_comparison.csv",
  "data_freshness.json",
  "run_manifest.json",
];

function sanitizeJson(value) {
  if (Array.isArray(value)) {
    return value.map(sanitizeJson);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, sanitizeJson(item)]));
  }
  if (typeof value === "string" && path.isAbsolute(value)) {
    return "[local path omitted]";
  }
  return value;
}

async function readJson(filePath, label) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    throw new Error(`Unable to read ${label} at ${filePath}: ${error.message}`);
  }
}

async function main() {
  const pointer = await readJson(pointerPath, "latest successful operational pointer");
  if (pointer.schema_version !== schemaVersion) {
    throw new Error(
      `Unsupported frontend schema ${pointer.schema_version ?? "missing"}; expected ${schemaVersion}.`,
    );
  }

  const runId = String(pointer.run_id ?? "");
  if (!/^[A-Za-z0-9._-]+$/.test(runId)) {
    throw new Error("The latest successful pointer contains an unsafe or missing run_id.");
  }

  const runDirectory = path.resolve(runsRoot, runId);
  if (!runDirectory.startsWith(`${path.resolve(runsRoot)}${path.sep}`)) {
    throw new Error("The latest successful run resolves outside outputs/operational/runs.");
  }
  if (pointer.run_dir && path.resolve(pointer.run_dir) !== runDirectory) {
    throw new Error("The latest successful pointer does not match its expected run directory.");
  }

  await mkdir(destination, { recursive: true });
  for (const entry of await readdir(destination, { withFileTypes: true })) {
    if (entry.isFile() && entry.name !== "README.md") {
      await unlink(path.join(destination, entry.name));
    }
  }

  for (const artifact of artifacts) {
    const source = path.join(runDirectory, artifact);
    const target = path.join(destination, artifact);
    if (artifact.endsWith(".json")) {
      const contents = sanitizeJson(await readJson(source, artifact));
      await writeFile(target, `${JSON.stringify(contents, null, 2)}\n`, "utf8");
    } else {
      const contents = await readFile(source, "utf8");
      if (/(^|[",\s])\/(?:Users|private|var|tmp)\//m.test(contents)) {
        throw new Error(`${artifact} contains an absolute local path and was not copied.`);
      }
      await copyFile(source, target);
    }
  }

  console.log(`Synced ${artifacts.length} frontend artifacts from run ${runId}.`);
  console.log(`Destination: ${path.relative(projectRoot, destination)}`);
}

main().catch((error) => {
  console.error(`Data sync failed: ${error.message}`);
  process.exitCode = 1;
});
