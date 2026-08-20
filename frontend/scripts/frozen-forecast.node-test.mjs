import assert from "node:assert/strict";
import { cp, mkdtemp, readFile, rm, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { FORECAST_FILES, freezeBundle, prepareBundle, validateFrozenBundle } from "./frozen-forecast.mjs";

const repositoryRoot = path.resolve(import.meta.dirname, "../..");
const validSource = path.join(repositoryRoot, "frontend/public/data");

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), "frozen-forecast-"));
  const source = path.join(root, "source");
  await copyValidSource(source);
  return { root, source };
}

async function copyValidSource(source) {
  await cp(validSource, source, { recursive: true, force: true });
  const projectionPath = path.join(source, "player_gameweek_projections.csv");
  const projectionContents = await readFile(projectionPath, "utf8");
  if (!projectionContents.split(/\r?\n/, 1)[0].includes("expected_points_given_appearance")) {
    const [header, ...rows] = projectionContents.trimEnd().split(/\r?\n/);
    await writeFile(
      projectionPath,
      [`${header},expected_points_given_appearance`, ...rows.map((row) => `${row},1`)].join("\n") + "\n",
    );
  }
}

test("valid frozen bundle passes and preparation preserves public bytes", async (t) => {
  const { root, source } = await fixture();
  t.after(() => rm(root, { recursive: true, force: true }));
  const frozen = path.join(root, "frozen");
  await freezeBundle(source, frozen);
  const prepared = path.join(root, "prepared");
  await prepareBundle(frozen, prepared);
  await validateFrozenBundle(frozen);
  for (const filename of FORECAST_FILES) assert.deepEqual(await readFile(prepared + "/" + filename), await readFile(frozen + "/" + filename));
});

test("missing files, invalid JSON, missing identity, and checksum mismatch fail closed", async (t) => {
  const { root, source } = await fixture();
  t.after(() => rm(root, { recursive: true, force: true }));
  await assert.rejects(() => freezeBundle(path.join(root, "missing"), path.join(root, "out")), /No valid frozen/);
  await unlink(path.join(source, "optimized_lineup.csv"));
  await assert.rejects(() => freezeBundle(source, path.join(root, "missing-file")), /Required file/);
  await copyValidSource(source);
  const projectionPath = path.join(source, "player_gameweek_projections.csv");
  const projectionRows = (await readFile(projectionPath, "utf8")).trimEnd().split(/\r?\n/);
  await writeFile(
    projectionPath,
    projectionRows.map((row) => row.slice(0, row.lastIndexOf(","))).join("\n") + "\n",
  );
  await assert.rejects(
    () => freezeBundle(source, path.join(root, "missing-conditional-xpoints")),
    /required public columns/,
  );
  await copyValidSource(source);
  await writeFile(path.join(source, "run_manifest.json"), "not json");
  await assert.rejects(() => freezeBundle(source, path.join(root, "invalid-json")), /invalid/);
  await copyValidSource(source);
  const status = JSON.parse(await readFile(path.join(source, "operational_status.json"), "utf8"));
  status.schema_version = "invalid";
  await writeFile(path.join(source, "operational_status.json"), JSON.stringify(status));
  await assert.rejects(() => freezeBundle(source, path.join(root, "invalid-schema")), /unsupported frontend schema/);
  await copyValidSource(source);
  const manifest = JSON.parse(await readFile(path.join(source, "run_manifest.json"), "utf8"));
  delete manifest.run_id;
  await writeFile(path.join(source, "run_manifest.json"), JSON.stringify(manifest));
  await assert.rejects(() => freezeBundle(source, path.join(root, "missing-identity")), /run ID/);
  await copyValidSource(source);
  const frozen = path.join(root, "checksum");
  await freezeBundle(source, frozen);
  await writeFile(
    path.join(frozen, "model_comparison.csv"),
    `${await readFile(path.join(frozen, "model_comparison.csv"), "utf8")}\n`,
  );
  await assert.rejects(() => validateFrozenBundle(frozen), /Checksum mismatch/);
});
