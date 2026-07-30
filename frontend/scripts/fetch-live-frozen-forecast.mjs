import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { FORECAST_FILES, freezeBundle } from "./frozen-forecast.mjs";

const destinationIndex = process.argv.indexOf("--destination");
const destination = destinationIndex === -1 ? undefined : process.argv[destinationIndex + 1];
const publicBaseUrl = "https://daniel-mehta.github.io/fpl-forecast/data";

if (!destination) throw new Error("Usage: fetch-live-frozen-forecast.mjs --destination DIR");
const download = path.join(destination, "download");
await mkdir(download, { recursive: true });
for (const filename of FORECAST_FILES) {
  const response = await fetch(`${publicBaseUrl}/${filename}`);
  if (!response.ok) throw new Error(`Unable to retrieve the current public forecast file ${filename}.`);
  await writeFile(path.join(download, filename), Buffer.from(await response.arrayBuffer()));
}
await freezeBundle(download, path.join(destination, "data"));
console.log("Current public forecast bundle was downloaded and frozen without regeneration.");
