import path from "node:path";
import { freezeBundle, prepareBundle, validateFrozenBundle } from "./frozen-forecast.mjs";

function option(name) {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

const [command] = process.argv.slice(2);
const source = option("--source");
const destination = option("--destination");

try {
  if (!source || (command !== "validate" && !destination)) throw new Error("Usage: frozen-forecast-cli.mjs <validate|freeze|prepare> --source DIR [--destination DIR]");
  if (command === "validate") await validateFrozenBundle(path.resolve(source));
  else if (command === "freeze") await freezeBundle(path.resolve(source), path.resolve(destination));
  else if (command === "prepare") await prepareBundle(path.resolve(source), path.resolve(destination));
  else throw new Error(`Unknown frozen forecast command: ${command}`);
  console.log(`Frozen forecast ${command} completed.`);
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
