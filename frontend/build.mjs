import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const outfile = path.join(here, "..", "src", "losscandles", "static", "index.js");

const watch = process.argv.includes("--watch");

const options = {
  entryPoints: [path.join(here, "src", "index.ts")],
  bundle: true,
  format: "esm",
  target: "es2020",
  outfile,
  minify: !watch,
  sourcemap: false,
  logLevel: "info",
};

if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  console.log("watching for changes...");
} else {
  await esbuild.build(options);
}
