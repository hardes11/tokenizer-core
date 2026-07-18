import * as esbuild from "esbuild";

await esbuild.build({
  entryPoints: ["src/index.ts"],
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "node18",
  outfile: "dist/index.js",
  // Keep deps external — consumer bundles them. Also keeps the 20MB
  // tokenizer.json out of dist (it's runtime-fetched, not bundled).
  external: ["@huggingface/tokenizers", "js-tiktoken"],
  logLevel: "info",
});