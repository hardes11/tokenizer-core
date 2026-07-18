import * as esbuild from "esbuild";
await esbuild.build({
  entryPoints: ["spike/spike.ts"],
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "node18",
  outfile: "spike/spike.bundle.cjs",
  logLevel: "info",
});
console.log("BUILD OK");
