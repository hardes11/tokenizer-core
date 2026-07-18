// Verification #1 spike: does @huggingface/tokenizers + js-tiktoken bundle and run under esbuild CJS/Node?
import { Tokenizer } from "@huggingface/tokenizers";
import { getEncoding } from "js-tiktoken";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const dir = join(__dirname);
const tokenizerJson = JSON.parse(readFileSync(join(dir, "glm-5.2-tokenizer.json"), "utf8"));
const tokenizerConfig = JSON.parse(readFileSync(join(dir, "glm-5.2-tokenizer_config.json"), "utf8"));

// Real API (spec said Tokenizer.fromOptions — actual is `new Tokenizer(json, config)`)
const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);

const samples = [
  "Hello, world! This is a token-count spike.",
  "你好世界！这是一个 token 计数测试。",
  "Mixed 中英文 token count: the agent calls vault_count_tokens before sending.",
  "---\ntitle: Note\n---\n# Heading\n\nSome body text with `code` and a [[wikilink]].",
  "a",
  "",
];

console.log("=== @huggingface/tokenizers (GLM-5.2, exact) ===");
for (const s of samples) {
  const enc = tokenizer.encode(s);
  console.log(`  tokens=${String(enc.ids.length).padStart(4)}  chars=${String(s.length).padStart(4)}  ${JSON.stringify(s.slice(0, 40))}`);
}

console.log("\n=== js-tiktoken (o200k_base, for GPT-5/4o + approx base) ===");
const o200k = getEncoding("o200k_base");
for (const s of samples) {
  const ids = o200k.encode(s);
  console.log(`  tokens=${String(ids.length).padStart(4)}  chars=${String(s.length).padStart(4)}  ${JSON.stringify(s.slice(0, 40))}`);
}

console.log("\n=== js-tiktoken (cl100k_base, for GPT-4/3.5) ===");
const cl100k = getEncoding("cl100k_base");
for (const s of samples) {
  const ids = cl100k.encode(s);
  console.log(`  tokens=${String(ids.length).padStart(4)}  chars=${String(s.length).padStart(4)}  ${JSON.stringify(s.slice(0, 40))}`);
}

console.log("\n=== Approx check: claude = o200k * 1.15 ===");
const sample = "Mixed 中英文 token count: the agent calls vault_count_tokens before sending.";
const base = o200k.encode(sample).length;
console.log(`  o200k=${base}  claude_approx=${Math.ceil(base * 1.15)}`);

console.log("\nSPIKE OK — both libs load, encode, and produce counts.");