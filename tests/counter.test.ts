import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  countTokens,
  isApproxModel,
  listSupportedModels,
} from "../src/counter.js";
import { getTiktokenCount } from "../src/tiktoken.js";
import { clearTokenizerCache, TokenizerLoadError } from "../src/loader.js";

// Deterministic expectations, captured from the spike (spike/spike.ts) against
// the pinned GLM-5.2 tokenizer (sha b4734de4facf877f85769a911abafc5283eab3d9)
// and the js-tiktoken o200k_base / cl100k_base encodings.
const SAMPLES: Array<{ text: string; glm: number; o200k: number; cl100k: number }> = [
  { text: "Hello, world! This is a token-count spike.",                                        glm: 11, o200k: 11, cl100k: 11 },
  { text: "你好世界！这是一个 token 计数测试。",                                                glm: 10, o200k: 11, cl100k: 15 },
  { text: "Mixed 中英文 token count: the agent calls vault_count_tokens before sending.",     glm: 15, o200k: 15, cl100k: 16 },
  { text: "---\ntitle: Note\n---\n# Heading\n\nSome body text with `code` and a [[wikilink]].", glm: 23, o200k: 23, cl100k: 23 },
  { text: "a",                                                                                  glm: 1,  o200k: 1,  cl100k: 1 },
  { text: "",                                                                                   glm: 0,  o200k: 0,  cl100k: 0 },
];

let cacheDir: string;

test("setup: temp cacheDir", async () => {
  cacheDir = await mkdtemp(join(tmpdir(), "tokenizers-core-test-"));
});

describe("GLM-5.2 exact counts (network on first run, cached after)", () => {
  for (const s of SAMPLES) {
    test(`glm-5.2  text=${JSON.stringify(s.text.slice(0, 30))} → ${s.glm}`, async () => {
      const r = await countTokens(s.text, "glm-5.2", cacheDir);
      assert.equal(r.source, "exact");
      assert.equal(r.model, "glm-5.2");
      assert.equal(r.tokens, s.glm, `glm-5.2 count mismatch for ${JSON.stringify(s.text)}`);
      assert.equal(r.chars, s.text.length);
      assert.equal(r.bytes, Buffer.byteLength(s.text, "utf8"));
    });
  }
});

describe("clearTokenizerCache (re-download bug fix)", () => {
  test("clearing cache deletes disk files for a loaded model", async () => {
    // Pre-condition: the GLM-5.2 describe block above already loaded the
    // tokenizer into the in-memory Map AND wrote the disk cache files.
    const tokPath = join(cacheDir, "glm-5.2.json");
    const cfgPath = join(cacheDir, "glm-5.2.config.json");
    assert.ok(existsSync(tokPath), "pre: tokenizer.json should exist on disk");
    assert.ok(existsSync(cfgPath), "pre: tokenizer_config.json should exist on disk");

    clearTokenizerCache("glm-5.2", cacheDir);

    assert.equal(existsSync(tokPath), false, "tokenizer.json should be gone after clear");
    assert.equal(existsSync(cfgPath), false, "tokenizer_config.json should be gone after clear");
  });

  test("after clear, countTokens re-fetches and still produces the exact count", async () => {
    // The in-memory Map is now empty AND disk files are gone — so countTokens
    // MUST re-fetch from HuggingFace. If the clear missed the in-memory
    // layer, this would still return a count (the bug) but here we additionally
    // confirm the disk files reappear (proving a real re-fetch happened).
    const tokPath = join(cacheDir, "glm-5.2.json");
    const cfgPath = join(cacheDir, "glm-5.2.config.json");
    assert.equal(existsSync(tokPath), false, "pre-clear: tokenizer.json gone");

    const r = await countTokens("Hello, world! This is a token-count spike.", "glm-5.2", cacheDir);
    assert.equal(r.source, "exact");
    assert.equal(r.tokens, 11, "re-fetched count must still match the pinned expectation");
    assert.ok(existsSync(tokPath), "post: tokenizer.json re-fetched to disk");
    assert.ok(existsSync(cfgPath), "post: tokenizer_config.json re-fetched to disk");
  });

  test("clearTokenizerCache is a no-op (ENOENT-safe) for never-cached models", () => {
    assert.doesNotThrow(() => clearTokenizerCache("glm-5", cacheDir));
    assert.equal(existsSync(join(cacheDir, "glm-5.json")), false);
  });

  test("clearTokenizerCache is a no-op for non-HF models (tiktoken/approx)", () => {
    // gpt-4o is tiktoken — no disk files, no in-memory entry. Must not throw.
    assert.doesNotThrow(() => clearTokenizerCache("gpt-4o", cacheDir));
    assert.doesNotThrow(() => clearTokenizerCache("claude", cacheDir));
  });
});

describe("tiktoken o200k_base (gpt-5, gpt-4o)", () => {
  for (const s of SAMPLES) {
    test(`gpt-5  text=${JSON.stringify(s.text.slice(0, 30))} → ${s.o200k}`, async () => {
      const r = await countTokens(s.text, "gpt-5", cacheDir);
      assert.equal(r.source, "exact");
      assert.equal(r.model, "gpt-5");
      assert.equal(r.tokens, s.o200k);
    });
    test(`gpt-4o text=${JSON.stringify(s.text.slice(0, 30))} → ${s.o200k}`, async () => {
      const r = await countTokens(s.text, "gpt-4o", cacheDir);
      assert.equal(r.source, "exact");
      assert.equal(r.model, "gpt-4o");
      assert.equal(r.tokens, s.o200k);
    });
  }
});

describe("tiktoken cl100k_base (gpt-4, gpt-3.5)", () => {
  for (const s of SAMPLES) {
    test(`gpt-4  text=${JSON.stringify(s.text.slice(0, 30))} → ${s.cl100k}`, async () => {
      const r = await countTokens(s.text, "gpt-4", cacheDir);
      assert.equal(r.source, "exact");
      assert.equal(r.model, "gpt-4");
      assert.equal(r.tokens, s.cl100k);
    });
    test(`gpt-3.5 text=${JSON.stringify(s.text.slice(0, 30))} → ${s.cl100k}`, async () => {
      const r = await countTokens(s.text, "gpt-3.5", cacheDir);
      assert.equal(r.source, "exact");
      assert.equal(r.model, "gpt-3.5");
      assert.equal(r.tokens, s.cl100k);
    });
  }
});

describe("isApproxModel", () => {
  test("claude is approx", () => {
    assert.equal(isApproxModel("claude"), true);
  });
  test("gemini is approx", () => {
    assert.equal(isApproxModel("gemini"), true);
  });
  test("glm-5.2 is NOT approx", () => {
    assert.equal(isApproxModel("glm-5.2"), false);
  });
  test("gpt-5 is NOT approx", () => {
    assert.equal(isApproxModel("gpt-5"), false);
  });
  test("unknown model is NOT approx (undefined)", () => {
    assert.equal(isApproxModel("does-not-exist"), false);
  });
});

describe("claude approx", () => {
  const sample = "Mixed 中英文 token count: the agent calls vault_count_tokens before sending.";
  test("claude count === ceil(o200k * 1.15), source=approx", async () => {
    const base = getTiktokenCount(sample, "o200k_base");
    const expected = Math.ceil(base * 1.15);
    const r = await countTokens(sample, "claude", cacheDir);
    assert.equal(r.source, "approx");
    assert.equal(r.model, "claude");
    assert.equal(r.tokens, expected);
  });
});

describe("listSupportedModels", () => {
  test("includes all 11 model keys", () => {
    const models = listSupportedModels();
    const expected = [
      "glm-5.2",
      "glm-5",
      "glm-4.6v-flash",
      "gpt-5",
      "gpt-4o",
      "gpt-4",
      "gpt-3.5",
      "qwen",
      "deepseek-v3.1",
      "claude",
      "gemini",
    ];
    assert.equal(models.length, expected.length);
    for (const m of expected) {
      assert.ok(models.includes(m), `missing model ${m}`);
    }
  });
});

describe("unknown model rejection", () => {
  test("countTokens throws TokenizerLoadError for unknown model", async () => {
    await assert.rejects(
      () => countTokens("hello", "not-a-real-model", cacheDir),
      (err: unknown) => {
        assert.ok(err instanceof TokenizerLoadError, "should be TokenizerLoadError");
        assert.match((err as Error).message, /Unknown model/);
        return true;
      },
    );
  });
});

// NOT tested here: non-markdown path rejection — that is the MCP tool's job,
// not the core's. Core is framework-agnostic and path-unaware.

// Manual validation gate (requires Zhipu API key — DO NOT run in CI):
// TODO: API validation: GLM-5.2 count from countTokens(text,"glm-5.2",dir)
//       should match `usage.prompt_tokens` from a real Zhipu API call (±1 token).
//       Run manually with a real key; not automated here.
test.skip("API validation: GLM-5.2 count matches usage.prompt_tokens from a real Zhipu API call (±1 token) — requires API key, run manually", async () => {
  // Intentionally empty — see TODO above.
});

test("teardown: remove temp cacheDir", async () => {
  if (cacheDir) await rm(cacheDir, { recursive: true, force: true });
});