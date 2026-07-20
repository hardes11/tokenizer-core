# tokenizer-core

**Framework-agnostic LLM token counter — GLM/GPT/Qwen/DeepSeek exact, Claude/Gemini approx. No Obsidian or MCP coupling; usable from any Node or bundler context.**

![npm version](https://img.shields.io/npm/v/tokenizer-core.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6.svg)

Count LLM tokens for text under a specific model's real tokenizer — so you can check whether content fits a context budget before sending it to a model. GLM, GPT, Qwen, and DeepSeek are counted **exactly** (HuggingFace `tokenizer.json` via `@huggingface/tokenizers`, or `js-tiktoken`); Claude and Gemini have no published offline tokenizer, so they use `o200k_base × 1.15` and are returned with `source: "approx"` — never silently passed off as exact.

This is the engine behind [`llm-token-count`](https://github.com/hardes11/llm-token-count) (an Obsidian status-bar plugin) and a `vault_count_tokens` MCP tool. It has no Obsidian or MCP imports — use it from a CLI, a script, another plugin, or any bundler.

## Install

```bash
npm install tokenizer-core
```

## Quick start

```typescript
import { countTokens, isApproxModel, listSupportedModels } from "tokenizer-core";

const result = await countTokens("Hello, world!", "glm-5.2", "/path/to/cache");
// => { tokens: 4, model: "glm-5.2", source: "exact", bytes: 13, chars: 13 }

isApproxModel("claude");   // true
isApproxModel("glm-5.2");  // false
listSupportedModels();
// => ["glm-5.2", "glm-5", "glm-4.6v-flash", "gpt-5", "gpt-4o", "gpt-4", "gpt-3.5", "qwen", "deepseek-v3.1", "claude", "gemini"]
```

## API

### `countTokens(text, model, cacheDir) → Promise<CountResult>`

- `text: string` — the text to count.
- `model: string` — one of `listSupportedModels()`.
- `cacheDir: string` — writable directory for caching HuggingFace `tokenizer.json` files (created if missing). The caller chooses the location (e.g. a plugin's `.obsidian/plugins/<id>/tokenizers/` dir).

Returns `{ tokens, model, source, bytes, chars }`. `source` is `"exact"` for HF/tiktoken models and `"approx"` for Claude/Gemini. Throws `TokenizerLoadError` on fetch failure — the caller should surface the error or fall back (the Obsidian plugins fall back to an approx count in the status bar).

### `isApproxModel(model) → boolean`

`true` for `claude` and `gemini`; `false` for all exact-tokenizer models.

### `listSupportedModels() → string[]`

The 11 supported model keys.

### `loadTokenizer(model, cacheDir) → Promise<Tokenizer | null>`

Lower-level access to the loaded tokenizer (returns `null` for tiktoken/approx models, which don't use a `Tokenizer` instance). Useful if you need `encode`/`decode` directly.

## Supported models

| Model | Source | Method |
|---|---|---|
| `glm-5.2`, `glm-5`, `glm-4.6v-flash` | exact | HuggingFace `tokenizer.json` (pinned SHA) |
| `gpt-5`, `gpt-4o` | exact | `js-tiktoken` `o200k_base` |
| `gpt-4`, `gpt-3.5` | exact | `js-tiktoken` `cl100k_base` |
| `qwen`, `deepseek-v3.1` | exact | HuggingFace `tokenizer.json` (pinned SHA) |
| `claude`, `gemini` | approx | `o200k_base × 1.15` (labeled `approx`) |

GLM/Qwen/DeepSeek tokenizer SHAs are pinned to a specific HuggingFace commit (see `src/registry.ts`) to guarantee reproducible counts across machines and over time. A version bump is a registry edit + a new release of this package.

## Why

GPT-based token counts are a poor proxy for GLM models: `o200k_base` over-counts pure Chinese by ~15–30% vs real GLM tokens, which is bad enough to break "does this fit in context?" decisions. This package loads each model's real tokenizer (HuggingFace for GLM/Qwen/DeepSeek, `js-tiktoken` for GPT) so the count is exact. For models with no published offline tokenizer (Claude, Gemini), it returns an honest labeled approximation rather than a silent guess.

## Implementation notes

- Pure TypeScript, no Obsidian or MCP imports — the cache path is a parameter, never hardcoded.
- `@huggingface/tokenizers` (pure JS port of the Rust `tokenizers` crate) + `js-tiktoken` are `dependencies`; the published `dist/` keeps them external so the consumer's bundler resolves them.
- Build: `npm run build` (esbuild → `dist/index.js`, `tsc --emitDeclarationOnly` → `dist/*.d.ts`). Test: `npm test` (Node's built-in test runner, no test framework dep).

## License

MIT