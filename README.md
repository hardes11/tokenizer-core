# @hardes11/tokenizers-core

Framework-agnostic LLM token counter. GLM/GPT/Qwen/DeepSeek exact (via HuggingFace `tokenizer.json` + `@huggingface/tokenizers`, or `js-tiktoken`); Claude/Gemini approx (`o200k_base` x 1.15, labeled). No Obsidian or MCP dependencies — usable from any Node or bundler context.

## Install

```bash
npm install @hardes11/tokenizers-core
```

## API

```typescript
import { countTokens, isApproxModel, listSupportedModels } from "@hardes11/tokenizers-core";

const result = await countTokens("Hello, world!", "glm-5.2", "/path/to/cache");
// => { tokens: 4, model: "glm-5.2", source: "exact", bytes: 13, chars: 13 }

isApproxModel("claude");   // true
isApproxModel("glm-5.2");  // false
listSupportedModels();     // ["glm-5.2", "glm-5", "glm-4.6v-flash", "gpt-5", "gpt-4o", "gpt-4", "gpt-3.5", "qwen", "deepseek-v3.1", "claude", "gemini"]
```

### `countTokens(text, model, cacheDir) -> Promise<CountResult>`

- `text: string` — the text to count.
- `model: string` — one of `listSupportedModels()`.
- `cacheDir: string` — writable directory for caching HuggingFace `tokenizer.json` files (created if missing). The caller chooses the location (e.g. a plugin's `.obsidian/plugins/<id>/tokenizers/` dir).

Returns `{ tokens, model, source, bytes, chars }`. `source` is `"exact"` for HF/tiktoken models and `"approx"` for Claude/Gemini. Throws `TokenizerLoadError` on fetch failure (caller should surface or fall back).

### Models

| Model | Source | Method |
|---|---|---|
| `glm-5.2`, `glm-5`, `glm-4.6v-flash` | exact | HuggingFace `tokenizer.json` (pinned SHA) |
| `gpt-5`, `gpt-4o` | exact | `js-tiktoken` `o200k_base` |
| `gpt-4`, `gpt-3.5` | exact | `js-tiktoken` `cl100k_base` |
| `qwen`, `deepseek-v3.1` | exact | HuggingFace `tokenizer.json` (pinned SHA) |
| `claude`, `gemini` | approx | `o200k_base` x 1.15 (labeled `approx`) |

GLM tokenizer SHAs are pinned to a specific HuggingFace commit (see `src/registry.ts`) to guarantee reproducible counts. A version bump is a registry edit + a new release.

## License

MIT