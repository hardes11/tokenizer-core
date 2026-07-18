import { Tokenizer } from "@huggingface/tokenizers";
import { existsSync, unlinkSync } from "node:fs";
import { mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { lookupModel } from "./registry.js";
import { getTiktokenCount } from "./tiktoken.js";

/**
 * Thrown when a tokenizer cannot be loaded — either the network fetch failed
 * or the on-disk cache is corrupt beyond recovery.
 */
export class TokenizerLoadError extends Error {
  readonly model: string;
  readonly cause?: unknown;
  constructor(message: string, model: string, cause?: unknown) {
    super(message);
    this.name = "TokenizerLoadError";
    this.model = model;
    if (cause !== undefined) {
      this.cause = cause;
    }
  }
}

/** In-memory cache of constructed HF Tokenizer instances, keyed by model. */
const tokenizerCache = new Map<string, Tokenizer>();

function hfUrl(repo: string, sha: string, file: string): string {
  return `https://huggingface.co/${repo}/resolve/${sha}/${file}`;
}

async function fetchJson(url: string): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new TokenizerLoadError(`Network error fetching ${url}`, "network", err);
  }
  if (!res.ok) {
    throw new TokenizerLoadError(
      `HTTP ${res.status} fetching ${url}`,
      "network",
    );
  }
  try {
    return await res.json();
  } catch (err) {
    throw new TokenizerLoadError(`Invalid JSON fetching ${url}`, "network", err);
  }
}

async function readJsonFile(path: string): Promise<unknown> {
  const buf = await readFile(path, "utf8");
  return JSON.parse(buf);
}

/**
 * Load (and cache) a tokenizer for the given model.
 *
 * - `hf` source: fetches `tokenizer.json` and `tokenizer_config.json` from
 *   HuggingFace at the pinned SHA, caches them under `cacheDir`, and
 *   constructs `new Tokenizer(json, config)`.
 * - `tiktoken` / `approx` sources: nothing to load — returns `null`.
 *
 * Repeated calls for the same model return the same in-memory instance.
 */
export async function loadTokenizer(
  model: string,
  cacheDir: string,
): Promise<Tokenizer | null> {
  const entry = lookupModel(model);
  if (!entry) {
    throw new TokenizerLoadError(`Unknown model: ${model}`, model);
  }
  if (entry.source !== "hf") {
    return null;
  }

  const cached = tokenizerCache.get(model);
  if (cached) return cached;

  const tokPath = join(cacheDir, `${model}.json`);
  const cfgPath = join(cacheDir, `${model}.config.json`);

  await mkdir(cacheDir, { recursive: true });

  let tokenizerJson: unknown;
  let tokenizerConfig: unknown;

  // Try cache first; on corrupt-cache JSON errors, delete + refetch once.
  try {
    tokenizerJson = await readJsonFile(tokPath);
  } catch (e) {
    try { await unlink(tokPath); } catch { /* ignore */ }
    tokenizerJson = await fetchJson(hfUrl(entry.repo!, entry.sha!, "tokenizer.json"));
    await writeFile(tokPath, JSON.stringify(tokenizerJson));
  }

  try {
    tokenizerConfig = await readJsonFile(cfgPath);
  } catch (e) {
    try { await unlink(cfgPath); } catch { /* ignore */ }
    tokenizerConfig = await fetchJson(hfUrl(entry.repo!, entry.sha!, "tokenizer_config.json"));
    await writeFile(cfgPath, JSON.stringify(tokenizerConfig));
  }

  let tokenizer: Tokenizer;
  try {
    tokenizer = new Tokenizer(tokenizerJson as object, tokenizerConfig as object);
  } catch (err) {
    // Corrupt cache that parsed as JSON but is wrong shape — purge + refetch once.
    try { await unlink(tokPath); } catch { /* ignore */ }
    try { await unlink(cfgPath); } catch { /* ignore */ }
    try {
      tokenizerJson = await fetchJson(hfUrl(entry.repo!, entry.sha!, "tokenizer.json"));
      tokenizerConfig = await fetchJson(hfUrl(entry.repo!, entry.sha!, "tokenizer_config.json"));
      await writeFile(tokPath, JSON.stringify(tokenizerJson));
      await writeFile(cfgPath, JSON.stringify(tokenizerConfig));
      tokenizer = new Tokenizer(tokenizerJson as object, tokenizerConfig as object);
    } catch (err2) {
      throw new TokenizerLoadError(
        `Failed to construct tokenizer for ${model}`,
        model,
        err2,
      );
    }
  }

  tokenizerCache.set(model, tokenizer);
  return tokenizer;
}

/**
 * Invalidate BOTH layers of the tokenizer cache for a model:
 *   - drops the in-memory `Tokenizer` instance (module-level Map), and
 *   - deletes the on-disk `<model>.json` + `<model>.config.json` files.
 *
 * After this, the next `loadTokenizer` / `countTokens` call for the model
 * will re-fetch from HuggingFace. Without the in-memory drop, deleting only
 * the disk files is a no-op until the process is restarted — which is exactly
 * the bug this function fixes (the standalone plugin's "Re-download
 * tokenizer" button was clearing disk but not memory).
 *
 * Non-HF models (tiktoken / approx) have no cache and are a no-op.
 * Missing disk files are ignored (ENOENT-safe).
 */
export function clearTokenizerCache(model: string, cacheDir: string): void {
  // 1. In-memory layer — always safe to delete even if absent.
  tokenizerCache.delete(model);

  // 2. Disk layer — best-effort unlink, ignore ENOENT / other errors.
  const tokPath = join(cacheDir, `${model}.json`);
  const cfgPath = join(cacheDir, `${model}.config.json`);
  for (const p of [tokPath, cfgPath]) {
    try {
      if (existsSync(p)) unlinkSync(p);
    } catch {
      /* ignore — file may be missing or locked; not our problem here */
    }
  }
}

// Re-export for consumers that only want the tiktoken helper path.
export { getTiktokenCount };