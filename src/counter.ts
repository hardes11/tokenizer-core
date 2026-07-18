import { Buffer } from "node:buffer";
import { REGISTRY, lookupModel } from "./registry.js";
import { loadTokenizer, TokenizerLoadError } from "./loader.js";
import { getTiktokenCount } from "./tiktoken.js";

export type TokenizerSource = "exact" | "approx";

export interface CountResult {
  tokens: number;
  model: string;
  /** "exact" for hf/tiktoken sources, "approx" for approx sources. */
  source: TokenizerSource;
  bytes: number;
  chars: number;
}

/**
 * Count tokens for `text` under the named `model`.
 *
 * - HF models: load tokenizer (network/cached) and `encode(text).ids.length`.
 * - tiktoken models: encode with the configured base.
 * - approx models: `ceil(tiktoken_base_count * multiplier)`.
 *
 * `cacheDir` is where HF tokenizer.json / tokenizer_config.json are stored.
 * It is created if missing. Never hardcoded — caller-supplied.
 */
export async function countTokens(
  text: string,
  model: string,
  cacheDir: string,
): Promise<CountResult> {
  const entry = lookupModel(model);
  if (!entry) {
    throw new TokenizerLoadError(`Unknown model: ${model}`, model);
  }

  const bytes = Buffer.byteLength(text, "utf8");
  const chars = text.length;

  if (entry.source === "hf") {
    const tokenizer = await loadTokenizer(model, cacheDir);
    // loadTokenizer returns null only for non-hf sources; entry.source==="hf" guarantees non-null.
    if (!tokenizer) {
      throw new TokenizerLoadError(`Tokenizer not loaded for ${model}`, model);
    }
    const encoded = tokenizer.encode(text);
    const tokens = encoded.ids.length;
    return { tokens, model: entry.model, source: "exact", bytes, chars };
  }

  if (entry.source === "tiktoken") {
    const tokens = getTiktokenCount(text, entry.encoding!);
    return { tokens, model: entry.model, source: "exact", bytes, chars };
  }

  // approx
  const baseCount = getTiktokenCount(text, entry.approxBase!);
  const tokens = Math.ceil(baseCount * (entry.approxMultiplier ?? 1));
  return { tokens, model: entry.model, source: "approx", bytes, chars };
}

/**
 * True if the model's count is an approximation rather than an exact
 * tokenizer count (currently: source === "approx").
 */
export function isApproxModel(model: string): boolean {
  const entry = lookupModel(model);
  return entry?.source === "approx";
}

/** All model keys the registry knows about. */
export function listSupportedModels(): string[] {
  return Object.keys(REGISTRY);
}