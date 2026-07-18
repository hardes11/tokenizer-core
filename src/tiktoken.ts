import { getEncoding, type Tiktoken } from "js-tiktoken";

/**
 * Cached tiktoken encoder instances, keyed by encoding name.
 * Repeated calls reuse the same Tiktoken instance to avoid the ~MB-scale
 * rank-table reload on every invocation.
 */
const encoderCache = new Map<"o200k_base" | "cl100k_base", Tiktoken>();

function getEncoder(encoding: "o200k_base" | "cl100k_base"): Tiktoken {
  let enc = encoderCache.get(encoding);
  if (!enc) {
    enc = getEncoding(encoding);
    encoderCache.set(encoding, enc);
  }
  return enc;
}

/**
 * Return the exact token count for `text` under the given tiktoken encoding.
 */
export function getTiktokenCount(
  text: string,
  encoding: "o200k_base" | "cl100k_base",
): number {
  return getEncoder(encoding).encode(text).length;
}