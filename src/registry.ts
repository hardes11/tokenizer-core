export type TokenizerSource = "hf" | "tiktoken" | "approx";

export interface ModelEntry {
  model: string;
  source: TokenizerSource;
  /** for "hf": e.g. "zai-org/GLM-5.2" */
  repo?: string;
  /** for "hf": pinned commit SHA */
  sha?: string;
  /** for "tiktoken" */
  encoding?: "o200k_base" | "cl100k_base";
  /** for "approx" */
  approxBase?: "o200k_base" | "cl100k_base";
  /** for "approx": e.g. 1.15 */
  approxMultiplier?: number;
}

export const REGISTRY: Record<string, ModelEntry> = {
  "glm-5.2":        { model: "glm-5.2",        source: "hf",       repo: "zai-org/GLM-5.2",         sha: "b4734de4facf877f85769a911abafc5283eab3d9" },
  "glm-5":          { model: "glm-5",          source: "hf",       repo: "zai-org/GLM-5",           sha: "4e6698ba8e85059d749020e3c4d2123719f23926" },
  "glm-4.6v-flash": { model: "glm-4.6v-flash", source: "hf",       repo: "zai-org/GLM-4.6V-Flash",  sha: "411bb4d77144a3f03accbf4b780f5acb8b7cde4e" },
  "gpt-5":          { model: "gpt-5",          source: "tiktoken", encoding: "o200k_base" },
  "gpt-4o":         { model: "gpt-4o",         source: "tiktoken", encoding: "o200k_base" },
  "gpt-4":          { model: "gpt-4",          source: "tiktoken", encoding: "cl100k_base" },
  "gpt-3.5":        { model: "gpt-3.5",        source: "tiktoken", encoding: "cl100k_base" },
  "qwen":           { model: "qwen",           source: "hf",       repo: "Qwen/Qwen3-8B",           sha: "b968826d9c46dd6066d109eabc6255188de91218" },
  "deepseek-v3.1":  { model: "deepseek-v3.1",  source: "hf",       repo: "deepseek-ai/DeepSeek-V3.1", sha: "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db" },
  "claude":         { model: "claude",         source: "approx",   approxBase: "o200k_base", approxMultiplier: 1.15 },
  "gemini":         { model: "gemini",         source: "approx",   approxBase: "o200k_base", approxMultiplier: 1.15 },
};

export function lookupModel(model: string): ModelEntry | undefined {
  return REGISTRY[model.toLowerCase()];
}