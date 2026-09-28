// Pure config resolution for scripts/smoke.mjs: which model ids a box serves.
//
// The compose defaults live in compose*.yml; `.env` holds the operator's
// overrides (.env.example catalogues them). Resolution order per id: explicit
// environment, then `.env`, then the compose default — the same precedence
// compose itself honours. Kept apart from the suite so it can be tested
// without a router (scripts/smoke-config.test.mjs), like router/backends.mjs.

import { readFileSync } from "node:fs";

/** Parse `.env` text into an object; ignores blanks, comments, and bad lines. */
export function parseDotEnv(text) {
  const out = {};
  for (const line of String(text).split("\n")) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (match === null) continue;
    let value = match[2];
    if (value.length >= 2 && "\"'".includes(value[0]) && value.endsWith(value[0])) {
      value = value.slice(1, -1);
    }
    out[match[1]] = value;
  }
  return out;
}

/** Read a `.env` file into an object; {} when absent or unreadable. */
export function loadDotEnv(path) {
  try {
    return parseDotEnv(readFileSync(path, "utf8"));
  } catch {
    return {};
  }
}

// The OmniJev size catalogue mirrors omnijev/server.py; an explicit
// OMNIJEV_CKPT beats it, exactly like the service.
const OMNI_BY_SIZE = { 0.8: "tinnel123/OmniJev-0.8B", 2: "tinnel123/OmniJev-2B", 4: "tinnel123/OmniJev" };
const OMNI_PREFIX = "tinnel123/OmniJev";

/** Every id the smoke suite checks, resolved from explicit env plus `.env`. */
export function resolveIds(env = {}, dotenv = {}) {
  const configured = (key, fallback) => env[key] ?? dotenv[key] ?? fallback;

  const VOYAGE = configured("MODEL_VOYAGE_EMBED", "voyageai/voyage-4-nano");
  const RERANKER = configured("MODEL_RERANKER", "Alibaba-NLP/gte-reranker-modernbert-base");

  // The Laya served id gains a subfolder suffix (laya/README.md: checkpoints).
  const LAYA_MODEL = configured("MODEL_LAYA", "convaiinnovations/laya");
  const LAYA_SUB = configured("LAYA_SUBFOLDER", "");
  const LAYA_ID = LAYA_SUB !== "" ? `${LAYA_MODEL}/${LAYA_SUB}` : LAYA_MODEL;

  const AGENTJEV_ID = configured("AGENTJEV_REPO", "aimeigaoshou/agent-jev");
  const JULIA_ID = configured("JULIA_REPO", "SupersonicLabs/Julia-1");
  const OMNI_ID =
    configured("OMNIJEV_CKPT", "") || OMNI_BY_SIZE[configured("OMNIJEV_SIZE", "0.8")] || OMNI_BY_SIZE[0.8];

  // Optional profiles: tested when advertised, skipped when absent.
  const EMBED_KNOWN = [
    VOYAGE,
    configured("MODEL_QWEN_EMBED", "Qwen/Qwen3-Embedding-0.6B"),
    configured("MODEL_JINA_EMBED", "jinaai/jina-embeddings-v5-omni-small"),
  ];
  const RERANK_KNOWN = [
    RERANKER,
    configured("MODEL_BGE_RERANKER", "BAAI/bge-reranker-v2-m3"),
    configured("MODEL_MS_MARCO", "cross-encoder/ms-marco-MiniLM-L6-v2"),
    configured("MODEL_GTE_MULTILINGUAL", "Alibaba-NLP/gte-multilingual-reranker-base"),
    configured("MODEL_GRANITE_RERANKER", "ibm-granite/granite-embedding-reranker-english-r2"),
    configured("MODEL_JINA_RERANKER", "jinaai/jina-reranker-v3.5"),
    AGENTJEV_ID,
  ];

  // A served id that is exactly the configured one, or a checkpoint variant
  // of the configured model (laya subfolders, omni sizes) — never a stranger's.
  const isLaya = (m) => m === LAYA_ID || (LAYA_MODEL !== "" && m.startsWith(`${LAYA_MODEL}/`));
  const isOmni = (m) => m === OMNI_ID || m.startsWith(OMNI_PREFIX);

  return {
    VOYAGE,
    RERANKER,
    EMBED_KNOWN,
    RERANK_KNOWN,
    LAYA_ID,
    AGENTJEV_ID,
    JULIA_ID,
    OMNI_ID,
    isLaya,
    isOmni,
  };
}
