// Runnable check for the smoke suite's id resolution:
//   node --test scripts/smoke-config.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadDotEnv, parseDotEnv, resolveIds } from "./smoke-config.mjs";

test("parseDotEnv reads KEY=value, quotes, and skips junk", () => {
  assert.deepEqual(
    parseDotEnv(
      [
        "# a comment",
        "",
        "MODEL_RERANKER=BAAI/bge-reranker-v2-m3",
        "QUOTED=\"hello world\"",
        "SINGLE='a b'",
        "SPACED = spaced value",
        "EMPTY=",
        "not a pair",
        "  # indented comment",
      ].join("\n"),
    ),
    {
      MODEL_RERANKER: "BAAI/bge-reranker-v2-m3",
      QUOTED: "hello world",
      SINGLE: "a b",
      SPACED: "spaced value",
      EMPTY: "",
    },
  );
});

test("loadDotEnv answers {} for a missing file", () => {
  assert.deepEqual(loadDotEnv("/nonexistent-smoke-test.env"), {});
});

test("resolveIds answers the compose defaults with empty inputs", () => {
  const ids = resolveIds({}, {});
  assert.equal(ids.VOYAGE, "voyageai/voyage-4-nano");
  assert.equal(ids.RERANKER, "Alibaba-NLP/gte-reranker-modernbert-base");
  assert.equal(ids.LAYA_ID, "convaiinnovations/laya");
  assert.equal(ids.AGENTJEV_ID, "aimeigaoshou/agent-jev");
  assert.equal(ids.JULIA_ID, "SupersonicLabs/Julia-1");
  assert.equal(ids.OMNI_ID, "tinnel123/OmniJev-0.8B");
  assert.ok(ids.EMBED_KNOWN.includes("Qwen/Qwen3-Embedding-0.6B"));
  assert.ok(ids.RERANK_KNOWN.includes("BAAI/bge-reranker-v2-m3"));
});

test("resolveIds prefers explicit env over .env over the default", () => {
  assert.equal(resolveIds({}, { MODEL_RERANKER: "dotenv-model" }).RERANKER, "dotenv-model");
  assert.equal(
    resolveIds({ MODEL_RERANKER: "env-model" }, { MODEL_RERANKER: "dotenv-model" }).RERANKER,
    "env-model",
  );
});

test("resolveIds builds the Laya subfolder id", () => {
  assert.equal(
    resolveIds({}, { MODEL_LAYA: "convaiinnovations/laya", LAYA_SUBFOLDER: "typed-decisions" }).LAYA_ID,
    "convaiinnovations/laya/typed-decisions",
  );
});

test("resolveIds maps the OmniJev size, and a CKPT beats it", () => {
  assert.equal(resolveIds({}, { OMNIJEV_SIZE: "2" }).OMNI_ID, "tinnel123/OmniJev-2B");
  assert.equal(
    resolveIds({ OMNIJEV_CKPT: "local/omni" }, { OMNIJEV_SIZE: "2" }).OMNI_ID,
    "local/omni",
  );
  assert.equal(resolveIds({}, { OMNIJEV_SIZE: "bogus" }).OMNI_ID, "tinnel123/OmniJev-0.8B");
});

test("isLaya and isOmni match the configured id and its variants only", () => {
  const ids = resolveIds({}, {});
  assert.equal(ids.isLaya("convaiinnovations/laya"), true);
  assert.equal(ids.isLaya("convaiinnovations/laya/typed-decisions"), true);
  assert.equal(ids.isLaya("someone-else/laya"), false);
  assert.equal(ids.isOmni("tinnel123/OmniJev-0.8B"), true);
  assert.equal(ids.isOmni("tinnel123/OmniJev-2B"), true);
  assert.equal(ids.isOmni("someone-else/OmniJev-0.8B"), false);

  const custom = resolveIds({}, { OMNIJEV_CKPT: "local/omni" });
  assert.equal(custom.isOmni("local/omni"), true);
  assert.equal(custom.isOmni("tinnel123/OmniJev-0.8B"), true);
});
