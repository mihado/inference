// Runnable check for the router's backend-id parsing:
//   node --test router/backends.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { openAiModelIds, teiModelId } from "./backends.mjs";

test("teiModelId reads TEI's /info id, else null", () => {
  assert.equal(teiModelId({ model_id: "Qwen/Qwen3-Embedding-0.6B" }), "Qwen/Qwen3-Embedding-0.6B");
  assert.equal(teiModelId({ model_id: 42 }), null);
  assert.equal(teiModelId({}), null);
  assert.equal(teiModelId(null), null);
  assert.equal(teiModelId(undefined), null);
});

test("openAiModelIds reads an OpenAI list and drops junk", () => {
  assert.deepEqual(
    openAiModelIds({ data: [{ id: "voyageai/voyage-4-nano" }, { id: "" }, { nope: 1 }, "x"] }),
    ["voyageai/voyage-4-nano"],
  );
  assert.deepEqual(openAiModelIds({ data: [] }), []);
  assert.deepEqual(openAiModelIds({}), []);
  assert.deepEqual(openAiModelIds(null), []);
});
