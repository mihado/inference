// Runnable check for the router's backend-id parsing:
//   node --test router/backends.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { addBackend, openAiModelIds, pickBackend, teiModelId } from "./backends.mjs";

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

test("addBackend keeps every replica, in discovery order, without duplicates", () => {
  const found = new Map();
  addBackend(found, "nano", "http://a:80");
  addBackend(found, "nano", "http://b:80");
  // A rescan sees the same backends again and must not grow the list.
  addBackend(found, "nano", "http://a:80");
  assert.deepEqual(found.get("nano"), ["http://a:80", "http://b:80"]);
});

test("addBackend replaces nothing when one model has one backend", () => {
  const found = new Map();
  addBackend(found, "qwen", "http://c:80");
  assert.deepEqual(found.get("qwen"), ["http://c:80"]);
});

test("pickBackend cycles in order, and answers undefined for an unknown model", () => {
  const urls = ["http://a:80", "http://b:80"];
  assert.equal(pickBackend(urls, 0), "http://a:80");
  assert.equal(pickBackend(urls, 1), "http://b:80");
  assert.equal(pickBackend(urls, 2), "http://a:80");
  assert.equal(pickBackend(urls, 3), "http://b:80");
  assert.equal(pickBackend(undefined, 0), undefined);
  assert.equal(pickBackend([], 0), undefined);
});
