// Runnable check for the router's backend-id parsing:
//   node --test router/backends.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  addBackend,
  backendName,
  openAiModelIds,
  pickBackend,
  teiModelId,
} from "./backends.mjs";

const MODEL = "voyageai/voyage-4-nano";
/** The two voyage-embed replicas as the Docker API reports them: compose names
 * the container with its project, while the network answers to the service. */
const VOYAGE = { Names: ["/inference-voyage-embed-1"], Labels: { "com.docker.compose.service": "voyage-embed" } };
const VOYAGE_B = { Names: ["/inference-voyage-embed-b-1"], Labels: { "com.docker.compose.service": "voyage-embed-b" } };

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

test("backendName prefers the compose service, then the container's own name", () => {
  assert.equal(backendName(VOYAGE), "voyage-embed");
  assert.equal(backendName(VOYAGE_B), "voyage-embed-b");
  // A plain `docker run` container carries no compose label.
  assert.equal(backendName({ Names: ["/tei-bge-m3"], Labels: {} }), "tei-bge-m3");
  assert.equal(backendName({ Names: ["/tei-bge-m3"] }), "tei-bge-m3");
  // An empty label is not a name.
  assert.equal(backendName({ Names: ["/x-1"], Labels: { "com.docker.compose.service": "" } }), "x-1");
  assert.equal(backendName({}), "");
});

test("a container is one entry in the rotation, whichever name found it", () => {
  // Discovery finds the container; a hand-written BACKENDS entry names the
  // service. Both are the same container. While the container name was the
  // identity these were two entries, so voyage-embed took two of every three
  // requests and voyage-embed-b served at half rate for no reason.
  const found = new Map();
  addBackend(found, MODEL, `http://${backendName(VOYAGE)}:80`);
  addBackend(found, MODEL, "http://voyage-embed:80");
  addBackend(found, MODEL, `http://${backendName(VOYAGE_B)}:80`);

  assert.deepEqual(found.get(MODEL), ["http://voyage-embed:80", "http://voyage-embed-b:80"]);
  assert.equal(pickBackend(found.get(MODEL), 0), "http://voyage-embed:80");
  assert.equal(pickBackend(found.get(MODEL), 1), "http://voyage-embed-b:80");
  assert.equal(pickBackend(found.get(MODEL), 2), "http://voyage-embed:80");
});
