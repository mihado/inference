// Runnable check for the router's backend-id parsing:
//   node --test router/backends.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  accessLine,
  addBackend,
  backendName,
  eligibleBackends,
  infoPaths,
  modelAdvertises,
  openAiModelIds,
  p50,
  pickBackend,
  recordStat,
  setBackendPaths,
  summarizeStats,
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

test("infoPaths keeps absolute paths, drops junk, dedupes", () => {
  assert.deepEqual(infoPaths({ model_id: "m", paths: ["/v1/novel", "/v1/novel", "/rerank"] }), [
    "/v1/novel",
    "/rerank",
  ]);
  assert.deepEqual(infoPaths({ model_id: "m", paths: ["relative", 42, null, ""] }), []);
  assert.deepEqual(infoPaths({ model_id: "m", paths: "nope" }), []);
  assert.deepEqual(infoPaths({ model_id: "m" }), []);
  assert.deepEqual(infoPaths(null), []);
});

test("setBackendPaths replaces per backend, and empty removes", () => {
  const found = new Map();
  setBackendPaths(found, "http://a:80", []);
  assert.equal(found.get("http://a:80"), undefined);
  setBackendPaths(found, "http://a:80", ["/a"]);
  setBackendPaths(found, "http://a:80", ["/b"]);
  assert.deepEqual([...found.get("http://a:80")], ["/b"]);
  setBackendPaths(found, "http://a:80", []);
  assert.equal(found.get("http://a:80"), undefined);
});

test("modelAdvertises answers whether any serving backend has the path", () => {
  const byBase = new Map([["http://a:80", new Set(["/v1/novel"])]]);
  assert.equal(modelAdvertises(["http://a:80", "http://b:80"], byBase, "/v1/novel"), true);
  assert.equal(modelAdvertises(["http://b:80"], byBase, "/v1/novel"), false);
  assert.equal(modelAdvertises(undefined, byBase, "/v1/novel"), false);
});

test("eligibleBackends keeps novel paths on the backends that advertised them", () => {
  const urls = ["http://a:80", "http://b:80"];
  const mixed = new Map([
    ["http://a:80", new Set(["/v1/novel"])],
    ["http://b:80", new Set(["/v1/legacy"])],
  ]);
  assert.deepEqual(eligibleBackends(urls, mixed, "/v1/novel", false), ["http://a:80"]);
  assert.deepEqual(eligibleBackends(urls, mixed, "/v1/legacy", false), ["http://b:80"]);
  assert.deepEqual(eligibleBackends(urls, mixed, "/rerank", true), urls);
  // A backend that advertises nothing stays eligible (legacy TEI/vLLM shape).
  assert.deepEqual(eligibleBackends(urls, new Map(), "/v1/novel", false), urls);
  assert.deepEqual(eligibleBackends(["http://b:80"], mixed, "/v1/novel", false), undefined);
  assert.equal(eligibleBackends(undefined, mixed, "/v1/novel", false), undefined);
  assert.equal(eligibleBackends([], mixed, "/v1/novel", false), undefined);
});

test("accessLine fits one line and strips forged newlines", () => {
  assert.equal(
    accessLine({ method: "POST", path: "/rerank", model: "m", backend: "http://a:80", status: 200, ms: 42 }),
    "router: POST /rerank model=m backend=http://a:80 status=200 42ms",
  );
  assert.equal(
    accessLine({ method: "POST", path: "/rerank", model: "a\nrouter: forged", backend: "-", status: 404, ms: 3 }),
    "router: POST /rerank model=a_router: forged backend=- status=404 3ms",
  );
});

test("accessLine strips ANSI escapes and tabs, keeps spaces and unicode", () => {
  assert.equal(
    accessLine({ method: "POST", path: "/r", model: "a\x1b[31mRED\x1b[0m", backend: "-", status: 200, ms: 1 }),
    "router: POST /r model=aRED backend=- status=200 1ms",
  );
  assert.equal(
    accessLine({ method: "POST", path: "/r", model: "m\tmodel=x", backend: "-", status: 200, ms: 1 }),
    "router: POST /r model=m_model=x backend=- status=200 1ms",
  );
  assert.equal(
    accessLine({ method: "POST", path: "/r", model: "my model héllo", backend: "-", status: 200, ms: 1 }),
    "router: POST /r model=my model héllo backend=- status=200 1ms",
  );
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

test("recordStat counts per model and backend; 5xx and missing backends err", () => {
  const models = {};
  recordStat(models, "m", "http://a:80", 200, 10);
  recordStat(models, "m", "http://b:80", 200, 30);
  recordStat(models, "m", "http://a:80", 502, 50);
  recordStat(models, "m", "-", 404, 1);
  recordStat(models, "m", "http://a:80", 400, 5);
  const entry = models.m;
  assert.equal(entry.requests, 5);
  assert.equal(entry.errors, 2);
  assert.equal(entry.backends["http://a:80"].requests, 3);
  assert.equal(entry.backends["http://a:80"].errors, 1);
  assert.equal(entry.backends["http://b:80"].errors, 0);
  assert.equal("-" in entry.backends, false);
});

test("p50 is the lower median, 0 when empty", () => {
  assert.equal(p50([]), 0);
  assert.equal(p50([40]), 40);
  assert.equal(p50([10, 30, 20]), 20);
  assert.equal(p50([10, 20, 30, 40]), 20);
});

test("recordStat ages samples past the ring", () => {
  const models = {};
  for (let i = 0; i < 300; i++) recordStat(models, "m", "http://a:80", 200, i);
  assert.equal(models.m.latencies.length, 256);
  assert.equal(summarizeStats(models).m.requests, 300);
});

test("summarizeStats hides samples, keeps counts and p50", () => {
  const models = {};
  recordStat(models, "m", "http://a:80", 200, 10);
  const view = summarizeStats(models);
  assert.deepEqual(view, {
    m: { requests: 1, errors: 0, p50_ms: 10, backends: { "http://a:80": { requests: 1, errors: 0, p50_ms: 10 } } },
  });
  assert.equal(JSON.parse(JSON.stringify(view)).m.p50_ms, 10);
});
