#!/usr/bin/env node
// Live smoke through the router only (clients never call a server port).
// One inference per model kind, asserting shape — not quality. Quality lives
// in docs/evaluation.md; this answers "is anything broken right now?".
//
//   node scripts/smoke.mjs                    # ROUTER_URL defaults below
//   make smoke                                 # ROUTER_URL from ROUTER_PORT
//   ROUTER_URL=http://box:8100 node scripts/smoke.mjs
//
// Knobs (all optional):
//   ROUTER_URL         router base (default http://localhost:8100)
//   SMOKE_TIMEOUT_MS   per-request timeout (default 180000; the first decision
//                      request compiles Triton kernels, so it can take minutes)
//   SMOKE_WAIT_MS      how long to wait for the expected models to appear in
//                      /v1/models (default 120000; the router rescans every 30s)
//   SMOKE_VOYAGE       expected embedder (default voyageai/voyage-4-nano)
//   SMOKE_RERANKER     expected reranker (default Alibaba-NLP/gte-reranker-modernbert-base)
//
// Exit 0 when every check passes or skips; 1 on the first failure path with
// any failure. Optional profiles that are not advertised are SKIP, not FAIL.
// Dependency-free (global fetch only), like router/index.mjs.

const ROUTER = (process.env.ROUTER_URL ?? "http://localhost:8100").replace(/\/+$/, "");
const TIMEOUT_MS = Number(process.env.SMOKE_TIMEOUT_MS ?? 180_000);
const WAIT_MS = Number(process.env.SMOKE_WAIT_MS ?? 120_000);
const VOYAGE = process.env.SMOKE_VOYAGE ?? "voyageai/voyage-4-nano";
const RERANKER = process.env.SMOKE_RERANKER ?? "Alibaba-NLP/gte-reranker-modernbert-base";

// Optional profiles: tested when advertised, skipped when absent.
const EMBED_KNOWN = [VOYAGE, "Qwen/Qwen3-Embedding-0.6B", "jinaai/jina-embeddings-v5-omni-small"];
const RERANK_KNOWN = [
  RERANKER,
  "BAAI/bge-reranker-v2-m3",
  "cross-encoder/ms-marco-MiniLM-L6-v2",
  "Alibaba-NLP/gte-multilingual-reranker-base",
  "ibm-granite/granite-embedding-reranker-english-r2",
  "jinaai/jina-reranker-v3.5",
  "aimeigaoshou/agent-jev",
];
// Prefix matches because a checkpoint variant changes the served id
// (laya/README.md: checkpoints; omnijev/README.md: OMNIJEV_SIZE).
const LAYA_PREFIX = "convaiinnovations/laya";
const OMNI_PREFIX = "tinnel123/OmniJev";
const JULIA_ID = "SupersonicLabs/Julia-1";
const AGENTJEV_ID = "aimeigaoshou/agent-jev";

let passed = 0;
let skipped = 0;
let failed = 0;

class Skip extends Error {}

function ok(name, detail = "") {
  passed += 1;
  console.log(`ok - ${name}${detail !== "" ? ` (${detail})` : ""}`);
}

function skip(name, reason) {
  skipped += 1;
  console.log(`SKIP - ${name}: ${reason}`);
}

function fail(name, reason) {
  failed += 1;
  console.log(`FAIL - ${name}: ${reason}`);
}

async function get(path) {
  const response = await fetch(`${ROUTER}${path}`, { signal: AbortSignal.timeout(TIMEOUT_MS) });
  const body = await response.json().catch(() => null);
  return { status: response.status, body };
}

async function post(path, payload) {
  const response = await fetch(`${ROUTER}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  const body = await response.json().catch(() => null);
  return { status: response.status, body };
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
}

function norm(vec) {
  return Math.sqrt(dot(vec, vec));
}

function cosine(a, b) {
  return dot(a, b) / (norm(a) * norm(b));
}

function assertEmbeddings(name, body, count) {
  if (body === null || !Array.isArray(body.data) || body.data.length !== count) {
    throw new Error(`expected data[${count}], got ${JSON.stringify(body)?.slice(0, 120)}`);
  }
  const dims = body.data[0]?.embedding?.length ?? 0;
  if (!(dims > 0)) throw new Error(`expected dims > 0, got ${dims}`);
  for (const [i, entry] of body.data.entries()) {
    if (!Array.isArray(entry?.embedding) || entry.embedding.length !== dims) {
      throw new Error(`row ${i}: ragged dims`);
    }
    if (!entry.embedding.every(Number.isFinite)) throw new Error(`row ${i}: non-finite value`);
    if (!(norm(entry.embedding) > 0.1)) throw new Error(`row ${i}: near-zero vector`);
  }
  return dims;
}

// --- catalogue -----------------------------------------------------------

let models = [];
try {
  const health = await get("/health");
  if (health.status === 200 && health.body?.status === "ok") {
    ok("router /health", `${health.body.models ?? "?"} models`);
  } else {
    fail("router /health", `status ${health.status}`);
  }
} catch (error) {
  fail("router /health", String(error?.cause ?? error).slice(0, 160));
  console.log(`\nsmoke: ${passed} passed, ${skipped} skipped, ${failed} failed`);
  process.exit(1);
}

// The router rescans backends every 30s, so a just-started model takes a scan
// to appear; poll instead of failing on a cold catalogue.
const deadline = Date.now() + WAIT_MS;
for (;;) {
  try {
    const { status, body } = await get("/v1/models");
    if (status === 200 && Array.isArray(body?.data)) {
      models = body.data.map((entry) => entry?.id).filter((id) => typeof id === "string");
    }
  } catch {
    models = [];
  }
  if (models.includes(VOYAGE) && models.includes(RERANKER)) break;
  if (Date.now() > deadline) break;
  await new Promise((resolve) => setTimeout(resolve, 5000));
}
console.log(`models: ${models.length > 0 ? models.join(", ") : "<none>"}`);
for (const id of [VOYAGE, RERANKER]) {
  if (models.includes(id)) ok(`model advertised: ${id}`);
  else fail(`model advertised: ${id}`, "absent after wait (still loading, or profile down?)");
}

// A model name nobody serves must 404, not 500: the router's contract.
try {
  const { status } = await post("/rerank", { model: "no-such-model", query: "q", texts: ["d"] });
  if (status === 404) ok("unknown model 404s");
  else fail("unknown model 404s", `status ${status}`);
} catch (error) {
  fail("unknown model 404s", String(error?.cause ?? error).slice(0, 160));
}

// --- embeddings ----------------------------------------------------------

const EMBED_INPUTS = [
  "A brewer may sell beer on the licensed premises.",
  "A brewer may sell beer at the taproom.",
  "Tax returns are due quarterly.",
];

for (const id of EMBED_KNOWN) {
  if (!models.includes(id)) {
    skip(`embed ${id}`, "not advertised");
    continue;
  }
  const name = `embed ${id}`;
  try {
    const { status, body } = await post("/v1/embeddings", { model: id, input: EMBED_INPUTS });
    if (status !== 200) throw new Error(`status ${status}: ${JSON.stringify(body)?.slice(0, 160)}`);
    const dims = assertEmbeddings(name, body, EMBED_INPUTS.length);
    const [a, b, c] = body.data.map((entry) => entry.embedding);
    const para = cosine(a, b);
    const unrelated = cosine(a, c);
    if (!(para > unrelated)) {
      throw new Error(`paraphrases not closer (${para.toFixed(3)} <= ${unrelated.toFixed(3)})`);
    }
    ok(name, `${dims}d cos_para=${para.toFixed(3)} cos_unrelated=${unrelated.toFixed(3)}`);
  } catch (error) {
    if (error instanceof Skip) skip(name, error.message);
    else fail(name, error.message.slice(0, 200));
  }
}

// --- rerank --------------------------------------------------------------
// `documents` (Cohere's shape) exercises the router's normalization to TEI's
// `texts` live; the unit pin is dispatch.test.mjs.

const RERANK_QUERY = "Can a brewer sell beer directly at the taproom?";
const RERANK_DOCS = [
  "A brewer may sell beer on the licensed premises.",
  "The label must carry a health warning.",
  "Tax returns are due quarterly.",
];

for (const id of RERANK_KNOWN) {
  if (!models.includes(id)) {
    skip(`rerank ${id}`, "not advertised");
    continue;
  }
  const name = `rerank ${id}`;
  try {
    const { status, body } = await post("/rerank", { model: id, query: RERANK_QUERY, documents: RERANK_DOCS });
    if (status !== 200) throw new Error(`status ${status}: ${JSON.stringify(body)?.slice(0, 160)}`);
    // TEI returns a bare array; the Python services wrap it in {results}.
    const results = Array.isArray(body) ? body : body?.results;
    if (!Array.isArray(results) || results.length !== RERANK_DOCS.length) {
      throw new Error(`expected results[${RERANK_DOCS.length}], got ${JSON.stringify(body)?.slice(0, 120)}`);
    }
    const order = [...results.map((entry) => entry?.index)].sort();
    if (order.join(",") !== "0,1,2") throw new Error(`indices not a permutation: ${order.join(",")}`);
    // TEI names it `score`, the Python services `relevance_score`; the router
    // forwards verbatim, so both shapes are live.
    const scores = results.map((entry) => entry?.relevance_score ?? entry?.score);
    if (!scores.every(Number.isFinite)) {
      throw new Error("non-finite score");
    }
    if (results[0].index !== 0) throw new Error(`relevant doc not first (order ${results.map((e) => e.index)})`);
    ok(name, `top=${scores[0]}`);
  } catch (error) {
    fail(name, error.message.slice(0, 200));
  }
}

// Laya doubles as a reranker under its own id; same shape, one request.
for (const id of models.filter((m) => m === LAYA_PREFIX || m.startsWith(`${LAYA_PREFIX}/`))) {
  if (RERANK_KNOWN.includes(id)) continue;
  skip(`rerank ${id}`, "covered below via /v1/decisions");
}

// --- decisions -----------------------------------------------------------

function assertProbMap(answerId, probs) {
  const values = Object.values(probs);
  if (values.length === 0 || !values.every((p) => Number.isFinite(p) && p >= 0 && p <= 1)) {
    throw new Error(`${answerId}: probabilities not in [0,1]`);
  }
  const sum = values.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 0.05) throw new Error(`${answerId}: probabilities sum to ${sum.toFixed(3)}`);
}

const layaId = models.find((m) => m === LAYA_PREFIX || m.startsWith(`${LAYA_PREFIX}/`));
if (layaId === undefined) {
  skip("decisions laya", "not advertised");
} else {
  const name = `decisions ${layaId}`;
  try {
    const { status, body } = await post("/v1/decisions", {
      model: layaId,
      state: "We were billed twice for March. Refund today or we cancel.",
      preset: "triage",
    });
    if (status !== 200) throw new Error(`status ${status}: ${JSON.stringify(body)?.slice(0, 160)}`);
    if (body === null || typeof body !== "object" || body.error !== undefined) {
      throw new Error(`no answers: ${JSON.stringify(body)?.slice(0, 120)}`);
    }
    if (body.answers !== undefined) {
      const entries = Object.entries(body.answers);
      if (entries.length === 0) throw new Error("empty answers");
      for (const [qid, answer] of entries) {
        if (answer?.probabilities !== undefined) assertProbMap(`${name}/${qid}`, answer.probabilities);
      }
      ok(name, `${entries.length} answers`);
    } else {
      ok(name, `keys: ${Object.keys(body).join(",")}`.slice(0, 80));
    }
  } catch (error) {
    fail(name, error.message.slice(0, 200));
  }
}

if (!models.includes(AGENTJEV_ID)) {
  skip("decisions agentjev", "not advertised");
} else {
  const name = `decisions ${AGENTJEV_ID}`;
  try {
    const { status, body } = await post("/api/evaluate", {
      model: AGENTJEV_ID,
      state: "Tests run: 25, Failures: 0.",
      questions: [{ id: "done", type: "boolean", question: "Are all tests passing?" }],
    });
    if (status !== 200) throw new Error(`status ${status}: ${JSON.stringify(body)?.slice(0, 160)}`);
    const probability = body?.results?.[0]?.answers?.[0]?.probability;
    if (!(Number.isFinite(probability) && probability >= 0 && probability <= 1)) {
      throw new Error(`bad probability: ${JSON.stringify(body)?.slice(0, 160)}`);
    }
    ok(name, `P=${probability.toFixed(3)}`);
  } catch (error) {
    fail(name, error.message.slice(0, 200));
  }
}

// A 1x1 PNG; the content is irrelevant, only that it decodes.
const PIXEL = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
const omniId = models.find((m) => m.startsWith(OMNI_PREFIX));
if (omniId === undefined) {
  skip("decisions omnijev", "not advertised");
} else {
  const name = `decisions ${omniId}`;
  try {
    const { status, body } = await post("/v1/systemone", {
      model: omniId,
      state: { images: [`data:image/png;base64,${PIXEL}`] },
      questions: { red: { type: "noul", instructions: "The image is a red square." } },
    });
    if (status !== 200) throw new Error(`status ${status}: ${JSON.stringify(body)?.slice(0, 160)}`);
    const answers = body?.answers;
    if (answers === null || typeof answers !== "object" || Object.keys(answers).length === 0) {
      throw new Error(`no answers: ${JSON.stringify(body)?.slice(0, 160)}`);
    }
    ok(name, `${Object.keys(answers).length} answers`);
  } catch (error) {
    fail(name, error.message.slice(0, 200));
  }
}

if (!models.includes(JULIA_ID)) {
  skip("decisions julia", "not advertised");
} else {
  const name = `decisions ${JULIA_ID}`;
  try {
    const { status, body } = await post("/v1/predict", {
      model: JULIA_ID,
      state: "I was charged twice for the same order.",
      questions: {
        team: {
          type: "choice",
          instructions: "Which team should handle this request?",
          criteria: { billing: "Billing and payment disputes", shipping: "Shipping and delivery" },
        },
      },
    });
    if (status !== 200) throw new Error(`status ${status}: ${JSON.stringify(body)?.slice(0, 160)}`);
    if (body?.object !== "predict" || typeof body?.answers?.team?.probabilities !== "object") {
      throw new Error(`bad shape: ${JSON.stringify(body)?.slice(0, 160)}`);
    }
    assertProbMap(`${name}/team`, body.answers.team.probabilities);
    ok(name, `team=${JSON.stringify(body.answers.team.probabilities)}`.slice(0, 100));
  } catch (error) {
    fail(name, error.message.slice(0, 200));
  }
}

// Anything advertised that no check above covers is a gap in this script,
// reported — not passed.
const covered = new Set([...EMBED_KNOWN, ...RERANK_KNOWN, AGENTJEV_ID, JULIA_ID]);
for (const id of models) {
  if (covered.has(id) || id === LAYA_PREFIX || id.startsWith(`${LAYA_PREFIX}/`) || id.startsWith(OMNI_PREFIX)) continue;
  skip(`unknown kind ${id}`, "no smoke check yet — extend scripts/smoke.mjs");
}

console.log(`\nsmoke: ${passed} passed, ${skipped} skipped, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
