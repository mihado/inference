#!/usr/bin/env node
// Reranker bake-off over the golden subset (eval/golden.json): fast
// regression, not a serving decision. The full 500-Q held-out set in
// docs/evaluation.md stays authoritative; this catches breakage, model
// swaps, and shape drift in seconds, through the router only.
//
//   node eval/run.mjs                       # ROUTER_URL defaults below
//   make eval-golden                        # ROUTER_URL from ROUTER_PORT
//
// Every advertised reranker runs the same 20 cases; unadvertised profiles
// SKIP. Paid APIs (Voyage, TypeSafe) route through your v1-compat endpoint
// once EVAL_V1_COMPAT is set and the adapter below is filled in — until
// then they SKIP with the reason. Quality deltas never fail the run;
// request errors do. Dependency-free (global fetch only).
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadDotEnv, resolveIds } from "../scripts/smoke-config.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROUTER = (process.env.ROUTER_URL ?? "http://localhost:8100").replace(/\/+$/, "");
const TIMEOUT_MS = Number(process.env.EVAL_TIMEOUT_MS ?? 120_000);
// Paid models go through the product gateway (v1 compat): POST {base}/v1/rerank
// with {model, query, documents} -> {results: [{index, relevance_score}]}.
// The key is LEXLAB_OPENAPI_KEY in the environment; UA must look like curl or
// Cloudflare answers the API host with 403.
const V1_COMPAT = (process.env.EVAL_V1_COMPAT ?? "https://lexlab-api.rter.cc").replace(/\/+$/, "");
const V1_KEY = process.env.LEXLAB_OPENAPI_KEY ?? "";

const SHARED = resolveIds(
  process.env,
  loadDotEnv(join(HERE, "..", ".env")),
);
// Every reranker the stack can serve, from the compose ids. EVAL_LOCAL
// narrows it to a subset, so re-measuring the new arrivals does not re-run the
// whole field: an id nothing advertises is skipped, never failed.
const LOCAL_KNOWN = (process.env.EVAL_LOCAL ?? [...SHARED.RERANK_KNOWN, SHARED.LAYA_ID].join(","))
  .split(",")
  .map((id) => id.trim())
  .filter(Boolean);
// Paid models and the endpoint that serves them; the adapter needs your
// v1-compat call shape (URL, auth, body) before these run.
const PAID_KNOWN = (process.env.EVAL_PAID ?? "rerank-3-lite,jev-latest").split(",").map((id) => id.trim()).filter(Boolean);

const golden = JSON.parse(readFileSync(join(HERE, "golden.json"), "utf8"));
// CI tier: the first N cases, a stable prefix of the file.
const LIMIT = Number(process.env.EVAL_LIMIT ?? 0);
const cases = LIMIT > 0 ? golden.slice(0, LIMIT) : golden;

async function get(path) {
  const response = await fetch(`${ROUTER}${path}`, { signal: AbortSignal.timeout(TIMEOUT_MS) });
  return { status: response.status, body: await response.json().catch(() => null) };
}

async function post(path, payload) {
  const started = Date.now();
  const response = await fetch(`${ROUTER}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  return { status: response.status, body: await response.json().catch(() => null), ms: Date.now() - started };
}

// Cohere's trial tier answers 10 rerank requests a minute, and the gateway
// answers a cooled key pool with 429 `insufficient_quota`. A tight loop trips
// both, and one refusal used to abandon the whole model — so a rate limit cost
// a column instead of a pause. Calls are spaced per model, and a throttled or
// transient answer is retried. The wait sits outside the measured window, so
// mean_ms stays the provider's latency and not our own throttle.
const MIN_INTERVAL_MS = Number(process.env.EVAL_MIN_INTERVAL_MS ?? 0);
const MAX_ATTEMPTS = Number(process.env.EVAL_ATTEMPTS ?? 4);
const lastSentAt = new Map();

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** `Retry-After` in ms, or null when absent or unparsable. */
function retryAfterMs(response) {
  const raw = response.headers.get("retry-after");
  if (raw === null) return null;
  const seconds = Number(raw);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds * 1000 : null;
}

/** One rerank through the product gateway; same response shape as ours.
 * Retries what a shared lab or a provider can answer transiently: a throttled
 * request, any 5xx, or a dropped socket. A 502 with a non-JSON body is the
 * edge in front of the gateway answering — a tight loop meets it on every
 * deploy — and it is not a statement about the model's quality. */
async function postV1(payload) {
  const gap = MIN_INTERVAL_MS - (Date.now() - (lastSentAt.get(payload.model) ?? 0));
  if (gap > 0) await sleep(gap);
  for (let attempt = 1; ; attempt += 1) {
    const started = Date.now();
    lastSentAt.set(payload.model, started);
    const last = attempt >= MAX_ATTEMPTS;
    try {
      const response = await fetch(`${V1_COMPAT}/v1/rerank`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${V1_KEY}`,
          "content-type": "application/json",
          "user-agent": "curl/8.0",
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
      const body = await response.json().catch(() => null);
      if (!(response.status === 429 || response.status >= 500) || last) {
        return { status: response.status, body, ms: Date.now() - started };
      }
      console.log(`  wait - ${payload.model}: ${response.status}, retry ${attempt}/${MAX_ATTEMPTS - 1}`);
      await sleep(retryAfterMs(response) ?? 5_000 * attempt);
    } catch (error) {
      if (last) throw error;
      console.log(
        `  wait - ${payload.model}: ${String(error?.cause ?? error).slice(0, 90)}, retry ${attempt}/${MAX_ATTEMPTS - 1}`,
      );
      await sleep(5_000 * attempt);
    }
  }
}

/** Rank (1-based) of the relevant doc, tolerant of both response shapes. */
function relevantRank(body, relPos, count) {
  const results = Array.isArray(body) ? body : body?.results;
  if (!Array.isArray(results) || results.length !== count) return null;
  // Numeric sort. A bare .sort() orders indices as strings, so at 11 or more
  // documents it yields 0,1,10,11,...,2 and every well-formed response is
  // rejected as malformed. Six-document pools cannot catch this.
  const order = [...results.map((entry) => entry?.index)].sort((a, b) => a - b);
  if (order.join(",") !== [...Array(count).keys()].join(",")) return null;
  const scores = results.map((entry) => entry?.relevance_score ?? entry?.score);
  if (!scores.every(Number.isFinite)) return null;
  const at = results.findIndex((entry) => entry?.index === relPos);
  return at === -1 ? null : at + 1;
}

const { body: catalogue } = await get("/v1/models").catch(() => ({ body: null }));
const advertised = new Set(
  (Array.isArray(catalogue?.data) ? catalogue.data : []).map((entry) => entry?.id).filter((id) => typeof id === "string"),
);
if (advertised.size === 0) {
  console.log("FAIL - no models advertised; is the router up?");
  process.exit(1);
}

let failures = 0;
const rows = [];
// Per-case ranks, so two rerankers can be compared pairwise rather than by
// aggregate: an aggregate cannot tell a 2-case gap from a 20-case one.
const perCase = [];

async function score(id, send) {
  let top1 = 0;
  let mrr = 0;
  let ms = 0;
  let ran = 0;
  for (const item of cases) {
    const texts = item.relevant_first ? [item.relevant, ...item.distractors] : [...item.distractors, item.relevant];
    const relPos = item.relevant_first ? 0 : texts.length - 1;
    let result;
    try {
      result = await send({ model: id, query: item.query, texts });
    } catch (error) {
      console.log(`FAIL - ${id} / ${item.id}: ${String(error?.cause ?? error).slice(0, 120)}`);
      return null;
    }
    if (result.status !== 200) {
      console.log(`FAIL - ${id} / ${item.id}: status ${result.status} ${JSON.stringify(result.body)?.slice(0, 140)}`);
      return null;
    }
    const rank = relevantRank(result.body, relPos, texts.length);
    if (rank === null) {
      console.log(`FAIL - ${id} / ${item.id}: bad shape`);
      return null;
    }
    if (rank === 1) top1 += 1;
    else if (process.env.EVAL_VERBOSE !== undefined) console.log(`miss - ${id} / ${item.id}: rank ${rank}`);
    perCase.push({ model: id, id: item.id, rank, ms: result.ms });
    mrr += 1 / rank;
    ms += result.ms;
    ran += 1;
  }
  return {
    recall1: ran > 0 ? `${top1}/${ran} (${(top1 / ran).toFixed(3)})` : "n/a",
    mrr: ran > 0 ? (mrr / ran).toFixed(3) : "n/a",
    mean_ms: ran > 0 ? Math.round(ms / ran) : "n/a",
  };
}

for (const id of LOCAL_KNOWN) {
  if (!advertised.has(id)) {
    rows.push({ model: id, note: "SKIP (not advertised)" });
    continue;
  }
  const row = await score(id, (payload) => post("/rerank", payload));
  if (row === null) failures += 1;
  else rows.push({ model: id, ...row });
}

for (const id of PAID_KNOWN) {
  if (V1_KEY === "") {
    rows.push({ model: id, note: "SKIP (LEXLAB_OPENAPI_KEY unset)" });
    continue;
  }
  // The gateway takes Cohere's `documents`, not TEI's `texts`.
  const row = await score(id, ({ model, query, texts }) => postV1({ model, query, documents: texts }));
  if (row === null) failures += 1;
  else rows.push({ model: id, ...row, paid: "v1-compat" });
}

const width = Math.max(...rows.map((row) => row.model.length));
console.log(`\n${"model".padEnd(width)}  recall@1      MRR    mean_ms  note`);
for (const row of rows) {
  const note = row.note ?? row.paid ?? "";
  if (row.note !== undefined || row.mrr === undefined) {
    console.log(`${row.model.padEnd(width)}  ${"-".padEnd(12)} ${"-".padEnd(6)} ${"-".padEnd(7)} ${row.note ?? ""}`);
  } else {
    console.log(`${row.model.padEnd(width)}  ${String(row.recall1).padEnd(12)} ${String(row.mrr).padEnd(6)} ${String(row.mean_ms).padEnd(7)} ${note}`);
  }
}
console.log(failures > 0 ? `\neval: ${failures} request failures` : "\neval: clean");
if (process.env.EVAL_PERCASE !== undefined) {
  writeFileSync(join(HERE, process.env.EVAL_PERCASE), perCase.map((row) => JSON.stringify(row)).join("\n") + "\n");
  console.log(`per-case rows: ${perCase.length} -> ${process.env.EVAL_PERCASE}`);
}
process.exit(failures > 0 ? 1 : 0);
