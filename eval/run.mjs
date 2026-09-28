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
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadDotEnv, resolveIds } from "../scripts/smoke-config.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROUTER = (process.env.ROUTER_URL ?? "http://localhost:8100").replace(/\/+$/, "");
const TIMEOUT_MS = Number(process.env.EVAL_TIMEOUT_MS ?? 120_000);
const V1_COMPAT = (process.env.EVAL_V1_COMPAT ?? "").replace(/\/+$/, "");

const SHARED = resolveIds(
  process.env,
  loadDotEnv(join(HERE, "..", ".env")),
);
const LOCAL_KNOWN = [...SHARED.RERANK_KNOWN, SHARED.LAYA_ID];
// Paid models and the endpoint that serves them; the adapter needs your
// v1-compat call shape (URL, auth, body) before these run.
const PAID_KNOWN = ["rerank-3-lite", "jev-latest"];

const golden = JSON.parse(readFileSync(join(HERE, "golden.json"), "utf8"));

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

/** Rank (1-based) of the relevant doc, tolerant of both response shapes. */
function relevantRank(body, relPos, count) {
  const results = Array.isArray(body) ? body : body?.results;
  if (!Array.isArray(results) || results.length !== count) return null;
  const order = [...results.map((entry) => entry?.index)].sort();
  if (order.join(",") !== [...Array(count).keys()].join(",")) return null;
  const scores = results.map((entry) => entry?.relevance_score ?? entry?.score);
  if (!scores.every(Number.isFinite)) return null;
  const at = results.findIndex((entry) => entry?.index === relPos);
  return at === -1 ? null : at + 1;
}

function paidAdapter() {
  // Needs the v1-compat call shape (endpoint, auth, body) — SKIPs until then.
  return { skipped: V1_COMPAT === "" ? "EVAL_V1_COMPAT unset" : "adapter pending call shape" };
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

for (const id of LOCAL_KNOWN) {
  if (!advertised.has(id)) {
    rows.push({ model: id, note: "SKIP (not advertised)" });
    continue;
  }
  let top1 = 0;
  let mrr = 0;
  let ms = 0;
  let ran = 0;
  for (const item of golden) {
    const texts = item.relevant_first ? [item.relevant, ...item.distractors] : [...item.distractors, item.relevant];
    const relPos = item.relevant_first ? 0 : texts.length - 1;
    let result;
    try {
      result = await post("/rerank", { model: id, query: item.query, texts });
    } catch (error) {
      console.log(`FAIL - ${id} / ${item.id}: ${String(error?.cause ?? error).slice(0, 120)}`);
      failures += 1;
      continue;
    }
    if (result.status !== 200) {
      console.log(`FAIL - ${id} / ${item.id}: status ${result.status}`);
      failures += 1;
      continue;
    }
    const rank = relevantRank(result.body, relPos, texts.length);
    if (rank === null) {
      console.log(`FAIL - ${id} / ${item.id}: bad shape`);
      failures += 1;
      continue;
    }
    if (rank === 1) top1 += 1;
    else if (process.env.EVAL_VERBOSE !== undefined) console.log(`miss - ${id} / ${item.id}: rank ${rank}`);
    mrr += 1 / rank;
    ms += result.ms;
    ran += 1;
  }
  rows.push({
    model: id,
    recall1: ran > 0 ? `${top1}/${ran} (${(top1 / ran).toFixed(3)})` : "n/a",
    mrr: ran > 0 ? (mrr / ran).toFixed(3) : "n/a",
    mean_ms: ran > 0 ? Math.round(ms / ran) : "n/a",
  });
}

for (const id of PAID_KNOWN) {
  rows.push({ model: id, note: `SKIP (${paidAdapter().skipped})` });
}

const width = Math.max(...rows.map((row) => row.model.length));
console.log(`\n${"model".padEnd(width)}  recall@1      MRR    mean_ms  note`);
for (const row of rows) {
  if (row.note !== undefined) console.log(`${row.model.padEnd(width)}  -            -      -        ${row.note}`);
  else console.log(`${row.model.padEnd(width)}  ${String(row.recall1).padEnd(12)} ${String(row.mrr).padEnd(6)} ${String(row.mean_ms).padEnd(7)}`);
}
console.log(failures > 0 ? `\neval: ${failures} request failures` : "\neval: clean");
process.exit(failures > 0 ? 1 : 0);
