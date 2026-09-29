// Model-aware reverse proxy for a pool of single-model backends.
//
// One endpoint that fronts N single-model servers (TEI and OpenAI-shaped, e.g.
// vLLM): it reports the union of their models and dispatches each
// model-carrying POST (/v1/embeddings, /rerank, /v1/decisions and the rest of
// the static set below, plus whatever backends advertise in /info `paths`)
// to the server serving the requested model. Dependency-free (node:http + fetch).
//
// Backends are the labelled containers on its network (`tei.backend=1`), found
// through the Docker socket and re-scanned on a TTL. With no socket, set
// BACKENDS to the comma-separated backend URLs instead, each named by its
// service — the same identity the discovery uses.
//
// Model -> backend comes from each server's TEI /info (`{ model_id }`) or, for
// OpenAI-shaped servers like vLLM, its /v1/models list. Traefik cannot dispatch
// on a JSON body, which is why this exists; put TLS/ingress in front of it if
// you need it.

import { createServer, request as httpRequest } from "node:http";
import { existsSync } from "node:fs";

import { accessLine, addBackend, backendName, eligibleBackends, infoMaxClientBatchSize, infoPaths, minClientBatchSize, openAiModelIds, pathServed, pickBackend, recordStat, rerankDocuments, setBackendPaths, summarizeStats, teiModelId } from "./backends.mjs";

const PORT = Number(process.env.PORT ?? 80);
const BACKENDS = (process.env.BACKENDS ?? "")
  .split(",")
  .map((entry) => entry.trim().replace(/\/+$/, ""))
  .filter((entry) => entry.length > 0);
const MODEL_TTL_MS = Number(process.env.MODEL_TTL_MS ?? 30_000);
/** A model-listing probe must answer fast; a slow backend is not routable. */
const PROBE_TIMEOUT_MS = 5_000;
/** Docker socket for label discovery of ad-hoc slots; absent -> static only. */
const DOCKER_SOCK = process.env.DOCKER_SOCK ?? "/var/run/docker.sock";
const BACKEND_LABEL = process.env.BACKEND_LABEL ?? "tei.backend";
/** Largest request body accepted; larger answers 413 before JSON parsing. */
const MAX_BODY_BYTES = Number(process.env.MAX_BODY_BYTES ?? 25_000_000);
/** Upstream inference budget per request; exceeded answers 502. */
const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS ?? 300_000);
/** Docker socket budget; exceeded fails the catalogue probe. */
const DOCKER_TIMEOUT_MS = Number(process.env.DOCKER_TIMEOUT_MS ?? 10_000);

/** model id -> every backend base url that serves it, in discovery order. A
 * model served by two GPUs has two entries, and the router takes turns. */
let modelBackend = new Map();
/** backend base url -> POST paths it advertised in /info (`paths`). A backend
 * that omits `paths` (TEI, vLLM) has no entry and routes the static set. */
let backendPaths = new Map();
/** backend base url -> the client batch bound its /info reported, absent when
 * the backend reports none. */
let backendBatch = new Map();
/** model id -> request/error counts plus p50 latencies, per model and per
 * backend, since boot. Unknown-model probes ("-") are logged but not kept. */
const stats = Object.create(null);
const startedAt = Date.now();
/** model id -> how many requests it has been asked for. Kept outside the scan, so
 * a rescan does not reset the rotation. */
const turns = new Map();
let lastScan = 0;
let scanning = null;

/** One Docker API GET over the socket; rejects on any failure or timeout, so
 * a stalled daemon fails the catalogue probe instead of wedging the router. */
function dockerJson(path) {
  return new Promise((resolve, reject) => {
    let done = false;
    const finish = (fn, value) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      fn(value);
    };
    const req = httpRequest({ socketPath: DOCKER_SOCK, path, method: "GET" }, (res) => {
      let data = "";
      res.on("data", (chunk) => {
        data += chunk;
      });
      res.on("end", () => {
        if (res.statusCode !== 200) return finish(reject, new Error(`docker ${res.statusCode}`));
        try {
          finish(resolve, JSON.parse(data));
        } catch (error) {
          finish(reject, error);
        }
      });
    });
    const timer = setTimeout(() => req.destroy(new Error(`docker ${path} timed out`)), DOCKER_TIMEOUT_MS);
    req.on("error", (error) => finish(reject, error));
    req.end();
  });
}

/** Base URLs of running containers labelled BACKEND_LABEL, reachable by name on
 * the shared compose network. Empty when the socket is absent or unreachable.
 * Each is named by `backendName`, so a container is one entry here and the same
 * entry a static BACKENDS list would name. */
async function dockerBackends() {
  if (!existsSync(DOCKER_SOCK)) return [];
  const filters = encodeURIComponent(JSON.stringify({ label: [`${BACKEND_LABEL}=1`] }));
  const containers = await dockerJson(`/containers/json?filters=${filters}`);
  const urls = [];
  for (const container of Array.isArray(containers) ? containers : []) {
    const name = backendName(container);
    if (name.length > 0) urls.push(`http://${name}:80`);
  }
  return urls;
}

/** One backend GET as JSON, or null on any fault — a backend that is down,
 * slow, or answers non-JSON is simply not routable. */
async function fetchJson(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

async function scan() {
  const found = new Map();
  const paths = new Map();
  const bounds = new Map();
  const discovered = await dockerBackends().catch(() => []);
  const bases = [...new Set([...BACKENDS, ...discovered])];
  await Promise.all(
    bases.map(async (base) => {
      for (const { id, paths: advertised, maxClientBatchSize } of await servedModels(base)) {
        addBackend(found, id, base);
        setBackendPaths(paths, base, advertised);
        if (maxClientBatchSize !== null) bounds.set(base, maxClientBatchSize);
      }
    }),
  );
  modelBackend = found;
  backendPaths = paths;
  backendBatch = bounds;
  lastScan = Date.now();
}

/** Every model one backend serves, with what its /info advertises: the POST
 * paths and the client batch bound. TEI names its one model in /info;
 * anything else is asked for an OpenAI-style list, which advertises neither. */
async function servedModels(base) {
  const info = await fetchJson(`${base}/info`);
  const teiId = teiModelId(info);
  if (teiId !== null) {
    return [
      {
        id: teiId,
        paths: infoPaths(info),
        maxClientBatchSize: infoMaxClientBatchSize(info),
      },
    ];
  }
  return openAiModelIds(await fetchJson(`${base}/v1/models`)).map((id) => ({
    id,
    paths: [],
    maxClientBatchSize: null,
  }));
}

function ensureFresh() {
  if (Date.now() - lastScan < MODEL_TTL_MS) return Promise.resolve();
  scanning ??= scan().finally(() => {
    scanning = null;
  });
  return scanning;
}

/** The next eligible backend in a model's rotation, after a fresh catalogue.
 * The turn advances only for served models, so junk model names cannot grow
 * the map without bound. */
async function resolveBackend(model, path, isStatic) {
  await ensureFresh();
  const urls = eligibleBackends(modelBackend.get(model), backendPaths, path, isStatic);
  if (urls === undefined) return undefined;
  const turn = turns.get(model) ?? 0;
  turns.set(model, turn + 1);
  return pickBackend(urls, turn);
}

/** The static POST paths that carry a model in their JSON body. Backends may
 * advertise more in /info (`paths`, see README.md "Router"); the union is
 * what routes. */
const POST_PATHS = new Set(["/v1/embeddings", "/rerank", "/v1/rerank", "/v1/decisions", "/v1/systemone", "/v1/predict", "/api/evaluate"]);

/** Maps a public path to the backend path and body: rerank accepts Cohere's
 * `documents` for TEI's `texts`, and keeps `model` so a backend's
 * model-mismatch 404 stays meaningful instead of vacuous (TEI ignores the
 * extra field). Everything else forwards untouched. */
function backendRequest(path, body) {
  if (path === "/rerank" || path === "/v1/rerank") {
    return {
      path: "/rerank",
      body: { query: body.query, texts: rerankDocuments(body), model: body.model ?? body.model_id },
    };
  }
  return { path, body };
}

function sendJson(response, status, body) {
  const payload = JSON.stringify(body);
  response.writeHead(status, { "content-type": "application/json" });
  response.end(payload);
}

function sendError(response, status, message, code) {
  sendJson(response, status, { error: { message, type: code, code } });
}

// Symbols, not strings: a body of `"aborted"` parses to that same string, so a
// string sentinel is indistinguishable from a client payload and a valid one
// silently stopped the router from ever answering.
const BODY_TOO_LARGE = Symbol("body-too-large");
const BODY_ABORTED = Symbol("body-aborted");

async function readBody(request) {
  // Fast path: a declared length over the cap never needs reading.
  if (Number(request.headers["content-length"]) > MAX_BODY_BYTES) {
    request.resume(); // drain: destroying the socket would kill the 413 too
    return BODY_TOO_LARGE;
  }
  const chunks = [];
  let size = 0;
  try {
    for await (const chunk of request) {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        request.resume();
        return BODY_TOO_LARGE;
      }
      chunks.push(chunk);
    }
  } catch {
    // client left mid-body; nothing to answer on a dead socket
    return BODY_ABORTED;
  }
  if (chunks.length === 0) return null;
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    return undefined; // present but not JSON
  }
}

/** Proxies one request to a backend path, streams the response back, and
 * answers the upstream status with the elapsed time for the access log.
 * Never throws: a backend that drops mid-response and a client that leaves
 * mid-response both end here, and either must be an error line, not an
 * exit (Node would otherwise turn the rejection into a process crash). */
async function proxy(response, backend, path, body) {
  const started = Date.now();
  const timeout = AbortSignal.timeout(UPSTREAM_TIMEOUT_MS);
  let upstream;
  try {
    upstream = await fetch(`${backend}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: timeout,
    });
  } catch {
    if (!response.headersSent) {
      sendError(
        response,
        502,
        timeout.aborted ? "The backend timed out." : "The backend is unreachable.",
        "server_error",
      );
    } else response.destroy();
    return { status: 502, ms: Date.now() - started };
  }
  let text;
  try {
    text = await upstream.text();
  } catch {
    if (!response.headersSent) sendError(response, 502, "The backend dropped the response.", "server_error");
    else response.destroy();
    return { status: 502, ms: Date.now() - started };
  }
  try {
    response.writeHead(upstream.status, {
      "content-type": upstream.headers.get("content-type") ?? "application/json",
    });
    response.end(text);
  } catch {
    response.destroy();
  }
  return { status: upstream.status, ms: Date.now() - started };
}

async function handle(request, response) {
  const url = new URL(request.url ?? "/", "http://localhost");
  const path = url.pathname;

  if (request.method === "GET" && (path === "/health" || path === "/healthz")) {
    return sendJson(response, 200, {
      status: "ok",
      models: modelBackend.size,
      uptime_s: Math.floor((Date.now() - startedAt) / 1000),
      stats: summarizeStats(stats),
    });
  }

  if (request.method === "GET" && path === "/v1/models") {
    await ensureFresh();
    return sendJson(response, 200, {
      object: "list",
      // Additive fields: a client that reads only `id` is unaffected. The
      // bound is the strictest replica's, so a client chunks safely without
      // reading a container command. See docs/operations.md,
      // "Client batch bounds".
      data: [...modelBackend.keys()].map((id) => {
        const bound = minClientBatchSize(modelBackend.get(id), backendBatch);
        const entry = { id, object: "model", owned_by: "local" };
        if (bound !== null) entry.max_client_batch_size = bound;
        return entry;
      }),
    });
  }

  if (request.method === "POST") {
    const started = Date.now();
    const body = await readBody(request);
    const model = body?.model ?? body?.model_id;
    // A backend may advertise paths beyond the static set in /info (`paths`,
    // see README.md "Router"); anything else 404s exactly as before — the
    // model is read first only to ask the catalogue, and the 400s below are
    // unchanged.
    // A backend may advertise paths beyond the static set in /info (`paths`,
    // see README.md "Router"). Whether a path exists is a question about the
    // backends, not about the body: asking it per-model meant a request whose
    // body carried no model was answered 404 for the path when the path was
    // served, hiding the real fault.
    if (!POST_PATHS.has(path) && !pathServed(backendPaths, path)) {
      return sendError(response, 404, "Not found.", "invalid_request_error");
    }
    const isStatic = POST_PATHS.has(path);
    const log = (model, backend, status, record = model !== "-") => {
      const ms = Date.now() - started;
      if (record) recordStat(stats, model, backend, status, ms);
      console.log(accessLine({ method: "POST", path, model, backend, status, ms }));
    };
    if (body === BODY_ABORTED) return;
    if (body === BODY_TOO_LARGE) {
      log("-", "-", 413);
      return sendError(response, 413, "Request body too large.", "invalid_request_error");
    }
    if (body === undefined) {
      log("-", "-", 400);
      return sendError(response, 400, "Request body must be JSON.", "invalid_request_error");
    }
    if (body === null) {
      log("-", "-", 400);
      return sendError(response, 400, "Request body is required.", "invalid_request_error");
    }
    if (typeof model !== "string") {
      log("-", "-", 400);
      return sendError(response, 400, "A model is required.", "invalid_request_error");
    }
    const backend = await resolveBackend(model, path, isStatic);
    if (backend === undefined) {
      log(model, "-", 404, false);
      return sendError(response, 404, `No backend serves model '${model}'.`, "model_not_found");
    }
    // Fail fast on an over-bound request instead of letting the backend
    // answer 400 with its own wording. The bound is the one advertised above,
    // so the error teaches the client the same number /v1/models does.
    if (path === "/rerank" || path === "/v1/rerank") {
      const list = rerankDocuments(body);
      if (list !== undefined && !Array.isArray(list)) {
        log(model, backend, 400);
        return sendError(response, 400, "A rerank request carries a document list.", "invalid_request_error");
      }
      const bound = minClientBatchSize(modelBackend.get(model), backendBatch);
      const count = Array.isArray(list) ? list.length : null;
      if (bound !== null && count !== null && count > bound) {
        log(model, backend, 413);
        return sendJson(response, 413, {
          error: {
            message: `This model accepts at most ${bound} documents per request, got ${count}.`,
            type: "invalid_request_error",
            code: "batch_too_large",
            max_client_batch_size: bound,
          },
        });
      }
    }
    const upstream = backendRequest(path, body);
    const { status, ms } = await proxy(response, backend, upstream.path, upstream.body);
    recordStat(stats, model, backend, status, ms);
    console.log(accessLine({ method: "POST", path, model, backend, status, ms }));
    return;
  }

  return sendError(response, 404, "Not found.", "invalid_request_error");
}

const server = createServer((request, response) => {
  handle(request, response).catch((error) => {
    // Last resort: a client that disconnects mid-body lands here (ECONNRESET
    // from readBody) alongside real bugs. Either way the answer is local —
    // never a process exit, which would drop every model at once.
    console.error(`router: unhandled request error: ${String(error?.message ?? error).slice(0, 200)}`);
    try {
      if (!response.headersSent && !response.writableEnded) {
        sendJson(response, 500, {
          error: { message: "Internal error.", type: "server_error", code: "server_error" },
        });
      } else {
        response.destroy();
      }
    } catch {
      response.destroy();
    }
  });
});

await scan();
setInterval(() => {
  scan().catch(() => {});
}, MODEL_TTL_MS).unref();

// A proxy stays up: log stragglers instead of taking Node's default, which
// turns an unhandled rejection into a process exit.
process.on("unhandledRejection", (error) => {
  console.error(`router: unhandled rejection: ${String(error?.message ?? error).slice(0, 200)}`);
});

server.listen(PORT, () => {
  console.log(
    `router on :${PORT} fronting labelled backends` +
      (BACKENDS.length > 0 ? ` plus ${BACKENDS.length} static` : ""),
  );
});
