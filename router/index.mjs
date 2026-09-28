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

import { accessLine, addBackend, addPaths, backendName, infoPaths, openAiModelIds, pickBackend, teiModelId } from "./backends.mjs";

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

/** model id -> every backend base url that serves it, in discovery order. A
 * model served by two GPUs has two entries, and the router takes turns. */
let modelBackend = new Map();
/** model id -> POST paths its backends advertise in /info (`paths`), beyond
 * the static set below. A backend that omits `paths` (TEI, vLLM) leaves no
 * entry and routes the static set only. */
let modelPaths = new Map();
/** model id -> how many requests it has been asked for. Kept outside the scan, so
 * a rescan does not reset the rotation. */
const turns = new Map();
let lastScan = 0;
let scanning = null;

/** One Docker API GET over the socket; rejects on any failure. */
function dockerJson(path) {
  return new Promise((resolve, reject) => {
    const req = httpRequest({ socketPath: DOCKER_SOCK, path, method: "GET" }, (res) => {
      let data = "";
      res.on("data", (chunk) => {
        data += chunk;
      });
      res.on("end", () => {
        if (res.statusCode !== 200) return reject(new Error(`docker ${res.statusCode}`));
        try {
          resolve(JSON.parse(data));
        } catch (error) {
          reject(error);
        }
      });
    });
    req.on("error", reject);
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
  const discovered = await dockerBackends().catch(() => []);
  const bases = [...new Set([...BACKENDS, ...discovered])];
  await Promise.all(
    bases.map(async (base) => {
      for (const { id, paths: advertised } of await servedModels(base)) {
        addBackend(found, id, base);
        addPaths(paths, id, advertised);
      }
    }),
  );
  modelBackend = found;
  modelPaths = paths;
  lastScan = Date.now();
}

/** Every model one backend serves, with the POST paths it advertises: TEI
 * names its one model in /info, anything else is asked for an OpenAI-style
 * list (which cannot advertise paths). */
async function servedModels(base) {
  const info = await fetchJson(`${base}/info`);
  const teiId = teiModelId(info);
  if (teiId !== null) return [{ id: teiId, paths: infoPaths(info) }];
  return openAiModelIds(await fetchJson(`${base}/v1/models`)).map((id) => ({ id, paths: [] }));
}

function ensureFresh() {
  if (Date.now() - lastScan < MODEL_TTL_MS) return Promise.resolve();
  scanning ??= scan().finally(() => {
    scanning = null;
  });
  return scanning;
}

/** The next backend in a model's rotation, after a fresh catalogue. */
async function resolveBackend(model) {
  await ensureFresh();
  const turn = turns.get(model) ?? 0;
  turns.set(model, turn + 1);
  return pickBackend(modelBackend.get(model), turn);
}

/** The static POST paths that carry a model in their JSON body. Backends may
 * advertise more in /info (`paths`, see README.md "Router"); the union is
 * what routes. */
const POST_PATHS = new Set(["/v1/embeddings", "/rerank", "/v1/rerank", "/v1/decisions", "/v1/systemone", "/v1/predict", "/api/evaluate"]);

/** Maps a public path to the backend path and body: rerank accepts Cohere's
 * `documents` for TEI's `texts`; everything else forwards untouched. */
function backendRequest(path, body) {
  if (path === "/rerank" || path === "/v1/rerank") {
    return { path: "/rerank", body: { query: body.query, texts: body.texts ?? body.documents } };
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

async function readBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  if (chunks.length === 0) return null;
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    return undefined; // present but not JSON
  }
}

/** Proxies one request to a backend path, streams the response back, and
 * answers the upstream status with the elapsed time for the access log. */
async function proxy(response, backend, path, body) {
  const started = Date.now();
  let upstream;
  try {
    upstream = await fetch(`${backend}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    sendError(response, 502, "The backend is unreachable.", "server_error");
    return { status: 502, ms: Date.now() - started };
  }
  const text = await upstream.text();
  response.writeHead(upstream.status, {
    "content-type": upstream.headers.get("content-type") ?? "application/json",
  });
  response.end(text);
  return { status: upstream.status, ms: Date.now() - started };
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? "/", "http://localhost");
  const path = url.pathname;

  if (request.method === "GET" && (path === "/health" || path === "/healthz")) {
    return sendJson(response, 200, { status: "ok", models: modelBackend.size });
  }

  if (request.method === "GET" && path === "/v1/models") {
    await ensureFresh();
    return sendJson(response, 200, {
      object: "list",
      data: [...modelBackend.keys()].map((id) => ({ id, object: "model", owned_by: "local" })),
    });
  }

  if (request.method === "POST") {
    const started = Date.now();
    const body = await readBody(request);
    const model = body?.model ?? body?.model_id;
    // A backend may advertise paths beyond the static set in /info (`paths`);
    // anything else 404s exactly as before — the model is read first only to
    // ask the catalogue, and the 400s below are unchanged.
    const advertised = typeof model === "string" && modelPaths.get(model)?.has(path) === true;
    if (!POST_PATHS.has(path) && !advertised) {
      return sendError(response, 404, "Not found.", "invalid_request_error");
    }
    const log = (model, backend, status) =>
      console.log(accessLine({ method: "POST", path, model, backend, status, ms: Date.now() - started }));
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
    const backend = await resolveBackend(model);
    if (backend === undefined) {
      log(model, "-", 404);
      return sendError(response, 404, `No backend serves model '${model}'.`, "model_not_found");
    }
    const upstream = backendRequest(path, body);
    const { status, ms } = await proxy(response, backend, upstream.path, upstream.body);
    console.log(accessLine({ method: "POST", path, model, backend, status, ms }));
    return;
  }

  return sendError(response, 404, "Not found.", "invalid_request_error");
});

await scan();
setInterval(() => {
  scan().catch(() => {});
}, MODEL_TTL_MS).unref();

server.listen(PORT, () => {
  console.log(
    `router on :${PORT} fronting labelled backends` +
      (BACKENDS.length > 0 ? ` plus ${BACKENDS.length} static` : ""),
  );
});
