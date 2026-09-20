// Model-aware reverse proxy for a pool of TEI containers.
//
// One endpoint that fronts N single-model TEI servers: it reports the union of
// their models and dispatches each /v1/embeddings and /rerank call to the
// container serving the requested model. Dependency-free (node:http + fetch).
//
//   BACKENDS="http://hf-1:80,http://hf-2:80,..." node index.mjs
//
// Model -> backend comes from each container's TEI /info (`{ model_id }`) or,
// for OpenAI-shaped servers like vLLM, its /v1/models list; re-scanned on a TTL.
// Traefik cannot dispatch on a JSON body, which is why this exists; put
// TLS/ingress in front of it if you need it.

import { createServer, request as httpRequest } from "node:http";
import { existsSync } from "node:fs";

const PORT = Number(process.env.PORT ?? 80);
const BACKENDS = (process.env.BACKENDS ?? "")
  .split(",")
  .map((entry) => entry.trim().replace(/\/+$/, ""))
  .filter((entry) => entry.length > 0);
const MODEL_TTL_MS = Number(process.env.MODEL_TTL_MS ?? 30_000);
/** Docker socket for label discovery of ad-hoc slots; absent -> static only. */
const DOCKER_SOCK = process.env.DOCKER_SOCK ?? "/var/run/docker.sock";
const BACKEND_LABEL = process.env.BACKEND_LABEL ?? "tei.backend";

/** model id -> backend base url. */
let modelBackend = new Map();
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
 * the shared compose network. Empty when the socket is absent or unreachable. */
async function dockerBackends() {
  if (!existsSync(DOCKER_SOCK)) return [];
  const filters = encodeURIComponent(JSON.stringify({ label: [`${BACKEND_LABEL}=1`] }));
  const containers = await dockerJson(`/containers/json?filters=${filters}`);
  const urls = [];
  for (const container of Array.isArray(containers) ? containers : []) {
    const name = String(container?.Names?.[0] ?? "").replace(/^\//, "");
    if (name.length > 0) urls.push(`http://${name}:80`);
  }
  return urls;
}

async function scan() {
  const found = new Map();
  const discovered = await dockerBackends().catch(() => []);
  const bases = [...new Set([...BACKENDS, ...discovered])];
  await Promise.all(
    bases.map(async (base) => {
      // TEI answers /info with the single model it serves.
      try {
        const response = await fetch(`${base}/info`, { signal: AbortSignal.timeout(5_000) });
        if (response.ok) {
          const body = await response.json();
          if (typeof body?.model_id === "string") {
            found.set(body.model_id, base);
            return;
          }
        }
      } catch {
        // Fall through to the OpenAI list probe.
      }
      // vLLM (and other OpenAI-shaped servers) answer /v1/models instead.
      try {
        const response = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(5_000) });
        if (!response.ok) return;
        const body = await response.json();
        for (const entry of Array.isArray(body?.data) ? body.data : []) {
          if (typeof entry?.id === "string" && entry.id.length > 0) found.set(entry.id, base);
        }
      } catch {
        // A backend being down is not fatal: it just is not routable.
      }
    }),
  );
  modelBackend = found;
  lastScan = Date.now();
}

function ensureFresh() {
  if (Date.now() - lastScan < MODEL_TTL_MS) return Promise.resolve();
  scanning ??= scan().finally(() => {
    scanning = null;
  });
  return scanning;
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

/** Proxies one request to a backend path, streaming the response back. */
async function proxy(response, backend, path, body) {
  let upstream;
  try {
    upstream = await fetch(`${backend}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    return sendError(response, 502, "The backend is unreachable.", "server_error");
  }
  const text = await upstream.text();
  response.writeHead(upstream.status, {
    "content-type": upstream.headers.get("content-type") ?? "application/json",
  });
  response.end(text);
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
      data: [...modelBackend.keys()].map((id) => ({ id, object: "model", owned_by: "tei" })),
    });
  }

  if (request.method === "POST" && (path === "/v1/embeddings" || path === "/rerank" || path === "/v1/rerank")) {
    const body = await readBody(request);
    if (body === undefined) return sendError(response, 400, "Request body must be JSON.", "invalid_request_error");
    if (body === null) return sendError(response, 400, "Request body is required.", "invalid_request_error");
    const model = body.model ?? body.model_id;
    if (typeof model !== "string") {
      return sendError(response, 400, "A model is required.", "invalid_request_error");
    }
    await ensureFresh();
    const backend = modelBackend.get(model);
    if (backend === undefined) {
      return sendError(response, 404, `No backend serves model '${model}'.`, "model_not_found");
    }
    if (path === "/v1/embeddings") return proxy(response, backend, "/v1/embeddings", body);
    // TEI's rerank takes `texts`; accept Cohere's `documents` too.
    const texts = body.texts ?? body.documents;
    return proxy(response, backend, "/rerank", { query: body.query, texts });
  }

  return sendError(response, 404, "Not found.", "invalid_request_error");
});

await scan();
setInterval(() => {
  scan().catch(() => {});
}, MODEL_TTL_MS).unref();

server.listen(PORT, () => {
  console.log(`tei-router on :${PORT} fronting ${BACKENDS.length} static + labelled backends`);
});
