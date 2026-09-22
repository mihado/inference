// Dispatch check for the router server: boots index.mjs as a child on an
// ephemeral port, in front of two stub backends, and pins what each public
// path forwards. Static BACKENDS only (DOCKER_SOCK points nowhere).
//   node --test router/dispatch.test.mjs
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { test } from "node:test";

const DIR = new URL(".", import.meta.url).pathname;

/** A stub backend: GET serves the catalogue shape, POST echoes path + body. */
function stub(name, catalogue) {
  const server = createServer((request, response) => {
    if (request.method === "GET") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(catalogue()));
      return;
    }
    let raw = "";
    request.on("data", (chunk) => (raw += chunk));
    request.on("end", () => {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ stub: name, path: request.url, body: JSON.parse(raw) }));
    });
  });
  return { server };
}

async function listen(server) {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return server.address().port;
}

/** A port nothing listens on right now — the child takes it immediately. */
async function freePort() {
  const probe = createServer();
  const port = await listen(probe);
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

let routerBase;
let child;
let stubA;
let stubB;

test.before(async () => {
  stubA = stub("a", () => ({ model_id: "test-rerank" }));
  stubB = stub("b", () => ({ object: "list", data: [{ id: "test-embed" }, { id: "test-rerank" }] }));
  const portA = await listen(stubA.server);
  const portB = await listen(stubB.server);

  const routerPort = await freePort();
  child = spawn(process.execPath, ["index.mjs"], {
    cwd: DIR,
    env: {
      ...process.env,
      PORT: String(routerPort),
      BACKENDS: `http://127.0.0.1:${portA},http://127.0.0.1:${portB}`,
      DOCKER_SOCK: "/nonexistent-router-test.sock",
      MODEL_TTL_MS: "60000",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  routerBase = `http://127.0.0.1:${routerPort}`;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("router did not boot")), 15000);
    child.stdout.on("data", (chunk) => {
      if (String(chunk).includes("router on")) {
        clearTimeout(timeout);
        resolve();
      }
    });
    child.on("exit", (code) => reject(new Error(`router exited with ${code}`)));
  });
});

test.after(async () => {
  child.kill();
  await new Promise((resolve) => child.on("exit", resolve));
  stubA.server.close();
  stubB.server.close();
});

async function post(path, body, raw) {
  const response = await fetch(`${routerBase}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: raw ?? JSON.stringify(body),
  });
  return { status: response.status, body: await response.json() };
}

test("GET /v1/models is the union of both catalogue shapes", async () => {
  const response = await fetch(`${routerBase}/v1/models`);
  assert.equal(response.status, 200);
  const ids = (await response.json()).data.map((entry) => entry.id).sort();
  assert.deepEqual(ids, ["test-embed", "test-rerank"]);
});

test("POST /rerank normalizes Cohere documents to TEI texts", async () => {
  const { status, body } = await post("/rerank", { model: "test-rerank", query: "q", documents: ["d1"] });
  assert.equal(status, 200);
  assert.equal(body.path, "/rerank");
  assert.deepEqual(body.body, { query: "q", texts: ["d1"] });
});

test("POST /v1/rerank keeps texts and still lands on TEI /rerank", async () => {
  const { status, body } = await post("/v1/rerank", { model: "test-rerank", query: "q", texts: ["d1"] });
  assert.equal(status, 200);
  assert.equal(body.path, "/rerank");
  assert.deepEqual(body.body, { query: "q", texts: ["d1"] });
});

test("POST /v1/decisions forwards untouched", async () => {
  const payload = { model: "test-rerank", state: "s", questions: [{ id: "q1", text: "t" }] };
  const { status, body } = await post("/v1/decisions", payload);
  assert.equal(status, 200);
  assert.equal(body.path, "/v1/decisions");
  assert.deepEqual(body.body, payload);
});

test("POST /v1/embeddings forwards untouched to the OpenAI-shaped backend", async () => {
  const payload = { model: "test-embed", input: "hi" };
  const { status, body } = await post("/v1/embeddings", payload);
  assert.equal(status, 200);
  assert.equal(body.path, "/v1/embeddings");
  assert.deepEqual(body.body, payload);
});

test("a model on two backends alternates between them", async () => {
  const who = [];
  for (let i = 0; i < 4; i++) {
    const { body } = await post("/rerank", { model: "test-rerank", query: "q", texts: ["d"] });
    who.push(body.stub);
  }
  assert.equal(new Set(who).size, 2);
  for (let i = 1; i < who.length; i++) assert.notEqual(who[i], who[i - 1]);
});

test("bad input answers 400, unknown model 404, unknown path 404", async () => {
  assert.equal((await post("/rerank", null, "")).status, 400);
  const notJson = await fetch(`${routerBase}/rerank`, {
    method: "POST",
    headers: { "content-type": "text/plain" },
    body: "hello",
  });
  assert.equal(notJson.status, 400);
  assert.equal((await post("/rerank", {})).status, 400);
  assert.equal((await post("/rerank", { model: "nope", query: "q", texts: [] })).status, 404);
  assert.equal((await fetch(`${routerBase}/nope`)).status, 404);
});

test("GET /health answers without a catalogue scan", async () => {
  const response = await fetch(`${routerBase}/health`);
  assert.equal(response.status, 200);
  assert.equal((await response.json()).status, "ok");
});
