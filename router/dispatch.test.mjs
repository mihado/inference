// Dispatch check for the router server: boots index.mjs as a child on an
// ephemeral port, in front of two stub backends, and pins what each public
// path forwards. Static BACKENDS only (DOCKER_SOCK points nowhere).
//   node --test router/dispatch.test.mjs
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { connect } from "node:net";
import { test } from "node:test";

const DIR = new URL(".", import.meta.url).pathname;

/** A stub backend: GET serves the catalogue shape, POST echoes path + body,
 * optionally after a delay (to exercise the upstream timeout). */
function stub(name, catalogue, delayMs = 0) {
  const server = createServer((request, response) => {
    if (request.method === "GET") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(catalogue()));
      return;
    }
    let raw = "";
    request.on("data", (chunk) => (raw += chunk));
    request.on("end", () => {
      setTimeout(
        () => {
          response.writeHead(200, { "content-type": "application/json" });
          response.end(JSON.stringify({ stub: name, path: request.url, body: JSON.parse(raw) }));
        },
        delayMs,
      );
    });
  });
  return { server };
}

/** A backend that dies mid-response: headers go out, the socket does not —
 * the shape of a backend OOM-kill or restart under load. */
function flakyBackend(catalogue) {
  const server = createServer((request, response) => {
    if (request.method === "GET") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(catalogue()));
      return;
    }
    request.resume();
    request.on("end", () => {
      response.writeHead(200, { "content-type": "application/json" });
      response.write('{"partial":');
      response.destroy();
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
let stubC;
let stubD;
let flaky;
let slow;

test.before(async () => {
  stubA = stub("a", () => ({ model_id: "test-rerank" }));
  stubB = stub("b", () => ({ object: "list", data: [{ id: "test-embed" }, { id: "test-rerank" }] }));
  // A backend advertising a path the router never hardcoded, with junk the
  // router must ignore (see README.md "Router").
  stubC = stub("c", () => ({
    model_id: "test-paths",
    paths: ["/v1/novel", "relative", 42, "/v1/novel"],
    max_client_batch_size: 64,
  }));
  // Same model, older paths and a stricter batch: a mid-rollout replica.
  stubD = stub("d", () => ({ model_id: "test-paths", paths: ["/v1/legacy"], max_client_batch_size: 32 }));
  flaky = flakyBackend(() => ({ model_id: "test-flaky" }));
  slow = stub("s", () => ({ model_id: "test-slow" }), 1000);
  const portA = await listen(stubA.server);
  const portB = await listen(stubB.server);
  const portC = await listen(stubC.server);
  const portD = await listen(stubD.server);
  const portF = await listen(flaky.server);
  const portS = await listen(slow.server);

  const routerPort = await freePort();
  child = spawn(process.execPath, ["index.mjs"], {
    cwd: DIR,
    env: {
      ...process.env,
      PORT: String(routerPort),
      BACKENDS: `http://127.0.0.1:${portA},http://127.0.0.1:${portB},http://127.0.0.1:${portC},http://127.0.0.1:${portD},http://127.0.0.1:${portF},http://127.0.0.1:${portS}`,
      DOCKER_SOCK: "/nonexistent-router-test.sock",
      MODEL_TTL_MS: "60000",
      MAX_BODY_BYTES: "1024",
      UPSTREAM_TIMEOUT_MS: "300",
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
  // A crashed child has no future 'exit' event; do not hang the suite on it.
  if (child.exitCode === null) {
    child.kill();
    await new Promise((resolve) => child.on("exit", resolve));
  }
  stubA.server.close();
  stubB.server.close();
  stubC.server.close();
  stubD.server.close();
  flaky.server.close();
  slow.server.close();
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
  assert.deepEqual(ids, ["test-embed", "test-flaky", "test-paths", "test-rerank", "test-slow"]);
});

test("GET /v1/models advertises the strictest replica's client batch bound", async () => {
  const data = (await (await fetch(`${routerBase}/v1/models`)).json()).data;
  const entry = data.find((row) => row.id === "test-paths");
  // c reports 64, d reports 32; a client must respect the stricter one.
  assert.equal(entry.max_client_batch_size, 32);
  // A backend that reports no bound leaves the field off entirely.
  assert.equal("max_client_batch_size" in data.find((row) => row.id === "test-rerank"), false);
});

test("POST /rerank normalizes Cohere documents to TEI texts, keeping model", async () => {
  const { status, body } = await post("/rerank", { model: "test-rerank", query: "q", documents: ["d1"] });
  assert.equal(status, 200);
  assert.equal(body.path, "/rerank");
  assert.deepEqual(body.body, { query: "q", texts: ["d1"], model: "test-rerank" });
});

test("POST /v1/rerank keeps texts and still lands on TEI /rerank", async () => {
  const { status, body } = await post("/v1/rerank", { model: "test-rerank", query: "q", texts: ["d1"] });
  assert.equal(status, 200);
  assert.equal(body.path, "/rerank");
  assert.deepEqual(body.body, { query: "q", texts: ["d1"], model: "test-rerank" });
});

test("POST /v1/decisions forwards untouched", async () => {
  const payload = { model: "test-rerank", state: "s", questions: [{ id: "q1", text: "t" }] };
  const { status, body } = await post("/v1/decisions", payload);
  assert.equal(status, 200);
  assert.equal(body.path, "/v1/decisions");
  assert.deepEqual(body.body, payload);
});

test("POST /api/evaluate forwards untouched", async () => {
  const payload = { model: "test-rerank", state: "s", questions: [{ id: "q1", type: "boolean", question: "done?" }] };
  const { status, body } = await post("/api/evaluate", payload);
  assert.equal(status, 200);
  assert.equal(body.path, "/api/evaluate");
  assert.deepEqual(body.body, payload);
});

test("POST /v1/predict forwards untouched", async () => {
  const payload = {
    model: "test-rerank",
    state: "I was charged twice.",
    questions: { team: { type: "noul", instructions: "t" } },
  };
  const { status, body } = await post("/v1/predict", payload);
  assert.equal(status, 200);
  assert.equal(body.path, "/v1/predict");
  assert.deepEqual(body.body, payload);
});

test("POST /v1/systemone forwards untouched", async () => {
  const payload = {
    model: "test-rerank",
    state: { images: ["data:image/png;base64,AA=="] },
    questions: { q1: { type: "noul", instructions: "t" } },
  };
  const { status, body } = await post("/v1/systemone", payload);
  assert.equal(status, 200);
  assert.equal(body.path, "/v1/systemone");
  assert.deepEqual(body.body, payload);
});

test("POST to an advertised path routes with no router change", async () => {
  const payload = { model: "test-paths", state: "s" };
  const { status, body } = await post("/v1/novel", payload);
  assert.equal(status, 200);
  assert.equal(body.stub, "c");
  assert.equal(body.path, "/v1/novel");
  assert.deepEqual(body.body, payload);
});

test("advertised junk never routes, and an unadvertised path still 404s", async () => {
  assert.equal((await post("/relative", { model: "test-paths" })).status, 404);
  assert.equal((await post("/v1/unknown", { model: "test-paths" })).status, 404);
  assert.equal((await post("/v1/novel", { model: "nope" })).status, 404);
});

test("a novel path never reaches the replica that did not advertise it", async () => {
  const who = new Set();
  for (let i = 0; i < 4; i++) {
    const { status, body } = await post("/v1/novel", { model: "test-paths", state: "s" });
    assert.equal(status, 200);
    who.add(body.stub);
  }
  assert.deepEqual([...who], ["c"]);
});

test("a static path still alternates across both replicas", async () => {
  const who = new Set();
  for (let i = 0; i < 4; i++) {
    const { body } = await post("/rerank", { model: "test-paths", query: "q", texts: ["d"] });
    assert.equal(body.path, "/rerank");
    who.add(body.stub);
  }
  assert.deepEqual([...who].sort(), ["c", "d"]);
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

test("an oversized body answers 413 without touching a backend", async () => {
  const { status } = await post("/rerank", { model: "test-rerank", query: "q", texts: ["d".repeat(2048)] });
  assert.equal(status, 413);
  assert.equal((await fetch(`${routerBase}/health`)).status, 200);
});

test("an upstream slower than the budget becomes a 502, not a hang", async () => {
  const { status } = await post("/rerank", { model: "test-slow", query: "q", texts: ["d"] });
  assert.equal(status, 502);
});

test("a backend that drops mid-response becomes a 502, not an exit", async () => {
  const { status } = await post("/rerank", { model: "test-flaky", query: "q", texts: ["d"] });
  assert.equal(status, 502);
  assert.equal((await fetch(`${routerBase}/health`)).status, 200);
  assert.equal(child.exitCode, null);
});

test("a client that disconnects mid-body does not take the router down", async () => {
  const port = new URL(routerBase).port;
  await new Promise((resolve) => {
    const socket = connect(port, "127.0.0.1", () => {
      socket.write(
        "POST /rerank HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: 10000\r\n\r\n{\"model\":\"x\"",
      );
      socket.destroy();
      socket.on("close", resolve);
    });
  });
  await new Promise((resolve) => setTimeout(resolve, 300));
  assert.equal((await fetch(`${routerBase}/health`)).status, 200);
  assert.equal(child.exitCode, null);
});

test("GET /health answers without a catalogue scan", async () => {
  const response = await fetch(`${routerBase}/health`);
  assert.equal(response.status, 200);
  assert.equal((await response.json()).status, "ok");
});

test("GET /health reports per-model and per-backend counts", async () => {
  const body = await (await fetch(`${routerBase}/health`)).json();
  assert.equal(typeof body.uptime_s, "number");
  const rerank = body.stats["test-rerank"];
  assert.ok(rerank.requests >= 10, `requests=${rerank.requests}`);
  assert.equal(rerank.errors, 0);
  assert.ok(Number.isFinite(rerank.p50_ms));
  assert.equal(Object.keys(rerank.backends).length, 2);
  assert.ok(body.stats["test-flaky"].errors >= 1);
  assert.ok(body.stats["nope"].errors >= 1);
});
