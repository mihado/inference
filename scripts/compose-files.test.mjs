// A compose file missing from COMPOSE_ALL survives `down-all` while still
// routed to.   node --test scripts/compose-files.test.mjs
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");


function composeFilesOnDisk() {
  const found = [];
  for (const entry of readdirSync(ROOT, { withFileTypes: true })) {
    if (entry.isFile() && /^compose[\w.-]*\.ya?ml$/.test(entry.name)) found.push(entry.name);
    else if (entry.isDirectory()) {
      for (const inner of readdirSync(join(ROOT, entry.name), { withFileTypes: true })) {
        if (inner.isFile() && /^compose[\w.-]*\.ya?ml$/.test(inner.name)) {
          found.push(`${entry.name}/${inner.name}`);
        }
      }
    }
  }
  return found.sort();
}

function composeAllInMakefile() {
  const line = readFileSync(join(ROOT, "Makefile"), "utf8")
    .split("\n")
    .find((row) => row.startsWith("COMPOSE_ALL :="));
  assert.ok(line, "COMPOSE_ALL is not defined in the Makefile");
  return [...line.matchAll(/(?:^|\s)-f\s+(\S+)/g)].map((match) => match[1]).sort();
}

test("COMPOSE_ALL names every compose file on disk", () => {
  const onDisk = composeFilesOnDisk();
  const listed = composeAllInMakefile();
  assert.deepEqual(
    listed.filter((file) => !onDisk.includes(file)),
    [],
    "COMPOSE_ALL lists files that do not exist",
  );
  assert.deepEqual(
    onDisk.filter((file) => !listed.includes(file)),
    [],
    "compose files on disk are missing from COMPOSE_ALL, so up-all and down-all skip them",
  );
});

test("every compose file declares the same project name", () => {
  const names = new Map();
  for (const file of composeFilesOnDisk()) {
    const match = readFileSync(join(ROOT, file), "utf8").match(/^name:\s*(\S+)/m);
    if (match) names.set(file, match[1]);
  }
  const distinct = new Set(names.values());
  assert.equal(distinct.size, 1, `compose files disagree on the project name: ${[...names]}`);
});

test("the client batch bound is declared once per model class", () => {
  // Rerankers are 64 and embedders 128 on purpose; one variable each.
  const rerankerFiles = ["compose.yml", "compose.rerankers.yml"];
  for (const file of rerankerFiles) {
    const source = readFileSync(join(ROOT, file), "utf8");
    for (const [, value] of source.matchAll(/--max-client-batch-size\s*\n\s*-?\s*"?\$\{MAX_CLIENT_BATCH_SIZE:-(\d+)\}"?/g)) {
      assert.equal(value, "64", `${file}: reranker bound default`);
    }
  }
  const compose = readFileSync(join(ROOT, "compose.yml"), "utf8");
  assert.match(compose, /\$\{MAX_CLIENT_BATCH_SIZE:-64\}/, "compose.yml must read MAX_CLIENT_BATCH_SIZE");
  const qwen = readFileSync(join(ROOT, "compose.qwen-embed.yml"), "utf8");
  assert.match(qwen, /\$\{MAX_EMBED_BATCH_SIZE:-128\}/, "compose.qwen-embed.yml must read MAX_EMBED_BATCH_SIZE");
  const run = readFileSync(join(ROOT, "scripts/run.sh"), "utf8");
  assert.match(run, /\$\{MAX_EMBED_BATCH_SIZE:-128\}/, "run.sh must read MAX_EMBED_BATCH_SIZE");
});
