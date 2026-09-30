// Runnable checks for the paired comparison:
//   node --test eval/compare.test.mjs
//
// The reference numbers are hand-computed from the definition, not copied from
// a run, so a change to the statistic fails here rather than silently
// reinterpreting a past decision.
import assert from "node:assert/strict";
import { test } from "node:test";

import { discordant, mcnemar, parseRows, summarize } from "./compare.mjs";

test("recall, MRR and n come from the ranks", () => {
  const rows = [
    { model: "a", id: "1", rank: 1 },
    { model: "a", id: "2", rank: 2 },
    { model: "a", id: "3", rank: 3 },
    { model: "a", id: "4", rank: 1 },
  ];
  const s = summarize(rows).get("a");
  assert.equal(s.n, 4);
  assert.equal(s.hits, 2);
  assert.equal(s.recall1, 0.5);
  assert.equal(s.mrr, (1 + 0.5 + 1 / 3 + 1) / 4);
});

test("a duplicate id is reported, not silently absorbed", () => {
  // The real case file held 370 rows under 365 ids. Collapsing without saying
  // so is how the published denominator went unexplained.
  const rows = [
    { model: "a", id: "dup", rank: 1 },
    { model: "a", id: "dup", rank: 7 },
    { model: "a", id: "other", rank: 1 },
  ];
  const s = summarize(rows).get("a");
  assert.equal(s.n, 2, "distinct cases, not rows");
  assert.deepEqual(s.duplicateIds, ["dup"]);
  assert.equal(s.hits, 2, "the first row's rank is kept, not the last");
  assert.equal(s.mrr, 1);
});

test("mcnemar reproduces a hand-computed exact p", () => {
  // 36 vs 16 of 52 discordant: 2 * P(X<=16 | Bin(52,0.5)) = 0.0078.
  const r = mcnemar(36, 16);
  assert.equal(r.n, 52);
  assert.ok(Math.abs(r.p - 0.0078) < 0.0001, `p=${r.p}`);

  // A symmetric split is the null: p = 1.
  assert.equal(mcnemar(26, 26).p, 1);
  // No disagreement is no evidence, not a pass.
  assert.equal(mcnemar(0, 0).p, null);
  // 4 vs 0 of 4 discordant: 2 * (1/16) = 0.125.
  assert.equal(mcnemar(4, 0).p, 0.125);
});

test("discordant counts only cases where exactly one model ranked first", () => {
  const a = new Map([
    ["1", 1],
    ["2", 2],
    ["3", 1],
    ["4", 3],
  ]);
  const b = new Map([
    ["1", 1], // agreed both first: ignored
    ["2", 1], // b wins
    ["3", 2], // a wins
    ["4", 4], // agreed neither: ignored
    ["5", 1], // only b has it: ignored
  ]);
  assert.deepEqual(discordant(a, b), { aWins: 1, bWins: 1, shared: 4 });
});

test("parseRows skips blanks and rejects nothing silently", () => {
  assert.deepEqual(parseRows('{"model":"a","id":"1","rank":1}\n\n{"model":"a","id":"2","rank":2}\n'), [
    { model: "a", id: "1", rank: 1 },
    { model: "a", id: "2", rank: 2 },
  ]);
  assert.throws(() => parseRows("{not json}"), "a malformed row must not parse to nothing");
});