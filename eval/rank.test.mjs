// Runnable check for the rerank scorer:
//   node --test eval/rank.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { relevantRank } from "./rank.mjs";

/** A well-formed response: one entry per document, in ranked order. */
const ranked = (count, order, relPos) =>
  order.map((index, at) => ({ index, score: count - at })).slice(0, count);

test("scores a TEI-shaped bare array and a gateway-shaped object alike", () => {
  const entries = [
    { index: 2, score: 0.9 },
    { index: 0, score: 0.5 },
    { index: 1, score: 0.1 },
  ];
  assert.equal(relevantRank(entries, 0, 3), 2);
  assert.equal(relevantRank({ results: entries }, 0, 3), 2);
});

test("accepts a full pool at 30 documents (the regression)", () => {
  // The bare .sort() ordering this guards against: at 30 indices the string
  // sort is 0,1,10,11,...,2,20,... and never equals 0,1,2,....
  const entries = [...Array(30).keys()].map((index) => ({ index, score: 1 - index / 100 }));
  assert.equal(relevantRank(entries, 0, 30), 1);
  assert.equal(relevantRank(entries, 29, 30), 30);
  assert.equal([...Array(30).keys()].sort().join(","), entries.map((e) => e.index).sort().join(","));
});

test("rejects a reversed order even though it is a valid permutation of the set", () => {
  const entries = [...Array(30).keys()].reverse().map((index) => ({ index, score: 1 }));
  assert.notEqual(relevantRank(entries, 0, 30), 1);
});

test("rejects a wrong count, a duplicate index, and a missing score", () => {
  const good = [...Array(6).keys()].map((index) => ({ index, score: 1 }));
  assert.equal(relevantRank(good, 0, 7), null, "count mismatch");
  assert.equal(relevantRank([...good.slice(0, 5), { index: 4, score: 1 }], 0, 6), null, "duplicate");
  assert.equal(relevantRank([...good.slice(0, 5), { index: 5 }], 0, 6), null, "no score");
  assert.equal(relevantRank([...good.slice(0, 5), { index: 5, score: null }], 0, 6), null, "null score");
  assert.equal(relevantRank([...good.slice(0, 5), { index: 5, score: "x" }], 0, 6), null, "string score");
  assert.equal(relevantRank(good, 99, 6), null, "relevant index absent");
});

test("rejects a non-array body and a null entry", () => {
  assert.equal(relevantRank(null, 0, 1), null);
  assert.equal(relevantRank({ results: "nope" }, 0, 1), null);
  assert.equal(relevantRank([null], 0, 1), null);
});
