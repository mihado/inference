// Scoring one reranked case, kept pure so it can be unit tested:
//   node --test eval/rank.test.mjs
//
// The permutation check here is what rejected every well-formed response at 30
// documents before the numeric sort was added, and a suite of six-document
// cases cannot catch that class of bug — so it gets its own test.

/** Rank (1-based) of the relevant doc, tolerant of both response shapes:
 * a bare array (TEI) and `{results: [...]}` (the gateway). Null when the
 * response cannot be scored, which the caller treats as a failure. */
export function relevantRank(body, relPos, count) {
  const results = Array.isArray(body) ? body : body?.results;
  if (!Array.isArray(results) || results.length !== count) return null;
  // Numeric sort. A bare .sort() orders indices as strings, so at 11 or more
  // documents it yields 0,1,10,11,...,2 and every real response looks
  // malformed. Six-document pools cannot catch this.
  const order = [...results.map((entry) => entry?.index)].sort((a, b) => a - b);
  if (order.join(",") !== [...Array(count).keys()].join(",")) return null;
  const scores = results.map((entry) => entry?.relevance_score ?? entry?.score);
  if (!scores.every(Number.isFinite)) return null;
  const at = results.findIndex((entry) => entry?.index === relPos);
  return at === -1 ? null : at + 1;
}
