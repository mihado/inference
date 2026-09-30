// Paired comparison of reranker results from eval/run.mjs per-case output:
//
//   EVAL_PERCASE=percase.jsonl node eval/run.mjs
//   node eval/compare.mjs percase.jsonl percase-voyage.jsonl
//   node eval/compare.mjs percase.jsonl --ref Alibaba-NLP/gte-reranker-modernbert-base
//
// Why this exists instead of eyeballing the table eval/run.mjs prints: an
// aggregate recall@1 gap cannot be read. A 20-case difference is the same
// number whether one model won 20 cases outright or the two split 100 ways and
// netted to 20. Only the discordant pairs separate those, and that is exactly
// what an exact McNemar test uses.
//
// Reads {model, id, rank, ms} rows. Holds no corpus content: `id` is whatever
// the case file used, and nothing here reads the cases themselves.
import { readFileSync } from "node:fs";

/** One model's per-case ranks, keyed by case id, plus its own bookkeeping. */
export function summarize(rows) {
  const byModel = new Map();
  const dupes = new Map();
  for (const row of rows) {
    let cases = byModel.get(row.model);
    if (!cases) {
      cases = new Map();
      byModel.set(row.model, cases);
    }
    if (cases.has(row.id)) {
      const list = dupes.get(row.model) ?? [];
      list.push(row.id);
      dupes.set(row.model, list);
      continue; // keep-first: a duplicate is a defect in the case file, not a second vote
    }
    cases.set(row.id, row.rank);
  }
  const out = new Map();
  for (const [model, cases] of byModel) {
    const ranks = [...cases.values()];
    out.set(model, {
      n: ranks.length,
      hits: ranks.filter((r) => r === 1).length,
      recall1: ranks.filter((r) => r === 1).length / ranks.length,
      mrr: ranks.reduce((sum, r) => sum + 1 / r, 0) / ranks.length,
      duplicateIds: dupes.get(model) ?? [],
    });
  }
  return out;
}

/**
 * Exact two-sided McNemar. `aWins` and `bWins` are the discordant counts: cases
 * one model ranked first and the other did not. Under a null of equal models
 * those split Bin(aWins + bWins, 0.5), so the p-value doubles the smaller tail.
 * Returns null when the two models never disagreed, which is no evidence
 * either way rather than a pass.
 */
export function mcnemar(aWins, bWins) {
  const n = aWins + bWins;
  if (n === 0) return { p: null, n: 0 };
  const tail = (k) => {
    let c = 0;
    for (let j = 0; j <= k; j += 1) c += binomial(n, j);
    return c / 2 ** n;
  };
  const k = Math.min(aWins, bWins);
  return { p: Math.min(1, 2 * tail(k)), n };
}

function binomial(n, k) {
  let result = 1;
  for (let i = 0; i < k; i += 1) result = (result * (n - i)) / (i + 1);
  return result;
}

/** Discordant wins for `a` against `b` over the cases both scored. */
export function discordant(casesA, casesB) {
  let aWins = 0;
  let bWins = 0;
  for (const [id, rankA] of casesA) {
    const rankB = casesB.get(id);
    if (rankB === undefined) continue;
    if (rankA === 1 && rankB !== 1) aWins += 1;
    else if (rankB === 1 && rankA !== 1) bWins += 1;
  }
  return { aWins, bWins, shared: [...casesA.keys()].filter((id) => casesB.has(id)).length };
}

export function parseRows(text) {
  return text
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line));
}

const isMain = process.argv[1]?.endsWith("compare.mjs") ?? false;
if (isMain) {
  const argv = process.argv.slice(2);
  const refFlag = argv.indexOf("--ref");
  const ref = refFlag === -1 ? null : argv[refFlag + 1];
  const files = argv.filter((a, i) => !a.startsWith("--") && i !== refFlag + 1);

  if (files.length === 0) {
    console.error("usage: node eval/compare.mjs <percase.jsonl>... [--ref MODEL]");
    process.exit(2);
  }

  const rows = files.flatMap((file) => parseRows(readFileSync(file, "utf8")));
  const stats = summarize(rows);
  const cases = new Map();
  for (const row of rows) {
    if (!cases.has(row.model)) cases.set(row.model, new Map());
    if (!cases.get(row.model).has(row.id)) cases.get(row.model).set(row.id, row.rank);
  }

  const models = [...stats.keys()].sort((a, b) => stats.get(b).recall1 - stats.get(a).recall1);
  const width = Math.max(...models.map((m) => m.length));

  console.log(`n = ${models.map((m) => stats.get(m).n).join("/")} cases per model\n`);
  console.log(`${"model".padEnd(width)}  recall@1  MRR     vs ref`);
  console.log("-".repeat(width + 34));
  for (const model of models) {
    const s = stats.get(model);
    let verdict = "";
    if (ref && ref !== model) {
      const d = discordant(cases.get(ref), cases.get(model));
      const { p } = mcnemar(d.aWins, d.bWins);
      const net = d.aWins - d.bWins;
      if (p === null) verdict = `tie on ${d.shared} shared`;
      else if (net === 0) verdict = `tie (p=${p.toFixed(3)})`;
      else verdict = `${net > 0 ? ref : model} leads by ${Math.abs(net)} (p=${p.toFixed(4)})`;
    }
    console.log(
      `${model.padEnd(width)}  ${s.recall1.toFixed(3)} (${String(s.hits).padStart(3)}/${s.n})` +
        `  ${s.mrr.toFixed(3)}  ${verdict}`,
    );
    if (s.duplicateIds.length > 0) {
      console.log(
        `${" ".repeat(width)}  ^ ${s.duplicateIds.length} duplicate id(s) collapsed: ` +
          `${[...new Set(s.duplicateIds)].slice(0, 3).join(", ")}`,
      );
    }
  }

  if (ref && !stats.has(ref)) {
    console.error(`\nref '${ref}' scored no cases; known: ${models.join(", ")}`);
  }
  console.log(
    "\np is an exact two-sided McNemar over discordant cases. Uncorrected for the\n" +
      "number of comparisons; read p against (0.05 / comparisons) for a family-wise claim.",
  );
}