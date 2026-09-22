/** Reproducible descriptive experiment; no fitting, popularity ranking, or API. */
import assert from "node:assert/strict";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { buildDataset, decodeItems } from "../data.js";
import { recommend, cosineSimilarity, dotProduct } from "../script.js";

const SEED = 20260922;
const SAMPLE_SIZE = 200;
const K = 5;
const root = new URL("../", import.meta.url);
const [itemBytes, ratingBytes] = await Promise.all([readFile(new URL("u.item", root)), readFile(new URL("u.data", root))]);
const { movies, ratings } = buildDataset(decodeItems(itemBytes), ratingBytes.toString("utf8"));
const byId = new Map(movies.map((movie) => [movie.id, movie]));
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const round = (value) => Number(value.toFixed(10));
const headSize = Math.ceil(movies.length * .2);
const popularityOrder = [...movies].sort((a, b) => b.ratingCount - a.ratingCount || a.id - b.id);
const head = new Set(popularityOrder.slice(0, headSize).map((movie) => movie.id));
const tailSize = movies.length - headSize;

// Mulberry32 + Fisher-Yates; start from numeric user ID order, never row order.
function randomGenerator(seed) {
  return () => {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function sampleUsers(users) {
  const shuffled = [...users];
  const random = randomGenerator(SEED);
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
  }
  return shuffled.slice(0, SAMPLE_SIZE);
}

const likedByUser = new Map();
for (const rating of ratings) {
  if (rating.rating < 4) continue;
  if (!likedByUser.has(rating.userId)) likedByUser.set(rating.userId, []);
  likedByUser.get(rating.userId).push(rating);
}
for (const liked of likedByUser.values()) liked.sort((a, b) => a.timestamp - b.timestamp || a.itemId - b.itemId);
const eligible = [...likedByUser.keys()].filter((id) => likedByUser.get(id).length >= 3).sort((a, b) => a - b);
const sampledIds = sampleUsers(eligible);

function overlap(left, right) {
  const shared = left.filter((movie) => right.some((other) => movie.id === other.id)).length;
  const union = left.length + right.length - shared;
  return { shared, fractionAt5: shared / K, jaccard: union ? shared / union : 0 };
}
const compact = (lists) => Object.fromEntries(Object.entries(lists).map(([mode, list]) =>
  [mode, list.map((movie) => ({ id: movie.id, score: round(movie.score) }))]));

function evaluateHistory(history, activeId) {
  const results = {};
  for (const scoring of ["cosine", "dot"]) {
    const lists = recommend(movies, history, activeId, { scoring, limit: K });
    for (const list of Object.values(lists)) {
      assert.ok(list.length <= K && list.every((movie) => !history.includes(movie.id)));
      assert.equal(new Set(list.map((movie) => movie.id)).size, list.length);
    }
    results[scoring] = { lists: compact(lists), overlap: overlap(lists.item, lists.profile) };
  }
  return results;
}

const examples = [
  { name: "One favorite (sanity check)", history: [1], activeId: 1 },
  { name: "Space-adventure trilogy", history: [50, 172, 181], activeId: 181 },
  { name: "Family animation", history: [1, 95, 422], activeId: 422 },
  { name: "Mixed tastes (UI example)", history: [1, 50, 100], activeId: 100 },
  { name: "Romance and drama", history: [213, 275, 313], activeId: 313 },
  { name: "Missing genres (edge case)", history: [267], activeId: 267 },
].map((example) => ({ ...example, results: evaluateHistory(example.history, example.activeId) }));

const users = sampledIds.map((userId) => {
  const liked = likedByUser.get(userId).slice(0, 3);
  const history = liked.map((rating) => rating.itemId);
  const results = evaluateHistory(history, history[2]);
  const activeSwitch = {};
  const historyGrowth = {};
  for (const scoring of ["cosine", "dot"]) {
    // Hold the history/candidates constant while changing only the active movie.
    const switched = recommend(movies, history, history[1], { scoring });
    // An additional descriptive measure of persistence when a third like is added.
    const shorter = recommend(movies, history.slice(0, 2), history[1], { scoring });
    activeSwitch[scoring] = {};
    historyGrowth[scoring] = {};
    for (const mode of ["item", "profile"]) {
      activeSwitch[scoring][mode] = overlap(results[scoring].lists[mode], switched[mode]).fractionAt5;
      historyGrowth[scoring][mode] = overlap(results[scoring].lists[mode], shorter[mode]).fractionAt5;
    }
    assert.equal(activeSwitch[scoring].profile, 1);
  }
  return { userId, history, activeId: history[2], selectedRatings: liked, results, activeSwitch, historyGrowth };
});

function aggregate(scoring, mode) {
  const exposures = users.flatMap((user) => user.results[scoring].lists[mode]);
  const counts = new Map();
  for (const { id } of exposures) counts.set(id, (counts.get(id) ?? 0) + 1);
  const exposureOrder = [...counts].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const tailExposures = exposures.filter((movie) => !head.has(movie.id)).length;
  const uniqueTail = [...counts.keys()].filter((id) => !head.has(id)).length;
  return {
    recommendationSlots: exposures.length,
    tailExposures,
    tailShare: tailExposures / exposures.length,
    averagePopularity: mean(exposures.map(({ id }) => byId.get(id).ratingCount)),
    averageGenreCount: mean(exposures.map(({ id }) => byId.get(id).genres.length)),
    uniqueMovies: counts.size,
    catalogCoverage: counts.size / movies.length,
    uniqueTailMovies: uniqueTail,
    tailCatalogCoverage: uniqueTail / tailSize,
    top10ExposureShare: exposureOrder.slice(0, 10).reduce((sum, [, count]) => sum + count, 0) / exposures.length,
    activeSwitchOverlapAt5: mean(users.map((user) => user.activeSwitch[scoring][mode])),
    historyGrowthOverlapAt5: mean(users.map((user) => user.historyGrowth[scoring][mode])),
    mostExposed: exposureOrder.slice(0, 10).map(([id, count]) => ({ id, title: byId.get(id).title, count })),
  };
}

const summary = {};
for (const scoring of ["cosine", "dot"]) {
  summary[scoring] = {
    item: aggregate(scoring, "item"),
    profile: aggregate(scoring, "profile"),
    betweenModes: {
      meanShared: mean(users.map((user) => user.results[scoring].overlap.shared)),
      meanOverlapAt5: mean(users.map((user) => user.results[scoring].overlap.fractionAt5)),
      meanJaccard: mean(users.map((user) => user.results[scoring].overlap.jaccard)),
      identicalLists: users.filter((user) => user.results[scoring].overlap.shared === K).length,
      disjointLists: users.filter((user) => user.results[scoring].overlap.shared === 0).length,
    },
  };
}
const normalizationOverlap = Object.fromEntries(["item", "profile"].map((mode) => [mode,
  mean(users.map((user) => overlap(user.results.cosine.lists[mode], user.results.dot.lists[mode]).fractionAt5))]));
const numericalQuery = [1, 1, 0, 0];
const numericalExample = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]].map((vector) => ({
  query: numericalQuery, candidate: vector, dot: dotProduct(numericalQuery, vector), cosine: cosineSimilarity(numericalQuery, vector),
}));

const output = {
  protocol: {
    seed: SEED, sampleSize: users.length, eligibleUsers: eligible.length, k: K,
    userSampling: "Numeric ID sort; Mulberry32-seeded Fisher-Yates; first 200 without replacement.",
    history: "First three chronological ratings >= 4; timestamp ties by movie ID; active = third.",
    candidates: "All catalog IDs except the three selected; positive matches only; no other history exclusions.",
    ranking: "Descending score rounded to 12 decimals, ascending ID; popularity never used.",
    head: "ceil(20% * catalog size); descending full-dataset rating count, ascending ID for ties.",
    caveat: "Descriptive exposure study, not held-out accuracy, temporal evaluation, or retention measurement.",
  },
  dataset: {
    movies: movies.length, ratings: ratings.length, users: new Set(ratings.map((rating) => rating.userId)).size,
    sha256: { "u.item": sha256(itemBytes), "u.data": sha256(ratingBytes) },
    zeroVectorIds: movies.filter((movie) => movie.genres.length === 0).map((movie) => movie.id),
    headSize, tailSize, headIds: [...head],
    lastHead: { id: popularityOrder[headSize - 1].id, ratingCount: popularityOrder[headSize - 1].ratingCount },
    firstTail: { id: popularityOrder[headSize].id, ratingCount: popularityOrder[headSize].ratingCount },
    meanGenreCount: mean(movies.map((movie) => movie.genres.length)),
    meanPopularity: mean(movies.map((movie) => movie.ratingCount)),
  },
  summary, normalizationOverlap, numericalExample, sampledUserIds: sampledIds, examples, users,
};

const pct = (value) => `${(value * 100).toFixed(2)}%`;
const number = (value) => value.toFixed(2);
const md = (text) => text.replaceAll("|", "\\|");
const lines = [
  "# Computed comparison results", "", "Generated by `npm run experiment`. Do not edit by hand. See `../ANALYSIS.md` for interpretation.", "",
  `Sample: ${users.length} of ${eligible.length} eligible users; seed ${SEED}; 3 likes (rating ≥ 4); K = 5.`, "",
  `Head: ${headSize} movies; tail: ${tailSize}. Last head movie: ID ${output.dataset.lastHead.id} (${output.dataset.lastHead.ratingCount} ratings); first tail: ID ${output.dataset.firstTail.id} (${output.dataset.firstTail.ratingCount} ratings).`, "",
  "## Exposure and normalization", "",
  "| Scoring | Mode | Tail share | Avg. ratings | Unique / catalog coverage | Tail coverage | Avg. genres | Top-10 exposure share |",
  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
];
for (const scoring of ["cosine", "dot"]) {
  for (const mode of ["item", "profile"]) {
    const value = summary[scoring][mode];
    lines.push(`| ${scoring} | ${mode} | ${pct(value.tailShare)} | ${number(value.averagePopularity)} | ${value.uniqueMovies} / ${pct(value.catalogCoverage)} | ${pct(value.tailCatalogCoverage)} | ${number(value.averageGenreCount)} | ${pct(value.top10ExposureShare)} |`);
  }
}
lines.push("", "## Overlap between modes", "", "| Scoring | Mean shared / 5 | Overlap@5 | Jaccard | Identical sets | Disjoint sets |", "| --- | ---: | ---: | ---: | ---: | ---: |");
for (const scoring of ["cosine", "dot"]) {
  const value = summary[scoring].betweenModes;
  lines.push(`| ${scoring} | ${number(value.meanShared)} | ${pct(value.meanOverlapAt5)} | ${number(value.meanJaccard)} | ${value.identicalLists} | ${value.disjointLists} |`);
}
lines.push("", "## Persistence", "", "Each percentage is mean shared IDs / 5, not a ranking-distance metric.", "", "| Scoring | Mode | Change active: third → second | Grow history: two → three likes |", "| --- | --- | ---: | ---: |");
for (const scoring of ["cosine", "dot"]) {
  for (const mode of ["item", "profile"]) {
    const value = summary[scoring][mode];
    lines.push(`| ${scoring} | ${mode} | ${pct(value.activeSwitchOverlapAt5)} | ${pct(value.historyGrowthOverlapAt5)} |`);
  }
}
lines.push("", `Cosine-vs-dot mean overlap@5: item ${pct(normalizationOverlap.item)}, profile ${pct(normalizationOverlap.profile)}.`, "", "## Numerical normalization example", "", "Query: [1, 1, 0, 0].", "", "| Candidate | Dot | Cosine |", "| --- | ---: | ---: |");
for (const example of numericalExample) lines.push(`| [${example.candidate.join(", ")}] | ${example.dot} | ${example.cosine.toFixed(6)} |`);

lines.push("", "## Example histories", "");
for (const example of examples) {
  lines.push(`### ${example.name}`, "", `History: ${example.history.map((id) => `${byId.get(id).title} (#${id})`).join("; ")}. Active: #${example.activeId}.`, "");
  for (const scoring of ["cosine", "dot"]) {
    const value = example.results[scoring];
    lines.push(`#### ${scoring} — ${value.overlap.shared} shared recommendations`, "", "| Rank | Item-to-item: title; genres; score; ratings | Profile-based: title; genres; score; ratings |", "| ---: | --- | --- |");
    const describe = (recommendation) => {
      if (!recommendation) return "—";
      const movie = byId.get(recommendation.id);
      return `${md(movie.title)} (#${movie.id}); ${movie.genres.join(", ")}; **${recommendation.score.toFixed(4)}**; ${movie.ratingCount}`;
    };
    for (let index = 0; index < Math.max(value.lists.item.length, value.lists.profile.length); index += 1) {
      lines.push(`| ${index + 1} | ${describe(value.lists.item[index])} | ${describe(value.lists.profile[index])} |`);
    }
    if (!value.lists.item.length && !value.lists.profile.length) lines.push("| — | No positive matches | No positive matches |");
    lines.push("");
  }
}
lines.push("## Most exposed under cosine", "");
for (const mode of ["item", "profile"]) {
  lines.push(`### ${mode}`, "", "| Movie | Recommendation slots |", "| --- | ---: |");
  for (const movie of summary.cosine[mode].mostExposed) lines.push(`| ${md(movie.title)} (#${movie.id}) | ${movie.count} |`);
  lines.push("");
}
lines.push("## Input checksums (SHA-256)", "", ...Object.entries(output.dataset.sha256).map(([file, hash]) => `- \`${file}\`: \`${hash}\``), "");
const results = new URL("results/", root);
await mkdir(results, { recursive: true });
await writeFile(new URL("comparison.json", results), `${JSON.stringify(output, null, 2)}\n`);
await writeFile(new URL("comparison.md", results), `${lines.join("\n")}\n`);
console.log(JSON.stringify({ dataset: { movies: movies.length, ratings: ratings.length, eligible: eligible.length, sampled: users.length }, summary, normalizationOverlap }, null, 2));
console.log("Saved results/comparison.json and results/comparison.md");
