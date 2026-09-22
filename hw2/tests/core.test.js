import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { GENRES, parseItemData, parseRatingData, buildDataset, decodeItems } from "../data.js";
import { dotProduct, cosineSimilarity, averageProfile, rankMovies, recommend } from "../script.js";

const flags = (indices) => Array.from({ length: 18 }, (_, i) => Number(indices.includes(i)));
const row = (id, vector = flags([]), unknown = 0, title = `Movie ${id}`) =>
  [id, title, "01-Jan-1995", "", "https://example.test", unknown, ...vector].join("|");
const movie = (id, indices, ratingCount = 0) => ({ id, title: `Movie ${id}`, vector: flags(indices), ratingCount });
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

test("parser maps each of the 18 flags, skipping unknown without shifting indices", () => {
  const expected = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"];
  const parsed = parseItemData(expected.map((_, index) => row(index + 1, flags([index]), 1)).join("\r\n") + "\r\n\r\n");
  assert.deepEqual(GENRES, expected);
  parsed.forEach((entry, index) => {
    assert.deepEqual(entry.genres, [expected[index]]);
    assert.deepEqual(entry.vector, flags([index]));
  });
});

test("unknown-only and completely unlabeled movies remain valid zero vectors", () => {
  const parsed = parseItemData(`\uFEFF${row(1, flags([]), 1)}\n${row(2)}`);
  parsed.forEach((entry) => { assert.deepEqual(entry.genres, []); assert.deepEqual(entry.vector, flags([])); });
});

test("item parser rejects malformed, shifted, duplicate, and nonbinary data with line numbers", () => {
  assert.throws(() => parseItemData(""), /no records/);
  assert.throws(() => parseItemData(row(1).split("|").slice(0, -1).join("|")), /line 1: Expected 24/);
  assert.throws(() => parseItemData(`${row(1)}\n${row(1)}`), /line 2: Duplicate/);
  assert.throws(() => parseItemData(row(1, flags([]), 2)), /binary/);
  assert.throws(() => parseItemData(row(1, [2, ...flags([]).slice(1)])), /binary/);
  assert.throws(() => parseItemData(row("1x")), /Invalid movie ID/);
  assert.throws(() => parseItemData(row(1, flags([]), 0, "")), /Missing movie title/);
});

test("Latin-1 decoder preserves accented titles", () => {
  assert.equal(decodeItems(Uint8Array.from([67, 97, 102, 233])), "Café");
});

test("rating parser validates fields and counts all ratings, not just liked ones", () => {
  const ratings = "1\t1\t5\t100\r\n2\t1\t1\t101\n1\t2\t4\t102\n";
  assert.deepEqual(parseRatingData(ratings)[0], { userId: 1, itemId: 1, rating: 5, timestamp: 100 });
  const dataset = buildDataset([row(1), row(2), row(3)].join("\n"), ratings);
  assert.deepEqual(dataset.movies.map((entry) => entry.ratingCount), [2, 1, 0]);
  assert.throws(() => parseRatingData("1\t1\t6\t100"), /between 1 and 5/);
  assert.throws(() => parseRatingData("1\t1\t0\t100"), /Invalid rating/);
  assert.throws(() => parseRatingData("1\t1\t5"), /4 tab-separated/);
  assert.throws(() => parseRatingData("1\t1\t5\t100\n1\t1\t4\t101"), /Duplicate user\/movie/);
  assert.throws(() => buildDataset(row(1), "1\t2\t4\t100"), /missing movie ID 2/);
});

test("cosine handles identity, orthogonality, partial overlap, and scale invariance", () => {
  close(cosineSimilarity([1, 1, 0], [1, 1, 0]), 1);
  close(cosineSimilarity([1, 0], [0, 1]), 0);
  close(cosineSimilarity([1, 1, 0], [0, 1, 1]), .5);
  close(cosineSimilarity([1, 2, 0], [0, 2, 2]), cosineSimilarity([3, 6, 0], [0, 2, 2]));
  assert.equal(dotProduct([1, 1, 0], [1, 1, 1]), 2);
  assert.throws(() => cosineSimilarity([1], [1, 0]), /dimensions/);
});

test("zero-length vectors never produce NaN, infinity, or arbitrary recommendations", () => {
  assert.equal(cosineSimilarity([], []), 0);
  assert.equal(cosineSimilarity([0, 0], [1, 0]), 0);
  assert.equal(cosineSimilarity([0, 0], [0, 0]), 0);
  assert.deepEqual(rankMovies([movie(1, [0])], flags([])), []);
});

test("profile averages raw vectors rather than normalizing each liked movie first", () => {
  const history = [movie(1, [0]), movie(2, [0, 1, 2]), movie(3, [1])];
  const profile = averageProfile(history);
  assert.deepEqual(profile.slice(0, 4), [2 / 3, 2 / 3, 1 / 3, 0]);
  assert.deepEqual(history[1].vector, flags([0, 1, 2]));
  assert.deepEqual(averageProfile([]), flags([]));
  assert.deepEqual(averageProfile([movie(1, [0]), movie(2, [])]), [0.5, ...Array(17).fill(0)]);
  assert.throws(() => averageProfile([{ vector: [1] }]), /dimensions/);
});

test("ranking excludes every selected ID, uses ID ties, ignores popularity, and caps at five", () => {
  const catalog = [movie(8, [0], 10000), movie(2, [0]), movie(6, [0]), movie(4, [0]), movie(7, [0]), movie(3, [0]), movie(1, [0]), movie(5, [0]), movie(9, [])];
  const originalOrder = catalog.map((entry) => entry.id);
  const rank = (entries) => rankMovies(entries, flags([0]), [1, 3]).map((entry) => entry.id);
  assert.deepEqual(rank(catalog), [2, 4, 5, 6, 7]);
  assert.deepEqual(rank([...catalog].reverse().map((entry) => ({ ...entry, ratingCount: 99999 - entry.ratingCount }))), [2, 4, 5, 6, 7]);
  assert.deepEqual(catalog.map((entry) => entry.id), originalOrder);
  assert.deepEqual(rankMovies(catalog, flags([0]), [], { limit: 0 }), []);
  assert.throws(() => rankMovies(catalog, flags([0]), [], { scoring: "popular" }), /Unknown scoring/);
  assert.throws(() => rankMovies(catalog, flags([0]), [], { limit: -1 }), /nonnegative/);
});

test("score quantization makes mathematically equal floating-point ties deterministic", () => {
  const catalog = [{ id: 20, vector: [1, 1] }, { id: 10, vector: [3, 3] }];
  assert.deepEqual(rankMovies(catalog, [1, 0]).map((entry) => entry.id), [10, 20]);
});

test("cosine penalizes extra nonmatching labels that raw dot product does not", () => {
  const catalog = [movie(1, [0, 1, 2, 3]), movie(2, [0, 1]), movie(3, [0])];
  const query = flags([0, 1]);
  assert.deepEqual(rankMovies(catalog, query, [], { scoring: "dot" }).map((entry) => entry.id), [1, 2, 3]);
  assert.deepEqual(rankMovies(catalog, query).map((entry) => entry.id), [2, 1, 3]);
  close(cosineSimilarity(query, catalog[0].vector), 1 / Math.sqrt(2));
});

test("shared recommender handles empty, duplicate, missing, and changing selections", () => {
  const catalog = [movie(1, [0]), movie(2, [1]), movie(3, [0]), movie(4, [1]), movie(5, [0, 1])];
  assert.deepEqual(recommend(catalog, [], null), { item: [], profile: [] });
  const single = recommend(catalog, [1], 1);
  assert.deepEqual(single.item, single.profile);
  assert.deepEqual(recommend(catalog, [1, 1], 1), single);
  const first = recommend(catalog, [1, 2], 1);
  const second = recommend(catalog, [1, 2], 2);
  assert.deepEqual(first.profile, second.profile);
  assert.notDeepEqual(first.item, second.item);
  for (const list of Object.values(first)) assert.ok(list.every((entry) => ![1, 2].includes(entry.id)));
  assert.deepEqual(recommend(catalog, [1], 2).item, []);
  assert.deepEqual(recommend(catalog, [999], 999), { item: [], profile: [] });
});

test("bundled MovieLens integration: counts, known genres, accents, and zero vectors", async () => {
  const [items, ratings] = await Promise.all([
    readFile(new URL("../u.item", import.meta.url)), readFile(new URL("../u.data", import.meta.url), "utf8"),
  ]);
  const dataset = buildDataset(decodeItems(items), ratings);
  assert.equal(dataset.movies.length, 1682);
  assert.equal(dataset.ratings.length, 100000);
  assert.equal(new Set(dataset.ratings.map((entry) => entry.userId)).size, 943);
  const byId = new Map(dataset.movies.map((entry) => [entry.id, entry]));
  assert.deepEqual(byId.get(1).genres, ["Animation", "Children's", "Comedy"]);
  assert.deepEqual(byId.get(50).genres, ["Action", "Adventure", "Romance", "Sci-Fi", "War"]);
  assert.deepEqual(byId.get(267).genres, []);
  assert.equal(byId.get(50).ratingCount, 583);
  assert.equal(byId.get(543).title, "Misérables, Les (1995)");
  assert.equal(dataset.movies.reduce((sum, entry) => sum + entry.ratingCount, 0), 100000);
  const result = recommend(dataset.movies, [1, 50, 100], 100);
  for (const list of Object.values(result)) {
    assert.equal(list.length, 5);
    assert.ok(list.every((entry) => entry.score > 0 && entry.score <= 1 && ![1, 50, 100].includes(entry.id)));
  }
});
