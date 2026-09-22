/** Data-only module: shared by the browser, Node tests, and experiments. */
export const GENRES = Object.freeze([
  "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]);

function positiveInteger(value, label) {
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < 1) {
    throw new Error(`Invalid ${label}: ${value}`);
  }
  return Number(value);
}

function parseLines(text, filename, parseRow) {
  const rows = [];
  for (const [index, raw] of text.replace(/^\uFEFF/, "").split(/\r?\n/).entries()) {
    if (!raw.trim()) continue;
    try {
      rows.push(parseRow(raw));
    } catch (error) {
      throw new Error(`${filename}, line ${index + 1}: ${error.message}`);
    }
  }
  if (!rows.length) throw new Error(`${filename} contains no records.`);
  return rows;
}

export function parseItemData(text) {
  const ids = new Set();
  return parseLines(text, "u.item", (line) => {
    const fields = line.split("|");
    if (fields.length !== 24) throw new Error(`Expected 24 fields, received ${fields.length}.`);
    const id = positiveInteger(fields[0], "movie ID");
    if (ids.has(id)) throw new Error(`Duplicate movie ID ${id}.`);
    ids.add(id);
    const title = fields[1].trim();
    if (!title) throw new Error("Missing movie title.");
    // Five metadata fields, then UNKNOWN, then the 18 useful flags. Validate
    // all 19 flags before deliberately discarding only the unknown column.
    const flags = fields.slice(5);
    if (flags.some((flag) => flag !== "0" && flag !== "1")) {
      throw new Error("Genre flags must be binary (0 or 1).");
    }
    const vector = flags.slice(1).map(Number);
    const genres = GENRES.filter((_, index) => vector[index] === 1);
    return { id, title, genres, vector, ratingCount: 0 };
  });
}

export function parseRatingData(text) {
  const pairs = new Set();
  return parseLines(text, "u.data", (line) => {
    const fields = line.split("\t");
    if (fields.length !== 4) throw new Error(`Expected 4 tab-separated fields, received ${fields.length}.`);
    const [userId, itemId, rating, timestamp] = fields.map((value, index) =>
      positiveInteger(value, ["user ID", "movie ID", "rating", "timestamp"][index]));
    if (rating > 5) throw new Error("Rating must be between 1 and 5.");
    const key = `${userId}:${itemId}`;
    if (pairs.has(key)) throw new Error(`Duplicate user/movie rating ${key}.`);
    pairs.add(key);
    return { userId, itemId, rating, timestamp };
  });
}

/** Popularity counts every rating, regardless of its score. Never used to rank. */
export function buildDataset(itemText, ratingText) {
  const movies = parseItemData(itemText);
  const ratings = parseRatingData(ratingText);
  const byId = new Map(movies.map((movie) => [movie.id, movie]));
  for (const { itemId } of ratings) {
    const movie = byId.get(itemId);
    if (!movie) throw new Error(`u.data references missing movie ID ${itemId}.`);
    movie.ratingCount += 1;
  }
  return { movies, ratings };
}

/** Original MovieLens titles are ISO-8859-1, not UTF-8. Keep bytes intact. */
export function decodeItems(bytes) {
  return new TextDecoder("iso-8859-1").decode(bytes);
}

export async function loadData(baseUrl = new URL("./", import.meta.url)) {
  const fetchFile = async (name) => {
    const response = await fetch(new URL(name, baseUrl), { signal: AbortSignal.timeout(20000) });
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}.`);
    return response;
  };
  const [items, ratings] = await Promise.all([fetchFile("u.item"), fetchFile("u.data")]);
  const [itemBytes, ratingText] = await Promise.all([items.arrayBuffer(), ratings.text()]);
  return buildDataset(decodeItems(itemBytes), ratingText);
}
