import { GENRES, loadData } from "./data.js";

// Pure scoring functions are exported from this file so experiments cannot
// silently drift from the service. Importing in Node has no DOM side effects.
export function dotProduct(left, right) {
  if (left.length !== right.length) throw new Error("Vector dimensions must match.");
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

export function cosineSimilarity(left, right) {
  const dot = dotProduct(left, right);
  const lengthProduct = Math.hypot(...left) * Math.hypot(...right);
  if (lengthProduct === 0) return 0;
  // Floating-point arithmetic can otherwise return 1.0000000000000002.
  return Math.max(-1, Math.min(1, dot / lengthProduct));
}

export function averageProfile(movies, dimensions = GENRES.length) {
  const average = Array(dimensions).fill(0);
  if (!movies.length) return average;
  for (const movie of movies) {
    if (movie.vector.length !== dimensions) throw new Error("Vector dimensions must match.");
    movie.vector.forEach((value, index) => { average[index] += value; });
  }
  // Average ORIGINAL binary vectors, not individually unit-normalized ones.
  return average.map((value) => value / movies.length);
}

const scoreKey = (score) => Math.round(score * 1e12);

/** Both modes use this exact candidate filter and total ordering. */
export function rankMovies(movies, query, selectedIds = [], { scoring = "cosine", limit = 5 } = {}) {
  if (!["cosine", "dot"].includes(scoring)) throw new Error(`Unknown scoring method: ${scoring}`);
  if (!Number.isInteger(limit) || limit < 0) throw new Error("Limit must be a nonnegative integer.");
  if (Math.hypot(...query) === 0 || limit === 0) return [];
  const exclude = new Set(selectedIds);
  const score = scoring === "cosine" ? cosineSimilarity : dotProduct;
  return movies
    .filter((movie) => !exclude.has(movie.id))
    .map((movie) => ({ ...movie, score: score(query, movie.vector) }))
    .filter((movie) => scoreKey(movie.score) > 0)
    .sort((a, b) => scoreKey(b.score) - scoreKey(a.score) || a.id - b.id)
    .slice(0, limit);
}

export function recommend(movies, selectedIds, activeId, options = {}) {
  const ids = new Set(selectedIds);
  const selected = movies.filter((movie) => ids.has(movie.id));
  const active = selected.find((movie) => movie.id === activeId);
  const profile = averageProfile(selected);
  return {
    item: active ? rankMovies(movies, active.vector, ids, options) : [],
    profile: rankMovies(movies, profile, ids, options),
  };
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export async function initApp() {
  const byId = (id) => document.getElementById(id);
  const state = { movies: [], byId: new Map(), selected: [], activeId: null };
  const search = byId("movie-search");
  const select = byId("movie-select");
  const add = byId("add-movie");
  const formatter = new Intl.NumberFormat("en-US");

  function updateOptions() {
    const previous = select.value;
    const query = search.value.trim().toLocaleLowerCase("en-US");
    const available = state.movies.filter((movie) =>
      !state.selected.includes(movie.id) && movie.title.toLocaleLowerCase("en-US").includes(query));
    const options = available.map((movie) => new Option(movie.title, String(movie.id)));
    select.replaceChildren(new Option(available.length ? "Select a title…" : "No matching titles", ""), ...options);
    if (available.some((movie) => String(movie.id) === previous)) select.value = previous;
    // A sole filtered match is safe to preselect, making Enter-to-add convenient.
    else if (available.length === 1) select.value = String(available[0].id);
    select.disabled = !available.length;
    add.disabled = !select.value;
    byId("search-count").textContent = `${formatter.format(available.length)} available ${available.length === 1 ? "title" : "titles"}${query ? " matching your search" : ""}. Selected movies are hidden.`;
  }

  function renderShelf() {
    byId("selected-count").textContent = state.selected.length;
    byId("shelf-empty").hidden = state.selected.length > 0;
    byId("shelf-fieldset").hidden = !state.selected.length;
    byId("clear-history").disabled = !state.selected.length;
    const cards = state.selected.map((id) => {
      const movie = state.byId.get(id);
      const active = id === state.activeId;
      const card = element("div", `selected-card${active ? " is-active" : ""}`);
      const label = element("label");
      const radio = element("input");
      radio.type = "radio";
      radio.name = "active-movie";
      radio.id = `active-${id}`;
      radio.value = id;
      radio.checked = active;
      const title = element("span", "selected-title", movie.title);
      if (active) title.append(element("span", "active-label", "Active movie"));
      label.append(radio, title);
      radio.addEventListener("change", () => {
        state.activeId = id;
        renderShelf();
        renderRecommendations();
        byId(`active-${id}`).focus();
      });
      const remove = element("button", "button-text remove-movie", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${movie.title}`);
      remove.addEventListener("click", () => {
        const index = state.selected.indexOf(id);
        state.selected = state.selected.filter((selectedId) => selectedId !== id);
        if (active) state.activeId = state.selected[Math.min(index, state.selected.length - 1)] ?? null;
        refresh();
        const focusId = state.selected[Math.min(index, state.selected.length - 1)];
        (focusId ? byId(`active-${focusId}`) : search).focus();
      });
      card.append(label, remove);
      return card;
    });
    byId("selected-movies").replaceChildren(...cards);
    const profile = averageProfile(state.selected.map((id) => state.byId.get(id)));
    byId("profile-summary").hidden = !state.selected.length;
    const tags = GENRES.flatMap((genre, index) => profile[index] > 0
      ? [element("span", "genre-tag", `${genre} ${Math.round(profile[index] * 100)}%`)] : []);
    byId("profile-genres").replaceChildren(...(tags.length ? tags : [element("span", "small muted", "No known genres on this shelf.")]));
    byId("profile-genres").title = "Percentage of your selected movies carrying each genre; values need not sum to 100%.";
  }

  function renderList(mode, movies) {
    byId(`${mode}-results`).replaceChildren(...movies.map((movie, index) => {
      const card = element("li", "recommendation-card");
      card.dataset.movieId = movie.id;
      const detail = element("div");
      detail.append(element("h4", "", movie.title), element("p", "movie-genres", movie.genres.join(" · ") || "No known genres"));
      const metrics = element("div", "movie-metrics");
      const score = element("span", "score", `${(movie.score * 100).toFixed(1)}% similarity`);
      score.title = `Cosine similarity: ${movie.score.toFixed(6)}`;
      metrics.append(score, element("span", "rating-count", `${formatter.format(movie.ratingCount)} ratings`));
      detail.append(metrics);
      const rank = element("span", "rank", String(index + 1).padStart(2, "0"));
      rank.setAttribute("aria-hidden", "true");
      card.append(rank, detail);
      return card;
    }));
    const empty = byId(`${mode}-empty`);
    empty.hidden = movies.length > 0;
    empty.textContent = state.selected.length
      ? "No positive genre matches. Try a movie with known genres."
      : "Add a movie to your shelf to see recommendations.";
  }

  function renderRecommendations() {
    const lists = recommend(state.movies, state.selected, state.activeId);
    const active = state.byId.get(state.activeId);
    byId("item-context").textContent = active ? `Because you liked ${active.title}` : "Choose a movie to set the direction.";
    byId("profile-context").textContent = state.selected.length
      ? `Blending the original genre vectors of ${state.selected.length} liked ${state.selected.length === 1 ? "movie" : "movies"}`
      : "Add a few favorites to blend their genres.";
    renderList("item", lists.item);
    renderList("profile", lists.profile);
    const shared = lists.item.filter((movie) => lists.profile.some((other) => movie.id === other.id)).length;
    const overlap = byId("overlap-summary");
    overlap.hidden = !state.selected.length;
    overlap.textContent = `${shared} ${shared === 1 ? "movie appears" : "movies appear"} in both lists. Popularity is shown for context, never used to rank.`;
    byId("recommendation-status").textContent = state.selected.length
      ? `${state.selected.length} movies on your shelf. Active: ${active?.title ?? "none"}. ${lists.item.length} item-to-item and ${lists.profile.length} profile-based recommendations. ${shared} shared.`
      : "Your shelf is empty. Add a movie to begin.";
  }

  function refresh() { updateOptions(); renderShelf(); renderRecommendations(); }

  byId("movie-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const id = Number(select.value);
    if (!state.byId.has(id) || state.selected.includes(id)) return;
    state.selected.push(id);
    state.activeId = id;
    search.value = "";
    refresh();
    search.focus();
  });
  search.addEventListener("input", updateOptions);
  select.addEventListener("change", () => { add.disabled = !select.value; });
  byId("clear-history").addEventListener("click", () => {
    state.selected = [];
    state.activeId = null;
    refresh();
    search.focus();
  });
  byId("try-example").addEventListener("click", () => {
    state.selected = [1, 50, 100].filter((id) => state.byId.has(id));
    state.activeId = state.selected.at(-1) ?? null;
    search.value = "";
    refresh();
    if (state.activeId) byId(`active-${state.activeId}`).focus();
  });

  async function load() {
    byId("workspace").setAttribute("aria-busy", "true");
    byId("load-error").hidden = true;
    byId("load-status").textContent = "Loading the MovieLens catalog and rating counts…";
    byId("retry-load").disabled = true;
    try {
      const { movies, ratings } = await loadData();
      state.movies = [...movies].sort((a, b) => a.title.localeCompare(b.title, "en") || a.id - b.id);
      state.byId = new Map(movies.map((movie) => [movie.id, movie]));
      search.disabled = false;
      byId("try-example").disabled = false;
      byId("load-status").textContent = `${formatter.format(movies.length)} movies · ${formatter.format(ratings.length)} ratings · MovieLens 100K ready to explore`;
      refresh();
    } catch (error) {
      byId("load-status").textContent = "Catalog unavailable.";
      byId("error-message").textContent = `Could not load MovieLens. ${error.message} Serve this folder over HTTP and check that u.item and u.data are available, then try again.`;
      byId("load-error").hidden = false;
    } finally {
      byId("workspace").setAttribute("aria-busy", "false");
      byId("retry-load").disabled = false;
    }
  }
  byId("retry-load").addEventListener("click", load);
  await load();
}

if (typeof document !== "undefined" && document.querySelector("[data-movie-app]")) {
  initApp();
}
