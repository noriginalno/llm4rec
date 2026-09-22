/** Optional end-to-end check. Starts a real static server at a Pages-like /hw2/. */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = fileURLToPath(new URL("../../", import.meta.url));
const types = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".md": "text/plain" };
const server = createServer(async (request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
  const path = resolve(root, `.${pathname.endsWith("/") ? `${pathname}index.html` : pathname}`);
  if (!path.startsWith(root.endsWith(sep) ? root : root + sep)) { response.writeHead(403).end(); return; }
  try {
    const body = await readFile(path);
    response.writeHead(200, { "Content-Type": types[extname(path)] ?? "application/octet-stream" }).end(body);
  } catch { response.writeHead(404).end("Not found"); }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${server.address().port}/hw2/`;
let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  await page.route("**/u.data", async (route) => { await gate; await route.continue(); });
  await page.goto(url, { waitUntil: "domcontentloaded" });
  assert.match(await page.locator("#load-status").textContent(), /Loading/);
  assert.equal(await page.locator("#movie-search").isDisabled(), true);
  release();
  await page.waitForFunction(() => document.querySelector("#load-status").textContent.includes("ready to explore"));
  assert.match(await page.locator("#load-status").textContent(), /1,682 movies · 100,000 ratings/);
  assert.equal(await page.locator(".recommendation-card").count(), 0);

  // Search + Enter is sufficient to select the sole match; no pointer required.
  const search = page.locator("#movie-search");
  await search.fill("Toy Story (1995)");
  await search.press("Enter");
  assert.equal(await page.locator("#selected-count").textContent(), "1");
  const ids = (mode) => page.locator(`#${mode}-results li`).evaluateAll((nodes) => nodes.map((node) => Number(node.dataset.movieId)));
  assert.deepEqual(await ids("item"), await ids("profile"));
  assert.equal((await ids("item")).length, 5);
  await search.fill("Toy Story (1995)");
  assert.equal(await page.locator("#add-movie").isDisabled(), true);
  await search.fill("a nonexistent title zzzz");
  assert.match(await page.locator("#search-count").textContent(), /0 available/);
  await search.fill("Star Wars (1977)");
  await search.press("Enter");
  await search.fill("Fargo (1996)");
  await search.press("Enter");
  const before = { item: await ids("item"), profile: await ids("profile") };
  for (const list of Object.values(before)) assert.ok(list.every((id) => ![1, 50, 100].includes(id)));
  await page.locator("#active-100").focus();
  await page.keyboard.press("ArrowLeft");
  assert.equal(await page.locator("#active-50").isChecked(), true);
  assert.equal(await page.locator("#active-50").evaluate((node) => node === document.activeElement), true);
  assert.deepEqual(await ids("profile"), before.profile);
  assert.notDeepEqual(await ids("item"), before.item);
  await page.getByRole("button", { name: "Remove Star Wars (1977)", exact: true }).click();
  assert.equal(await page.locator("#active-100").isChecked(), true);
  assert.equal(await page.locator("#active-100").evaluate((node) => node === document.activeElement), true);
  await page.getByRole("button", { name: "Remove Toy Story (1995)", exact: true }).click();
  assert.equal(await page.locator("#selected-count").textContent(), "1");
  await page.getByRole("button", { name: "Remove Fargo (1996)", exact: true }).click();
  assert.equal(await page.locator(".recommendation-card").count(), 0);
  assert.equal(await search.evaluate((node) => node === document.activeElement), true);
  await search.fill("unknown");
  await search.press("Enter");
  assert.match(await page.locator("#item-empty").textContent(), /No positive genre matches/);
  assert.equal(await page.locator(".recommendation-card").count(), 0);
  await page.locator("#clear-history").click();
  await search.fill("Misérables");
  assert.match(await page.locator("#movie-select option:checked").textContent(), /Misérables, Les/);
  await page.locator("#try-example").click();
  assert.equal(await page.locator(".recommendation-card").count(), 10);
  assert.equal(await page.locator(".movie-genres").count(), 10);
  assert.equal(await page.locator(".rating-count").count(), 10);
  assert.equal(await page.locator(".score").count(), 10);
  const columns = await page.locator(".recommendation-grid").evaluate((node) => getComputedStyle(node).gridTemplateColumns.split(" ").length);
  assert.equal(columns, 2);

  const screenshotDirectory = process.env.BROWSER_SCREENSHOTS;
  if (screenshotDirectory) {
    await mkdir(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: resolve(screenshotDirectory, "desktop.png"), fullPage: true });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.locator(".recommendation-grid").evaluate((node) => getComputedStyle(node).gridTemplateColumns.split(" ").length), 1);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  if (screenshotDirectory) await page.screenshot({ path: resolve(screenshotDirectory, "mobile.png"), fullPage: true });
  await page.setViewportSize({ width: 320, height: 700 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));

  // Real loading failures and malformed data both get recoverable error states.
  await page.unroute("**/u.data");
  await page.route("**/u.data", (route) => route.fulfill({ status: 404, body: "Not found" }));
  await page.reload();
  await page.locator("#load-error").waitFor({ state: "visible" });
  assert.match(await page.locator("#error-message").textContent(), /HTTP 404/);
  assert.equal(await search.isDisabled(), true);
  await page.unroute("**/u.data");
  await page.locator("#retry-load").click();
  await page.waitForFunction(() => document.querySelector("#load-status").textContent.includes("ready to explore"));
  assert.equal(await page.locator("#load-error").isVisible(), false);
  await page.route("**/u.item", (route) => route.fulfill({ status: 200, body: "broken data" }));
  await page.reload();
  await page.locator("#load-error").waitFor({ state: "visible" });
  assert.match(await page.locator("#error-message").textContent(), /u.item, line 1/);
  assert.deepEqual(errors, []);
  console.log(`Browser checks passed in Chromium ${browser.version()}: loading, selection, keyboard navigation, removal, exclusions, scores, empty/zero states, responsive layout, HTTP errors, retry, parser errors; no page errors.`);
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
