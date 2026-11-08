import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const { isStaleBundleError, shouldAutoReload, RELOAD_WINDOW_MS } = await vite.ssrLoadModule("/src/lib/staleBundle.ts");
after(() => vite.close());

const memory = () => {
  const m = new Map();
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, v) };
};
const STALE = new Error("Failed to fetch dynamically imported module: http://x/assets/TracesPage-abc.js");

test("a lazy chunk that is gone is a stale-bundle error, an ordinary error is not", () => {
  assert.equal(isStaleBundleError(new TypeError("Failed to fetch dynamically imported module: http://x/assets/TracesPage-abc.js")), true);
  assert.equal(isStaleBundleError(new Error("error loading dynamically imported module")), true);
  assert.equal(isStaleBundleError(new Error("Importing a module script failed.")), true);
  assert.equal(isStaleBundleError(new Error("Cannot read properties of undefined (reading 'startsWith')")), false);
  assert.equal(isStaleBundleError(null), false);
  assert.equal(isStaleBundleError(undefined), false);
});

test("a stale bundle reloads once, and a reload that does not cure it ends in a message", () => {
  const store = memory();
  const t0 = 1_000_000;
  assert.equal(shouldAutoReload(STALE, store, t0), true);
  // The reloaded page hits the same missing file a moment later: no second reload,
  // however many times it is asked (this was an endless reload loop).
  for (const dt of [50, 400, 3_000, RELOAD_WINDOW_MS - 1]) {
    assert.equal(shouldAutoReload(STALE, store, t0 + dt), false);
  }
});

test("a later upgrade, long after the first, may reload again", () => {
  const store = memory();
  assert.equal(shouldAutoReload(STALE, store, 1_000), true);
  assert.equal(shouldAutoReload(STALE, store, 1_000 + RELOAD_WINDOW_MS + 1), true);
});

test("an ordinary render error never triggers a reload or leaves a mark", () => {
  const store = memory();
  assert.equal(shouldAutoReload(new Error("boom"), store, 5), false);
  assert.equal(store.getItem("ag_chunk_reload"), null);
});
