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
const T = await vite.ssrLoadModule("/src/lib/panelTabs.ts");
after(() => vite.close());

const file = (path) => ({ key: T.fileKey(path), kind: "file", path });
const term = (id, ordinal) => ({ key: T.terminalKey(id), kind: "terminal", id, ordinal });
const keys = (st) => st.tabs.map((t) => t.key);

test("opening a file twice focuses the tab it already has", () => {
  let st = T.initialTabs();
  st = T.openTab(st, file("a.py"));
  st = T.openTab(st, file("b.py"));
  st = T.openTab(st, file("a.py"));
  assert.deepEqual(keys(st), ["files", "file:a.py", "file:b.py"]);
  assert.equal(st.active, "file:a.py");
});

test("a new tab opens right after the active one", () => {
  let st = T.initialTabs();
  st = T.openTab(st, file("a.py"));
  st = T.openTab(st, file("b.py"));
  st = T.activateTab(st, "file:a.py");
  st = T.openTab(st, file("c.py"));
  assert.deepEqual(keys(st), ["files", "file:a.py", "file:c.py", "file:b.py"]);
});

test("closing the active tab activates its right neighbour, else its left", () => {
  let st = T.initialTabs();
  for (const p of ["a", "b", "c"]) st = T.openTab(st, file(p));
  st = T.activateTab(st, "file:b");
  st = T.closeTab(st, "file:b");
  assert.equal(st.active, "file:c");
  st = T.closeTab(st, "file:c");
  assert.equal(st.active, "file:a");
  // Closing an inactive tab leaves the selection alone.
  st = T.openTab(st, file("d"));
  st = T.closeTab(st, "file:a");
  assert.equal(st.active, "file:d");
});

test("the file browser tab cannot be closed", () => {
  const st = T.openTab(T.initialTabs(), file("a"));
  assert.deepEqual(keys(T.closeTab(st, "files")), ["files", "file:a"]);
});

test("terminal numbers reuse the lowest free one", () => {
  let st = T.initialTabs();
  st = T.openTab(st, term("x", T.nextTerminalOrdinal(st)));
  st = T.openTab(st, term("y", T.nextTerminalOrdinal(st)));
  assert.deepEqual(st.tabs.slice(1).map((t) => t.ordinal), [1, 2]);
  st = T.closeTab(st, T.terminalKey("x"));
  assert.equal(T.nextTerminalOrdinal(st), 1);
});

test("adopting terminals after a reload adds the missing ones only, without stealing focus", () => {
  let st = T.openTab(T.initialTabs(), term("x", 1));
  st = T.activateTab(st, "files");
  st = T.adoptTerminals(st, ["x", "y", "z"]);
  assert.deepEqual(keys(st), ["files", "term:x", "term:y", "term:z"]);
  assert.deepEqual(st.tabs.slice(1).map((t) => t.ordinal), [1, 2, 3]);
  assert.equal(st.active, "files");
});

test("the start page turns into the page picked on it, in its own place", () => {
  let st = T.openTab(T.initialTabs(), file("a"));
  st = T.activateTab(st, "files");
  st = T.openTab(st, T.GUIDE_TAB);
  assert.deepEqual(keys(st), ["files", "guide", "file:a"]);
  st = T.replaceTab(st, "guide", term("x", 1));
  assert.deepEqual(keys(st), ["files", "term:x", "file:a"]);
  assert.equal(st.active, "term:x");
});

test("picking a page that is already open reveals it and closes the start page", () => {
  let st = T.openTab(T.initialTabs(), T.GUIDE_TAB);
  assert.equal(st.active, "guide");
  st = T.replaceTab(st, "guide", T.FILES_TAB);
  assert.deepEqual(keys(st), ["files"]);
  assert.equal(st.active, "files");
});

test("there is one start page: opening it again focuses it", () => {
  let st = T.openTab(T.initialTabs(), T.GUIDE_TAB);
  st = T.openTab(st, file("a"));
  st = T.openTab(st, T.GUIDE_TAB);
  assert.deepEqual(keys(st), ["files", "guide", "file:a"]);
  assert.equal(st.active, "guide");
});

test("a file's diff tab is reused: reopening it swaps the edits in place and focuses it", () => {
  const diff = (path, n) => ({ key: T.diffKey(path), kind: "diff", path, edits: [[{ oldStart: n, oldLines: 1, newStart: n, newLines: 1, lines: ["+x"] }]] });
  let st = T.upsertTab(T.initialTabs(), diff("a.py", 1));
  st = T.openTab(st, file("b"));
  st = T.upsertTab(st, diff("a.py", 9));
  assert.deepEqual(keys(st), ["files", "diff:a.py", "file:b"]);
  assert.equal(st.active, "diff:a.py");
  assert.equal(st.tabs[1].edits[0][0].oldStart, 9);
});
