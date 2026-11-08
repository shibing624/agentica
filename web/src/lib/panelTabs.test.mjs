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

test("a round's review tab is reused: picking another of its files swaps it in place and focuses it", () => {
  const files = [{ path: "a.py", edits: [] }, { path: "b.py", edits: [] }];
  const review = (path) => ({ key: T.reviewKey(3), kind: "review", turn: 3, files, path });
  let st = T.upsertTab(T.initialTabs(), review("a.py"));
  st = T.openTab(st, file("c"));
  st = T.upsertTab(st, review("b.py"));
  assert.deepEqual(keys(st), ["files", "review:3", "file:c"]);
  assert.equal(st.active, "review:3");
  assert.equal(st.tabs[1].path, "b.py");
});

test("each session persists its own opened file tabs, active tab, open state, and directory path", () => {
  const store = new Map();
  const storage = {
    getItem: (k) => store.get(k) ?? null,
    setItem: (k, v) => { store.set(k, v); },
  };
  let s1 = T.initialTabs();
  s1 = T.openTab(s1, file("src/main.tex"));
  s1 = T.openTab(s1, file("README.md"));
  s1 = T.openTab(s1, T.GUIDE_TAB);
  s1 = T.openTab(s1, term("t1", 1));
  s1 = T.activateTab(s1, "file:src/main.tex");

  let s2 = T.initialTabs();
  s2 = T.openTab(s2, file("notes.txt"));

  T.writeSessionPanel("sess_1", { ...s1, open: true, dirPath: "src/sub" }, storage);
  T.writeSessionPanel("sess_2", { ...s2, open: false, dirPath: "docs" }, storage);

  const r1 = T.readSessionPanel("sess_1", storage);
  const r2 = T.readSessionPanel("sess_2", storage);
  assert.equal(r1.open, true);
  assert.equal(r1.dirPath, "src/sub");
  assert.equal(r1.active, "file:src/main.tex");
  // Transient guide and terminal tabs are stripped on save; file history is kept per session.
  assert.deepEqual(r1.tabs.map((t) => t.key), ["files", "file:src/main.tex", "file:README.md"]);

  assert.equal(r2.open, false);
  assert.equal(r2.dirPath, "docs");
  assert.equal(r2.active, "file:notes.txt");
  assert.deepEqual(r2.tabs.map((t) => t.key), ["files", "file:notes.txt"]);

  // When a terminal tab was active before reload, preferredActive remembers it until adoptTerminals runs.
  T.writeSessionPanel("sess_term", { ...s1, active: "term:t1", open: true, dirPath: "" }, storage);
  const rt = T.readSessionPanel("sess_term", storage);
  assert.equal(rt.active, "files");
  assert.equal(rt.preferredActive, "term:t1");
});
