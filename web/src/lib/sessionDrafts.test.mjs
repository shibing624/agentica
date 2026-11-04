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
const { emptyChatIds } = await vite.ssrLoadModule("/src/lib/sessionDrafts.ts");
after(() => vite.close());

const draft = (over = {}) => ({
  title: "New Chat", msgs: [], dir: "/work/alpha", ts: 1, ...over,
});

test("clicking + repeatedly still finds exactly one draft per workspace", () => {
  const sessions = {
    a: draft({ ts: 10 }),
    b: draft({ ts: 20 }),
    c: draft({ dir: "/work/beta", ts: 30 }),
  };
  assert.deepEqual(emptyChatIds(sessions, "/work/alpha"), ["b", "a"]);
  assert.deepEqual(emptyChatIds(sessions, "/work/beta"), ["c"]);
  assert.deepEqual(emptyChatIds(sessions, "/work/alpha/"), ["b", "a"]);
});

test("only unsent local drafts count", () => {
  const sessions = {
    sent: draft({ msgs: [{ role: "user", content: "hi" }] }),
    renamed: draft({ title: "Planning" }),
    server: draft({ sizeBytes: 900 }),
    withUsers: draft({ userCount: 2 }),
    archived: draft({ archived: true }),
    running: draft({ running: true }),
    real: draft({ ts: 5 }),
  };
  assert.deepEqual(emptyChatIds(sessions, "/work/alpha"), ["real"]);
});

test("empty workspace has no drafts", () => {
  assert.deepEqual(emptyChatIds({}, "/work/alpha"), []);
  assert.deepEqual(emptyChatIds({ a: draft() }, "/work/other"), []);
});
