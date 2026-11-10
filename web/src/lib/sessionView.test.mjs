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
const { createViewStore, MAX_VIEWS } = await vite.ssrLoadModule("/src/lib/sessionView.ts");
after(() => vite.close());

function memoryStorage() {
  const data = new Map();
  return {
    data,
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => { data.set(k, v); },
  };
}

test("a view survives a reload: a new store reads what the old one flushed", () => {
  const storage = memoryStorage();
  const a = createViewStore(storage, () => "ag_view:u1");
  a.update("s1", { page: "traces" });
  a.update("s1", { trace: { call: 7 } });
  a.update("s1", { trace: { scrollTop: 420 } });
  a.flush();

  const b = createViewStore(storage, () => "ag_view:u1");
  assert.deepEqual(b.get("s1"), { page: "traces", trace: { call: 7, scrollTop: 420 } });
});

test("views are per account", () => {
  const storage = memoryStorage();
  let account = "u1";
  const store = createViewStore(storage, () => "ag_view:" + account);
  store.update("s1", { page: "traces" });
  store.flush();
  account = "u2";
  assert.equal(store.get("s1"), undefined);
});

test("the least recently touched session is dropped first", () => {
  const storage = memoryStorage();
  const store = createViewStore(storage, () => "k");
  for (let i = 0; i < MAX_VIEWS; i++) store.update("s" + i, { page: "chat" });
  store.update("s0", { page: "traces" });
  store.update("new", { page: "chat" });
  store.flush();
  assert.equal(store.get("s1"), undefined);
  assert.deepEqual(store.get("s0"), { page: "traces" });
  assert.equal(JSON.parse(storage.data.get("k")).length, MAX_VIEWS);
});
