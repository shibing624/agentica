import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const storage = new Map();
globalThis.localStorage = {
  getItem: (key) => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, value),
  removeItem: (key) => storage.delete(key),
};
const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)), configFile: false, appType: "custom",
  logLevel: "silent", server: { middlewareMode: true },
});
const { loadSessions, pollSessionsPulse } = await vite.ssrLoadModule("/src/sessions.ts");
const { getState, setState } = await vite.ssrLoadModule("/src/store.ts");
const { sessionSideStatus } = await vite.ssrLoadModule("/src/lib/format.ts");
after(() => vite.close());

test("a completion pulse is not acknowledged by an older catalog or a failed refresh", async (t) => {
  const completed = { session_id: "background", running: false, unread: true, size_bytes: 500 };
  const running = { ...completed, running: true, unread: false, size_bytes: 200 };
  const reply = (row) => Response.json({ sessions: [row] });
  let releaseCatalog;
  const oldCatalog = new Promise((resolve) => { releaseCatalog = resolve; });
  let version = 1;
  const catalogs = [() => reply(running), () => oldCatalog, () => new Response(null, { status: 503 }), () => reply(completed)];
  t.mock.method(globalThis, "fetch", async (url) => {
    if (url === "/api/sessions/pulse") return Response.json({ ver: version });
    if (url.startsWith("/api/sessions?")) return catalogs.shift()();
    return Response.json({ jobs: [] });
  });
  setState({ curSess: "other" });
  await pollSessionsPulse();
  assert.equal(getState().sessions.background.running, true);
  const refresh = loadSessions();
  version = 2;
  const pulse = pollSessionsPulse();
  releaseCatalog(reply(running));
  await Promise.all([refresh, pulse]);
  assert.equal(catalogs.length, 1, "pulse must request a fresh catalog after the older request");
  await pollSessionsPulse();
  const { running: busy, unread } = getState().sessions.background;
  assert.equal(sessionSideStatus({ id: "background", curSess: "other", running: busy, unread }), "unread");
  await pollSessionsPulse();
  assert.equal(catalogs.length, 0);
});