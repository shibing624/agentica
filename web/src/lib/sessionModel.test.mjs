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
after(() => vite.close());
const { setSessionModelApi } = await vite.ssrLoadModule("/src/api.ts");

test("model updates distinguish profile selection from setting and resetting effort", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch", async () => new Response("{}"));
  await setSessionModelApi("session#1", "profile-a", "/workspace");
  await setSessionModelApi("session#1", "profile-a", "/workspace", "max");
  await setSessionModelApi("session#1", "profile-a", "/workspace", null);
  assert.deepEqual(fetch.mock.calls.map(({ arguments: [url, init] }) => [url, JSON.parse(init.body)]), [
    ["/api/sessions/session%231/model?work_dir=%2Fworkspace", { profile: "profile-a" }],
    ["/api/sessions/session%231/model?work_dir=%2Fworkspace", { profile: "profile-a", reasoning_effort_override: "max" }],
    ["/api/sessions/session%231/model?work_dir=%2Fworkspace", { profile: "profile-a", reasoning_effort_override: null }],
  ]);
});