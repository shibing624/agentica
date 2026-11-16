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
const W = await vite.ssrLoadModule("/src/lib/workspacePick.ts");
after(() => vite.close());

test("a chosen folder is picked, a cancelled dialog is not an error", () => {
  assert.deepEqual(W.pickOutcome({ ok: true, status: 200, data: { path: "/a/b" } }), { kind: "picked", path: "/a/b" });
  assert.deepEqual(W.pickOutcome({ ok: true, status: 200, data: { path: null } }), { kind: "cancelled" });
});

test("a remote client, a system without a dialog, or a failed request falls back to the page picker", () => {
  assert.deepEqual(W.pickOutcome({ ok: false, status: 409, data: { detail: "x" } }), { kind: "fallback" });
  assert.deepEqual(W.pickOutcome({ ok: false, status: 501, data: null }), { kind: "fallback" });
  assert.deepEqual(W.pickOutcome({ ok: false, status: 0, data: null }), { kind: "fallback" });
});
