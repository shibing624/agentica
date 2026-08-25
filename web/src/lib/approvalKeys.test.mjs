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
const { approvalKeyAction } = await vite.ssrLoadModule("/src/lib/approvalKeys.ts");
after(() => vite.close());

test("Enter allows once; Escape denies", () => {
  assert.equal(approvalKeyAction("Enter"), "allow");
  assert.equal(approvalKeyAction("Enter", { composerText: "" }), "allow");
  assert.equal(approvalKeyAction("Enter", { composerText: "   " }), "allow");
  assert.equal(approvalKeyAction("Escape"), "deny");
});

test("leftover composer text does not steal Enter; Shift+Enter is a newline", () => {
  assert.equal(approvalKeyAction("Enter", { composerText: "don't rm that" }), "allow");
  assert.equal(approvalKeyAction("Enter", { shiftKey: true, composerText: "" }), null);
});

test("other keys are ignored", () => {
  assert.equal(approvalKeyAction("y"), null);
  assert.equal(approvalKeyAction("n"), null);
});
