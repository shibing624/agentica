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
const { normalizeApprovalMode } = await vite.ssrLoadModule("/src/lib/approvalMode.ts");
after(() => vite.close());

test("the retired ask tier migrates to read-only, not to the default", () => {
  // `ask` was renamed, not removed: dropping it would read as "no preference"
  // and hand a saved Read Only session the `auto` (Workspace Write) default.
  assert.equal(normalizeApprovalMode("ask"), "read-only");
  assert.equal(normalizeApprovalMode("  ASK  "), "read-only");
});

test("current tiers survive unchanged", () => {
  for (const mode of ["read-only", "auto", "allow-all"]) {
    assert.equal(normalizeApprovalMode(mode), mode);
  }
});

test("a value that was never a tier still falls back to the default", () => {
  assert.equal(normalizeApprovalMode("strict"), "auto");
  assert.equal(normalizeApprovalMode(""), "auto");
  assert.equal(normalizeApprovalMode(null), "auto");
  assert.equal(normalizeApprovalMode(undefined), "auto");
  assert.equal(normalizeApprovalMode(3), "auto");
});
