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
const { normalizeApprovalMode, resolveChatTier } = await vite.ssrLoadModule("/src/lib/approvalMode.ts");
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

test("a chat's own tier always wins", () => {
  assert.equal(resolveChatTier("s1", "read-only", "allow-all", true), "read-only");
  assert.equal(resolveChatTier("s1", "ask", "allow-all", true), "read-only");
});

test("a chat from before per-chat tiers uses the gateway default, not the pick", () => {
  // The old conversation has no recorded tier; inheriting the tier just picked
  // for another chat would silently loosen (or tighten) it.
  assert.equal(resolveChatTier("old-chat", undefined, "allow-all", true), "auto");
  assert.equal(resolveChatTier("old-chat", null, "read-only", true), "auto");
  assert.equal(resolveChatTier("old-chat", "", "allow-all", true), "auto");
});

test("a new chat starts on the current pick", () => {
  // No session yet, or a session whose server-side log is still empty (the
  // first send has already pushed its user bubble by then).
  assert.equal(resolveChatTier(null, undefined, "allow-all", false), "allow-all");
  assert.equal(resolveChatTier("fresh", undefined, "read-only", false), "read-only");
});
