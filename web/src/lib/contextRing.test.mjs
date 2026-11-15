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
const { ringPercent } = await vite.ssrLoadModule("/src/lib/contextRing.ts");
after(() => vite.close());

test("a session with history shows nothing until its own usage arrives", () => {
  // A stale local counter over another chat's window would be replaced a
  // moment later: that jump is the bug.
  assert.equal(ringPercent({ hasLog: true, settled: false, localTokens: 50_000, fallbackWindow: 1_000_000 }), null);
});

test("its own window wins over the one another chat published", () => {
  const usage = { window: 200_000, context_tokens: 114_000 };
  assert.equal(ringPercent({ usage, hasLog: true, settled: true, localTokens: 9, fallbackWindow: 1_000_000 }), 57);
  // Cached from an earlier visit: final on the first frame, before any request.
  assert.equal(ringPercent({ usage, hasLog: true, settled: false, localTokens: 0, fallbackWindow: 1_000_000 }), 57);
});

test("a new chat, or one whose request failed, shows the local figure", () => {
  assert.equal(ringPercent({ hasLog: false, settled: false, localTokens: 10_000, fallbackWindow: 100_000 }), 10);
  assert.equal(ringPercent({ hasLog: true, settled: true, localTokens: 10_000, fallbackWindow: 100_000 }), 10);
});

test("an unknown usage window falls back to the server window, not 128k", () => {
  // `usage.window === 0` means "unknown". The popover divides by the server's
  // window; the ring must do the same or the same session shows two figures.
  const usage = { window: 0, context_tokens: 50_000 };
  assert.equal(ringPercent({ usage, hasLog: true, settled: true, localTokens: 0, fallbackWindow: 100_000 }), 50);
  // Nothing at all known: the last-resort estimate is the only denominator.
  assert.equal(ringPercent({ hasLog: false, settled: false, localTokens: 64_000, fallbackWindow: 0 }), 50);
});
