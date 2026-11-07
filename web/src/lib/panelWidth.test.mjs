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
const { preferredWidth, resolvedWidth } = await vite.ssrLoadModule("/src/workspace/panelWidth.ts");
after(() => vite.close());

test("first open takes 45% of the window", () => {
  assert.equal(preferredWidth(null, 1600), 720);
});

test("a saved drag is clamped to [300, 70% of the window]", () => {
  assert.equal(preferredWidth(100, 1600), 300);
  assert.equal(preferredWidth(1500, 1600), 1120);
  assert.equal(preferredWidth(800, 1600), 800);
});

test("the chat keeps 400px: the panel shrinks first, then closes", () => {
  // 1600 - 280 sidebar - 400 chat = 920 available; 1120 wanted.
  assert.equal(resolvedWidth(1120, 1600, 280), 920);
  // 1000 - 280 - 400 = 320, still above the 300 floor.
  assert.equal(resolvedWidth(450, 1000, 280), 320);
  // 900 - 280 - 400 = 220 < 300: no room, the track closes.
  assert.equal(resolvedWidth(405, 900, 280), 0);
  // Collapsed sidebar gives the room back.
  assert.equal(resolvedWidth(405, 900, 0), 405);
});
