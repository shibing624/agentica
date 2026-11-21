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
const { preferredWidth, resolvePanelLayout, resolveTreeLayout } = await vite.ssrLoadModule("/src/workspace/panelWidth.ts");
after(() => vite.close());

test("first open takes 45% of the window", () => {
  assert.equal(preferredWidth(null, 1600), 720);
});

test("a saved drag is clamped to [300, 70% of the window]", () => {
  assert.equal(preferredWidth(100, 1600), 300);
  assert.equal(preferredWidth(1500, 1600), 1120);
  assert.equal(preferredWidth(800, 1600), 800);
});

test("the dock keeps room for chat; a narrow viewport uses a reachable overlay instead of a zero-width track", () => {
  // 1600 - 280 sidebar - 400 chat = 920 available; 1120 wanted.
  assert.deepEqual(resolvePanelLayout(1120, 1600, 280), { width: 920, floating: false });
  // 1000 - 280 - 400 = 320, still above the 300 floor.
  assert.deepEqual(resolvePanelLayout(450, 1000, 280), { width: 320, floating: false });
  assert.deepEqual(resolvePanelLayout(405, 900, 280), { width: 405, floating: true });
  assert.deepEqual(resolvePanelLayout(300, 280, 0), { width: 280, floating: true });
  // Collapsed sidebar gives the room back.
  assert.deepEqual(resolvePanelLayout(405, 900, 0), { width: 405, floating: false });
});

test("a wider tree leaves preview space, uses an overlay when needed, and fills only narrow panes", () => {
  assert.deepEqual(resolveTreeLayout(760, 380), { mode: "beside", width: 380, min: 180, max: 424 });
  assert.deepEqual(resolveTreeLayout(560, 380), { mode: "beside", width: 224, min: 180, max: 224 });
  assert.deepEqual(resolveTreeLayout(500, 460), { mode: "overlay", width: 364, min: 180, max: 364 });
  assert.deepEqual(resolveTreeLayout(399, 460), { mode: "fill", width: 399, min: 399, max: 399 });
});
