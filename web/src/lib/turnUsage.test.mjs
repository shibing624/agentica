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
const { placeAbove, turnUsageRows } = await vite.ssrLoadModule("/src/lib/turnUsage.ts");
after(() => vite.close());

const L = {
  turnUsageModel: "Model", cacheHitTip: "Cache hit", turnUsageInput: "Input (uncached)",
  turnUsageCacheRead: "Cache read", turnUsageCacheWrite: "Cache write", ctxOutput: "Output",
  turnUsageRequests: "Model calls", turnUsageSpeed: "tok/s", costUsdTip: "Cost",
};
const view = (m) => {
  const { total, rows } = turnUsageRows(m, L);
  return { total, rows: Object.fromEntries(rows.map((r) => [r.label, r.value])), order: rows.map((r) => r.label) };
};

test("uncached input is the prompt minus both cache buckets, totals are exact", () => {
  const v = view({ tokIn: 470_000, tokOut: 7_700, cacheRead: 440_000, cacheWrite: 5_000, cacheHitPercent: 93.6 });
  assert.equal(v.total, "477,700");
  assert.equal(v.rows["Input (uncached)"], "25,000");
  assert.equal(v.rows["Cache read"], "440,000");
  assert.equal(v.rows["Cache write"], "5,000");
  assert.equal(v.rows["Cache hit"], "93.6%");
  assert.equal(v.rows.Output, "7,700");
});

test("rows the provider never reported are left out, not shown as 0", () => {
  const v = view({ tokIn: 1200, tokOut: 30 });
  assert.deepEqual(v.order, ["Input (uncached)", "Output"]);
});

test("model, calls, speed and cost appear when known, in a stable order", () => {
  const v = view({ tokIn: 1000, tokOut: 200, cacheRead: 0, cacheHitPercent: 0, model: "deepseek-v4",
    requests: 3, llmMs: 4000, costUsd: 0.0123 });
  assert.deepEqual(v.order, ["Model", "Cache hit", "Input (uncached)", "Cache read", "Output", "Model calls", "tok/s", "Cost"]);
  assert.equal(v.rows["tok/s"], "50.0 tok/s");
  assert.equal(v.rows.Cost, "$0.01");
});

test("panel sits above the trigger and stays inside the viewport", () => {
  const vp = { width: 1000, height: 800 };
  assert.deepEqual(placeAbove({ left: 300, top: 500, bottom: 520 }, { width: 300, height: 200 }, vp), { left: 300, top: 292 });
  // Near the right edge: shifted left to keep the 12px margin.
  assert.deepEqual(placeAbove({ left: 900, top: 500, bottom: 520 }, { width: 300, height: 200 }, vp), { left: 688, top: 292 });
  // No room above (reply at the top of the chat): drops below the trigger.
  assert.deepEqual(placeAbove({ left: 5, top: 100, bottom: 120 }, { width: 300, height: 200 }, vp), { left: 12, top: 128 });
});
