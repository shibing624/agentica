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
const {
  appendThink,
  finishThink,
  settleWork,
  workSummary,
} = await vite.ssrLoadModule("/src/lib/msgParts.ts");
after(() => vite.close());

function msg(parts) {
  return { role: "assistant", content: "", parts, steps: [] };
}

test("finishThink freezes every open think, not only the last", () => {
  const m = msg([
    { kind: "think", text: "a", t0: 1000 },
    { kind: "tool", name: "glob", argsStr: "{}", t0: 1300, ms: 40, result: "ok" },
    { kind: "think", text: "b", t0: 1400 },
  ]);
  finishThink(m, 2000);
  assert.equal(m.parts[0].ms, 300);
  assert.equal(m.parts[2].ms, 600);
  assert.equal(workSummary(m.parts, 9_000_000).running, false);
});

test("earlier think duration is next tool t0, not wall clock at settle", () => {
  const m = msg([
    { kind: "think", text: "a", t0: 1000 },
    { kind: "tool", name: "glob", argsStr: "{}", t0: 1300, ms: 20, result: "ok" },
    { kind: "think", text: "b", t0: 1500 },
    { kind: "tool", name: "execute", argsStr: "{}", t0: 1800, ms: 50, result: "ok" },
  ]);
  const changed = settleWork(m);
  assert.equal(changed, true);
  assert.equal(m.parts[0].ms, 300);
  assert.equal(m.parts[2].ms, 300);
  const sum = workSummary(m.parts, Date.now());
  assert.equal(sum.running, false);
  assert.equal(sum.ms, 300 + 20 + 300 + 50);
});

test("settling a stale card does not use Date.now() for a trailing think", () => {
  const t0 = Date.now() - 23 * 3600_000;
  const m = msg([
    { kind: "think", text: "left open", t0 },
    { kind: "text", text: "done" },
  ]);
  settleWork(m);
  assert.equal(m.parts[0].ms, 0);
  assert.equal(workSummary(m.parts).running, false);
});

test("workSummary is not running when the message is no longer live", () => {
  const items = [
    { kind: "think", text: "a", t0: 1 },
    { kind: "tool", name: "glob", argsStr: "{}", t0: 2, result: "ok", ms: 10 },
  ];
  assert.equal(workSummary(items, 50_000, false).running, false);
  assert.equal(workSummary(items, 50_000, true).running, true);
});

test("appendThink after a frozen slice opens a new think", () => {
  const m = msg([{ kind: "think", text: "a", t0: 1000, ms: 200 }]);
  m.steps = [{ type: "thinking", text: "a", t0: 1000, ms: 200 }];
  appendThink(m, " more");
  assert.equal(m.parts.length, 2);
  assert.equal(m.parts[0].text, "a");
  assert.equal(m.parts[1].kind, "think");
  assert.equal(m.parts[1].text, " more");
  assert.equal(m.parts[1].ms, undefined);
});
