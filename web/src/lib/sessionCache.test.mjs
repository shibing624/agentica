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
const { sessionStatsCache } = await vite.ssrLoadModule("/src/lib/sessionCache.ts");
after(() => vite.close());

const stats = {
  tokIn: 10, tokOut: 2, tokTotal: 12, requests: 1, totalTime: 3,
  lastInputTokens: 10, contextTokens: 9, costUsd: 0.01,
};

test("keeps only the counters, never the transcript", () => {
  const huge = "x".repeat(100_000);
  const out = sessionStatsCache({
    a: { ...stats, title: "t", msgs: [{ role: "user", text: huge }], dir: "/d", ts: 1, lastTs: 2 },
  });
  assert.deepEqual(out, { a: stats });
});

test("sessions this browser never ran a turn in are not cached", () => {
  // Thousands of CLI sessions listed from the server: none of them belongs here.
  const sessions = {};
  for (let i = 0; i < 5000; i++) sessions[`cli-${i}`] = { ...stats, requests: 0, msgs: [] };
  sessions.web = stats;
  assert.deepEqual(Object.keys(sessionStatsCache(sessions)), ["web"]);
});
