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
const B = await vite.ssrLoadModule("/src/lib/backgroundJobs.ts");
after(() => vite.close());

const job = (id, status, started, finished = null) => ({
  id, status, started_at_ms: started, finished_at_ms: finished,
});

test("live jobs come first oldest-first, finished ones newest-first", () => {
  const ordered = B.orderJobs([
    job("old-done", "completed", 0, 10),
    job("live-late", "running", 50),
    job("new-done", "failed", 5, 40),
    job("live-early", "running", 20),
  ]);
  assert.deepEqual(ordered.map((j) => j.id), ["live-early", "live-late", "new-done", "old-done"]);
});

test("a finished job's duration stops at its finish time, a live one follows the clock", () => {
  assert.equal(B.jobDuration(job("a", "completed", 0, 8000), 999999), "8s");
  assert.equal(B.jobDuration(job("b", "running", 0), 185000), "3m 5s");
  assert.equal(B.jobDuration(job("c", "running", 0), 3720000), "1h 2m");
});
