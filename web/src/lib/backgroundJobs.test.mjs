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

test("visibleJobBadgeCount shows live count when work is running, else settled count", () => {
  assert.equal(B.visibleJobBadgeCount(2, 5), 2);
  assert.equal(B.visibleJobBadgeCount(0, 3), 3);
  assert.equal(B.visibleJobBadgeCount(0, 0), 0);
});

test("parseBackgroundStart extracts id, num, pid, and log path from execute(background=True) receipt", () => {
  const parsed = B.parseBackgroundStart(
    "Started background command #3 (PID 74981, id: term_3).\nLog: /Users/me/.agentica/projects/default/x/background/20261111-term_3.log",
  );
  assert.deepEqual(parsed, {
    num: 3,
    pid: 74981,
    id: "term_3",
    logPath: "/Users/me/.agentica/projects/default/x/background/20261111-term_3.log",
  });
  assert.equal(B.parseBackgroundStart("hello world\n[Exit code: 0]"), null);
});
