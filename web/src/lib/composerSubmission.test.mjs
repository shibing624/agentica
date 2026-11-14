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
const C = await vite.ssrLoadModule("/src/lib/composerSubmission.ts");
after(() => vite.close());

test("plain Enter follows the preference; the chord inverts it", () => {
  assert.equal(C.resolveSubmitMode("queue", true, "enter", true), "queue");
  assert.equal(C.resolveSubmitMode("queue", true, "chord", true), "steer");
  assert.equal(C.resolveSubmitMode("steer", true, "enter", true), "steer");
  assert.equal(C.resolveSubmitMode("steer", true, "chord", true), "queue");
});

test("nothing is steered when the turn is idle or the draft cannot steer", () => {
  // Not running: Enter starts a turn, so the preference does not apply at all.
  assert.equal(C.resolveSubmitMode("steer", false, "enter", true), "queue");
  // A slash command, a document attachment, or a turn already stopping.
  assert.equal(C.resolveSubmitMode("steer", true, "enter", false), "queue");
  assert.equal(C.resolveSubmitMode("steer", true, "chord", false), "queue");
});

test("the default is queue, and only the stored word steer changes it", () => {
  assert.equal(C.DEFAULT_BUSY_ENTER, "queue");
  assert.equal(C.readBusyEnter(null), "queue");
  assert.equal(C.readBusyEnter("steer"), "steer");
  assert.equal(C.readBusyEnter("Steer"), "queue");
  assert.equal(C.readBusyEnter("something else"), "queue");
});
