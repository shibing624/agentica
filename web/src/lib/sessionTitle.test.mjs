import test, { after } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const { titleForFirstMessage, shouldApplyDisplayTitle } = await vite.ssrLoadModule("/src/lib/sessionTitle.ts");
after(() => vite.close());

test("first message replaces the local New Chat draft label", () => {
  assert.equal(
    titleForFirstMessage({ title: "New Chat", named: false }, "帮我整理今天的 CI 失败原因"),
    "帮我整理今天的 CI 失败原因",
  );
});

test("a user-picked title is not replaced", () => {
  assert.equal(
    titleForFirstMessage({ title: "我的巡检", named: true }, "帮我整理今天的 CI 失败原因"),
    "我的巡检",
  );
});

test("a generated catalog title replaces the first-message placeholder", () => {
  assert.equal(
    shouldApplyDisplayTitle({ title: "帮我整理今天的 CI 失败原因", named: false }, "CI 失败原因整理"),
    true,
  );
});

test("a user-picked title is not replaced by the catalog", () => {
  assert.equal(
    shouldApplyDisplayTitle({ title: "我的巡检", named: true }, "CI 失败原因整理"),
    false,
  );
});