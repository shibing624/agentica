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
const { buildChatNavTicks, previewText, CHAT_NAV_MIN_TICKS } = await vite.ssrLoadModule("/src/lib/chatNav.ts");
after(() => vite.close());

test("each tick carries the truncated query and skips steer chips", () => {
  const ticks = buildChatNavTicks([
    { role: "user", content: "first question" },
    { role: "assistant", content: "first answer" },
    { role: "user", content: "interrupt", steer: true },
    { role: "user", content: "second question" },
    { role: "assistant", parts: [{ kind: "think", text: "hmm" }, { kind: "text", text: "second answer" }] },
  ]);
  assert.equal(CHAT_NAV_MIN_TICKS, 2);
  assert.equal(ticks.length, 2);
  assert.deepEqual(ticks.map((t) => t.idx), [0, 3]);
  assert.equal(ticks[0].query, "first question");
  assert.equal(ticks[1].query, "second question");
});

test("preview collapses whitespace and truncates to 50 by default", () => {
  assert.equal(previewText("  hello   world  ", 8), "hello wo");
  const long = "一二三四五六七八九十".repeat(6);
  assert.equal(long.length, 60);
  assert.equal(previewText(long).length, 50);
  assert.equal(buildChatNavTicks([{ role: "user", content: long }])[0].query, long.slice(0, 50));
});
