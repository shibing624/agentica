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
const { agoStr, lastQueryTs, sessionSideStatus } = await vite.ssrLoadModule("/src/lib/format.ts");
after(() => vite.close());

const zh = {
  agoJustNow: "刚刚",
  agoMinutes: (n) => `${n}分钟前`,
  agoHours: (n) => `${n}小时前`,
  agoDays: (n) => `${n}天前`,
};
const en = {
  agoJustNow: "just now",
  agoMinutes: (n) => `${n} min ago`,
  agoHours: (n) => (n === 1 ? "1 hour ago" : `${n} hours ago`),
  agoDays: (n) => (n === 1 ? "1 day ago" : `${n} days ago`),
};

const now = Date.parse("2026-08-22T15:32:00Z");

test("idle label is coarse and bilingual", () => {
  assert.equal(agoStr(now - 30_000, zh, now), "刚刚");
  assert.equal(agoStr(now - 4 * 60_000, en, now), "just now");
  assert.equal(agoStr(now - 7 * 60_000, zh, now), "5分钟前");
  assert.equal(agoStr(now - 12 * 60_000, en, now), "10 min ago");
  assert.equal(agoStr(now - 6 * 3_600_000, zh, now), "6小时前");
  assert.equal(agoStr(now - 1 * 3_600_000, en, now), "1 hour ago");
  assert.equal(agoStr(now - 11 * 3_600_000, en, now), "11 hours ago");
  assert.equal(agoStr(now - 2 * 86_400_000, zh, now), "2天前");
  assert.equal(agoStr(now - 1 * 86_400_000, en, now), "1 day ago");
});

test("lastQueryTs prefers the latest non-steer user turn", () => {
  const sess = {
    ts: 100,
    lastTs: 200,
    msgs: [
      { role: "user", ts: 300 },
      { role: "assistant", ts: 400 },
      { role: "user", ts: 500, steer: true },
    ],
  };
  assert.equal(lastQueryTs(sess), 300);
  assert.equal(lastQueryTs({ ts: 100, lastTs: 200, msgs: [] }), 200);
  assert.equal(lastQueryTs({ ts: 100, msgs: [] }), 100);
});

test("sidebar slot is busy, unread, or idle — never two at once", () => {
  assert.equal(sessionSideStatus({ id: "a", curSess: "b", running: true, unread: true }), "busy");
  assert.equal(sessionSideStatus({ id: "a", curSess: "b", running: false, unread: true }), "unread");
  assert.equal(sessionSideStatus({ id: "a", curSess: "a", running: false, unread: true }), "idle");
  assert.equal(sessionSideStatus({ id: "a", curSess: "b", running: false, unread: false }), "idle");
});
