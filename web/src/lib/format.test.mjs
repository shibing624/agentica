import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.document = { documentElement: { lang: "", dataset: {} } };
globalThis.window = { setTimeout, clearTimeout, localStorage: globalThis.localStorage };

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const {
  agoStr, fmtDurationMs, fmtRunDuration, fmtSpan, lastQueryTs,
  sessionSideStatus, updateQueuedMessageText,
} = await vite.ssrLoadModule("/src/lib/format.ts");
const { setState } = await vite.ssrLoadModule("/src/store.ts");
const { getStrings } = await vite.ssrLoadModule("/src/i18n.ts");
after(() => vite.close());

const agoLabels = () => {
  const c = getStrings().chat;
  return {
    agoJustNow: c.agoJustNow,
    agoMinutes: c.agoMinutes,
    agoHours: c.agoHours,
    agoDays: c.agoDays,
    agoMonths: c.agoMonths,
    agoYears: c.agoYears,
  };
};

const now = Date.parse("2026-08-22T15:32:00Z");

test("duration drops leading zeros", () => {
  assert.equal(fmtDurationMs(5 * 60_000 + 13_000), "5m13s");
  assert.equal(fmtDurationMs(5 * 60_000 + 3_000), "5m3s");
  assert.equal(fmtDurationMs(5 * 60_000), "5m");
  assert.equal(fmtDurationMs(3_600_000 + 5 * 60_000), "1h5m");
  assert.equal(fmtDurationMs(3_600_000), "1h");
  assert.equal(fmtDurationMs((26 * 3600 + 5 * 60) * 1000), "1d2h");
  assert.equal(fmtDurationMs(45 * 86_400_000), "1mo15d");
  assert.equal(fmtDurationMs(400 * 86_400_000), "1y1mo");
});

test("sidebar idle labels use dsh's compact bilingual form without ago suffixes", () => {
  setState({ lang: "zh" });
  assert.equal(agoStr(now - 30_000, agoLabels(), now), "刚刚");
  assert.equal(agoStr(now - 7 * 60_000, agoLabels(), now), "5分钟");
  assert.equal(agoStr(now - 2 * 3_600_000, agoLabels(), now), "2小时");
  assert.equal(agoStr(now - 6 * 3_600_000, agoLabels(), now), "6小时");
  assert.equal(agoStr(now - 86_400_000, agoLabels(), now), "1天");
  assert.equal(agoStr(now - 2 * 86_400_000, agoLabels(), now), "2天");
  assert.equal(agoStr(now - 45 * 86_400_000, agoLabels(), now), "1个月");

  setState({ lang: "en" });
  assert.equal(agoStr(now - 4 * 60_000, agoLabels(), now), "now");
  assert.equal(agoStr(now - 7 * 60_000, agoLabels(), now), "5min");
  assert.equal(agoStr(now - 12 * 60_000, agoLabels(), now), "10min");
  assert.equal(agoStr(now - 2 * 3_600_000, agoLabels(), now), "2h");
  assert.equal(agoStr(now - 11 * 3_600_000, agoLabels(), now), "11h");
  assert.equal(agoStr(now - 86_400_000, agoLabels(), now), "1d");
  assert.equal(agoStr(now - 45 * 86_400_000, agoLabels(), now), "1mo");
  assert.equal(agoStr(now - 400 * 86_400_000, agoLabels(), now), "1y");
  for (const delta of [7 * 60_000, 2 * 3_600_000, 86_400_000, 45 * 86_400_000]) {
    assert.doesNotMatch(agoStr(now - delta, agoLabels(), now), /前|ago/);
  }
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

test("fmtSpan rolls over into hours, days, months and years", () => {
  const u = {
    y: (n) => `${n}年`, mo: (n) => `${n}个月`, d: (n) => `${n}天`,
    h: (n) => `${n}小时`, m: (n) => `${n}分`, s: (n) => `${n}秒`,
    sec: (s) => `${s}秒`, ms: (ms) => `${ms}毫秒`, join: "",
  };
  assert.equal(fmtSpan(450, u), "450毫秒");
  assert.equal(fmtSpan(49_800, u), "49.8秒");
  assert.equal(fmtSpan((22 * 60 + 35) * 1000, u), "22分35秒");
  assert.equal(fmtSpan(3_599_000, u), "59分59秒");
  assert.equal(fmtSpan(3_600_000, u), "1小时");
  assert.equal(fmtSpan((152 * 60 + 54) * 1000, u), "2小时32分");
  assert.equal(fmtSpan((27 * 3600 + 10 * 60) * 1000, u), "1天3小时");
  assert.equal(fmtSpan(30 * 86_400_000, u), "1个月");
  assert.equal(fmtSpan(70 * 86_400_000, u), "2个月10天");
  assert.equal(fmtSpan(800 * 86_400_000, u), "2年2个月");
});

test("fmtRunDuration formats whole-second turn elapsed time like dsh (7分31秒 / 7m 31s)", () => {
  const zhUnits = { h: "小时", m: "分", s: "秒" };
  const enUnits = { h: "h ", m: "m ", s: "s" };
  assert.equal(fmtRunDuration(1, zhUnits), "1秒");
  assert.equal(fmtRunDuration((7 * 60 + 31) * 1000, zhUnits), "7分31秒");
  assert.equal(fmtRunDuration((7 * 60 + 31) * 1000, enUnits), "7m 31s");
  assert.equal(fmtRunDuration((3600 + 2 * 60 + 5) * 1000, zhUnits), "1小时2分5秒");
  assert.equal(fmtRunDuration((3600 + 2 * 60 + 5) * 1000, enUnits), "1h 2m 5s");
});

test("updateQueuedMessageText edits a queued item in place and preserves queue order", () => {
  const queue = [
    { id: "q1", sessionId: "s1", text: "first query", files: [], ts: 1 },
    { id: "q2", sessionId: "s1", text: "second query", files: [], ts: 2 },
    { id: "q3", sessionId: "s1", text: "third query", files: [], ts: 3 },
  ];
  const edited = updateQueuedMessageText(queue, "q1", "first query edited");
  assert.deepEqual(
    edited.map((q) => [q.id, q.text]),
    [
      ["q1", "first query edited"],
      ["q2", "second query"],
      ["q3", "third query"],
    ],
  );
});
