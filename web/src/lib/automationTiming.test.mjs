import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false, appType: "custom", logLevel: "silent", server: { middlewareMode: true },
});
const T = await vite.ssrLoadModule("/src/lib/automationTiming.ts");
after(() => vite.close());

const cron = (expression, timezone = "Asia/Shanghai") => ({ kind: "cron", expression, timezone });

test("a stored rule opens in the preset that saves it back unchanged", () => {
  for (const [expr, freq] of [
    ["30 7 * * *", "daily"], ["0 9 * * 1-5", "weekdays"], ["15 8 * * 1,3,0", "weekly"],
    ["0 10 15 * *", "monthly"],
  ]) {
    const d = T.draftFromSpec(cron(expr));
    assert.equal(d.frequency, freq, expr);
    const out = T.scheduleFromDraft(d);
    assert.equal(out.timezone, "Asia/Shanghai");
    // Weekday lists are normalized (sorted, Sunday as 0); the rule is the same.
    const norm = (e) => e.split(" ").map((f, i) => (i === 4 ? f.split(",").sort().join(",") : f)).join(" ");
    assert.equal(norm(out.schedule), norm(expr), expr);
  }
});

test("anything the presets cannot reproduce opens as custom text", () => {
  for (const expr of ["*/5 * * * *", "0 9 * 1 *", "0 0 9 * * *", "0 9-17 * * *"]) {
    const d = T.draftFromSpec(cron(expr));
    assert.equal(d.frequency, "custom", expr);
    assert.equal(T.scheduleFromDraft(d).schedule, expr);
  }
});

test("intervals keep their unit", () => {
  assert.deepEqual(
    [T.draftFromSpec({ kind: "every", seconds: 7200 })].map((d) => [d.frequency, d.interval]), [["hourly", 2]]);
  assert.deepEqual(
    [T.draftFromSpec({ kind: "every", seconds: 900 })].map((d) => [d.frequency, d.interval]), [["minutely", 15]]);
  const odd = T.draftFromSpec({ kind: "every", seconds: 45 });
  assert.equal(odd.frequency, "custom");
  assert.equal(T.scheduleFromDraft(odd).schedule, "every 45s");
  assert.equal(T.scheduleFromDraft({ ...odd, frequency: "hourly", interval: 3 }).schedule, "every 10800s");
});

test("a one-shot time is sent as the instant the user picked", () => {
  const now = Date.parse("2026-11-11T10:00:00Z");
  const d = { ...T.defaultDraft(now), date: "2026-11-12", time: "08:30" };
  const out = T.scheduleFromDraft(d, now);
  assert.equal(Date.parse(out.schedule), new Date("2026-11-12T08:30:00").getTime());
  assert.match(out.schedule, /Z$/);
  assert.deepEqual(T.scheduleFromDraft({ ...d, date: "2020-01-01" }, now), { error: "notFuture" });
});

test("invalid drafts name the field to fix", () => {
  const d = T.defaultDraft();
  assert.deepEqual(T.scheduleFromDraft({ ...d, frequency: "weekly", weekdays: [] }), { error: "requiredWeekday" });
  assert.deepEqual(T.scheduleFromDraft({ ...d, frequency: "hourly", interval: 0 }), { error: "invalidInterval" });
  assert.deepEqual(T.scheduleFromDraft({ ...d, frequency: "daily", time: "25:00" }), { error: "invalidTime" });
  assert.deepEqual(T.scheduleFromDraft({ ...d, frequency: "custom", custom: " " }), { error: "invalidRule" });
});
