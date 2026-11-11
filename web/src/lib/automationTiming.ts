/**
 * The Automations editor's timing model: dsh's guided presets, mapped to the
 * three schedule forms the gateway stores (cron expression, `every Ns`
 * interval, one-shot ISO time).
 *
 * A stored schedule opens in the preset that reproduces it exactly; anything
 * else opens as "custom" with its cron expression as text, so opening a job
 * and saving it unchanged never rewrites its rule.
 */

export type Frequency = "once" | "minutely" | "hourly" | "daily" | "weekdays" | "weekly" | "monthly" | "custom";

export const FREQUENCIES: Frequency[] = [
  "weekly", "weekdays", "daily", "hourly", "minutely", "monthly", "once", "custom",
];

export type ScheduleSpec =
  | { kind: "cron"; expression: string; timezone: string }
  | { kind: "every"; seconds: number }
  | { kind: "at"; at_ms: number }
  | { kind: "unknown" };

export type TimingDraft = {
  frequency: Frequency;
  /** Repeat interval: minutes / hours for the interval presets. */
  interval: number;
  /** Local `YYYY-MM-DD` for "once". */
  date: string;
  /** Local `HH:MM`. */
  time: string;
  /** ISO weekdays, Monday = 1 … Sunday = 7. */
  weekdays: number[];
  monthDay: number;
  timezone: string;
  /** Cron expression for "custom". */
  custom: string;
};

const pad = (n: number) => String(n).padStart(2, "0");

export function systemZone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
}

function localParts(ms: number): { date: string; time: string; weekday: number; day: number } {
  const d = new Date(ms);
  return {
    date: `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`,
    time: `${pad(d.getHours())}:${pad(d.getMinutes())}`,
    weekday: ((d.getDay() + 6) % 7) + 1,
    day: d.getDate(),
  };
}

/** A new task: once, an hour from now, in the browser's zone (dsh's default). */
export function defaultDraft(now = Date.now()): TimingDraft {
  const p = localParts(now + 3600_000);
  return {
    frequency: "once", interval: 1, date: p.date, time: p.time,
    weekdays: [p.weekday], monthDay: p.day, timezone: systemZone(), custom: "",
  };
}

const isNum = (s: string) => /^\d+$/.test(s);
/** Cron's weekday (0/7 = Sunday) as ISO (7 = Sunday). */
const isoDay = (n: number) => (n === 0 ? 7 : n);

/** The draft an existing job's schedule opens as. */
export function draftFromSpec(spec: ScheduleSpec, now = Date.now()): TimingDraft {
  const base = defaultDraft(now);
  if (spec.kind === "at") {
    const p = localParts(spec.at_ms);
    return { ...base, frequency: "once", date: p.date, time: p.time };
  }
  if (spec.kind === "every") {
    if (spec.seconds % 3600 === 0) return { ...base, frequency: "hourly", interval: spec.seconds / 3600 };
    if (spec.seconds % 60 === 0) return { ...base, frequency: "minutely", interval: spec.seconds / 60 };
    return { ...base, frequency: "custom", custom: `every ${spec.seconds}s` };
  }
  if (spec.kind !== "cron") return { ...base, frequency: "custom" };
  const expr = spec.expression.trim();
  const tz = spec.timezone || base.timezone;
  const custom = { ...base, frequency: "custom" as const, custom: expr, timezone: tz };
  const parts = expr.split(/\s+/);
  if (parts.length !== 5) return custom;
  const [mi, h, dom, mon, dow] = parts;
  if (mon !== "*") return custom;
  if (!isNum(mi) || !isNum(h)) return custom;
  const time = `${pad(Number(h))}:${pad(Number(mi))}`;
  if (dom === "*" && dow === "*") return { ...base, frequency: "daily", time, timezone: tz };
  if (dom === "*" && dow === "1-5") return { ...base, frequency: "weekdays", time, weekdays: [1, 2, 3, 4, 5], timezone: tz };
  if (dom === "*" && /^\d(,\d)*$/.test(dow)) {
    const days = [...new Set(dow.split(",").map((d) => isoDay(Number(d))))].sort();
    if (days.every((d) => d >= 1 && d <= 7)) return { ...base, frequency: "weekly", time, weekdays: days, timezone: tz };
  }
  if (dow === "*" && isNum(dom) && Number(dom) >= 1 && Number(dom) <= 31) {
    return { ...base, frequency: "monthly", time, monthDay: Number(dom), timezone: tz };
  }
  return custom;
}

/** What the gateway stores: `{ schedule, timezone }`, or an error key. */
export function scheduleFromDraft(d: TimingDraft, now = Date.now()):
  { schedule: string; timezone: string } | { error: DraftIssue } {
  const [h, mi] = d.time.split(":").map(Number);
  const okTime = Number.isInteger(h) && Number.isInteger(mi) && h >= 0 && h < 24 && mi >= 0 && mi < 60;
  const tz = d.timezone || systemZone();
  switch (d.frequency) {
    case "once": {
      if (!okTime || !/^\d{4}-\d{2}-\d{2}$/.test(d.date)) return { error: "invalidTime" };
      const at = new Date(`${d.date}T${d.time}:00`).getTime();
      if (!(at > now)) return { error: "notFuture" };
      // An absolute instant: the gateway must not reread it in its own zone.
      return { schedule: new Date(at).toISOString().replace(/\.\d{3}Z$/, "Z"), timezone: tz };
    }
    case "minutely":
    case "hourly": {
      if (!Number.isInteger(d.interval) || d.interval < 1) return { error: "invalidInterval" };
      return { schedule: `every ${d.interval * (d.frequency === "hourly" ? 3600 : 60)}s`, timezone: tz };
    }
    case "daily":
      if (!okTime) return { error: "invalidTime" };
      return { schedule: `${mi} ${h} * * *`, timezone: tz };
    case "weekdays":
      if (!okTime) return { error: "invalidTime" };
      return { schedule: `${mi} ${h} * * 1-5`, timezone: tz };
    case "weekly": {
      if (!okTime) return { error: "invalidTime" };
      if (!d.weekdays.length) return { error: "requiredWeekday" };
      const days = [...d.weekdays].sort().map((x) => (x === 7 ? 0 : x)).join(",");
      return { schedule: `${mi} ${h} * * ${days}`, timezone: tz };
    }
    case "monthly":
      if (!okTime) return { error: "invalidTime" };
      if (!Number.isInteger(d.monthDay) || d.monthDay < 1 || d.monthDay > 31) return { error: "invalidMonthDay" };
      return { schedule: `${mi} ${h} ${d.monthDay} * *`, timezone: tz };
    case "custom":
      if (!d.custom.trim()) return { error: "invalidRule" };
      return { schedule: d.custom.trim(), timezone: tz };
  }
}

export type DraftIssue = "invalidTime" | "notFuture" | "invalidInterval" | "requiredWeekday" | "invalidMonthDay" | "invalidRule";

/** Relative distance to a time, as dsh's row summary shows it. */
export function relativeParts(target: number, now: number): { unit: "s" | "m" | "h" | "d"; value: number; future: boolean } {
  const diff = target - now;
  const abs = Math.abs(diff) / 1000;
  const [unit, size] = abs >= 86400 ? ["d", 86400] as const : abs >= 3600 ? ["h", 3600] as const
    : abs >= 60 ? ["m", 60] as const : ["s", 1] as const;
  return { unit, value: Math.max(1, Math.ceil(abs / size)), future: diff >= 0 };
}
