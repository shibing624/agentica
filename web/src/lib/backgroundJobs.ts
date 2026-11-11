import type { BackgroundJob } from "../api";

export const isLiveJob = (job: BackgroundJob) => job.status === "running";

/** Live jobs oldest-first, then finished ones newest-first (tie-broken by start). */
export function orderJobs(jobs: BackgroundJob[]): BackgroundJob[] {
  return [...jobs].sort((a, b) => {
    if (isLiveJob(a) !== isLiveJob(b)) return isLiveJob(a) ? -1 : 1;
    if (isLiveJob(a)) return a.started_at_ms - b.started_at_ms;
    const diff = (b.finished_at_ms ?? b.started_at_ms) - (a.finished_at_ms ?? a.started_at_ms);
    return diff !== 0 ? diff : a.started_at_ms - b.started_at_ms;
  });
}

/** Elapsed time in at most two adjacent units: 8s, 3m 5s, 1h 2m. */
export function jobDuration(job: BackgroundJob, now: number): string {
  const total = Math.max(0, Math.floor(((job.finished_at_ms ?? now) - job.started_at_ms) / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${total % 60}s`;
  return `${total}s`;
}

/** Header badge count (dsh style): live count while work runs, else visible settled count. */
export function visibleJobBadgeCount(liveCount: number, settledCount: number): number {
  return liveCount > 0 ? liveCount : settledCount;
}

const BG_START_RE = /^Started background command #(\d+)\s+\(PID\s+(\d+),\s*id:\s*([A-Za-z0-9_-]+)\)\.(?:\s*\nLog:\s*(.+))?/m;

export type ParsedBackgroundStart = {
  num: number;
  pid: number;
  id: string;
  logPath: string;
};

/** Parse `execute(..., background=True)` launch receipt from tool result text. */
export function parseBackgroundStart(resultText: string): ParsedBackgroundStart | null {
  if (!resultText) return null;
  const m = BG_START_RE.exec(resultText.trim());
  if (!m) return null;
  return {
    num: Number(m[1]),
    pid: Number(m[2]),
    id: m[3],
    logPath: (m[4] || "").trim(),
  };
}
