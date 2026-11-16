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

/** When the latest request a person wrote began. Automated turns (a job's own
 *  report, another session's message) and mid-run steers are not new requests,
 *  so a job's report arriving does not start a fresh scope. */
export function currentRequestStartMs(
  msgs: ReadonlyArray<{ role: string; ts?: number; automated?: boolean; steer?: boolean }>,
): number | undefined {
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m.role === "user" && !m.automated && !m.steer) return m.ts;
  }
  return undefined;
}

/** Finished jobs belong to the request that started them: once the next request
 *  begins, they leave the header. Running jobs are never dropped, whichever
 *  request started them, so they stay visible and stoppable. */
export function settledInRequest<T extends { started_at_ms: number }>(
  settled: T[],
  requestStartMs: number | undefined,
): T[] {
  if (requestStartMs === undefined) return settled;
  return settled.filter((j) => j.started_at_ms >= requestStartMs);
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
