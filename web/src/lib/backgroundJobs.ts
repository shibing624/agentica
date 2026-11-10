import type { BackgroundJob } from "../api";

export const isLiveJob = (job: BackgroundJob) => job.status === "running";

/** Live jobs oldest-first, then finished ones newest-first. */
export function orderJobs(jobs: BackgroundJob[]): BackgroundJob[] {
  return [...jobs].sort((a, b) => {
    if (isLiveJob(a) !== isLiveJob(b)) return isLiveJob(a) ? -1 : 1;
    if (isLiveJob(a)) return a.started_at_ms - b.started_at_ms;
    return (b.finished_at_ms ?? b.started_at_ms) - (a.finished_at_ms ?? a.started_at_ms);
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
