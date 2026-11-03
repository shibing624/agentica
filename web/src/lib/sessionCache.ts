/** What the browser keeps of a session across reloads: only the counters the
 *  server log does not return. Transcript, title, timestamps and work_dir come
 *  back from `GET /api/sessions` and `hydrateSession` on every load and open,
 *  so a cached copy is dead weight — and it is what filled the localStorage
 *  quota (thousands of CLI sessions plus every opened transcript), after which
 *  `setItem` threw on the send path and messages silently stopped going out.
 *  Sessions this browser never ran a turn in have nothing to keep. */

const STAT_KEYS = [
  "tokIn", "tokOut", "tokTotal", "requests", "totalTime",
  "lastInputTokens", "contextTokens", "costUsd",
] as const;

type StatKey = (typeof STAT_KEYS)[number];
export type SessionStats = Record<StatKey, number>;

export function sessionStatsCache(
  sessions: Record<string, SessionStats>,
): Record<string, SessionStats> {
  const out: Record<string, SessionStats> = {};
  for (const [id, sess] of Object.entries(sessions)) {
    if (!sess.requests) continue;
    const stats = {} as SessionStats;
    for (const key of STAT_KEYS) stats[key] = sess[key];
    out[id] = stats;
  }
  return out;
}
