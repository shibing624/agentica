/** What the composer ring needs about one session. */
export type RingInput = {
  /** This session's last `/usage` answer, if any. */
  usage?: { window: number; context_tokens: number };
  /** The session has a log, so its real occupancy is on the server. */
  hasLog: boolean;
  /** Its first `/usage` request came back (or failed). */
  settled: boolean;
  localTokens: number;
  fallbackWindow: number;
};

/** The ring percentage, or `null` while a session with history waits for its
 *  first answer: drawing the previous chat's window or a stale local counter
 *  there is what made the figure change right after a switch. */
export function ringPercent({ usage, hasLog, settled, localTokens, fallbackWindow }: RingInput): number | null {
  if (usage) return Math.min(100, Math.round((usage.context_tokens / (usage.window || 128000)) * 100));
  if (hasLog && !settled) return null;
  return Math.min(100, Math.round((localTokens / (fallbackWindow || 128000)) * 100));
}
