/** What the composer ring needs about one session. */
export type RingInput = {
  /** This session's last `/usage` answer, if any. */
  usage?: { window: number; context_tokens: number };
  /** The session has a log, so its real occupancy is on the server. */
  hasLog: boolean;
  /** Its first `/usage` request came back (or failed). */
  settled: boolean;
  localTokens: number;
  /** This session's own last measured window. Not another chat's. */
  localWindow?: number;
  fallbackWindow: number;
};

/** The ring percentage, or `null` while a session with history waits for its
 *  first answer: drawing the previous chat's window or a stale local counter
 *  there is what made the figure change right after a switch. A window this
 *  session already measured is not that — the button can show it on the first
 *  frame and let the refresh replace the number in place. */
export function ringPercent({ usage, hasLog, settled, localTokens, localWindow, fallbackWindow }: RingInput): number | null {
  if (!usage && hasLog && !settled) {
    if (localWindow && localWindow > 0) {
      return Math.min(100, Math.round((localTokens / localWindow) * 100));
    }
    return null;
  }
  // One denominator for the ring and the popover: the chat's own window, else
  // the server's, else the last-resort estimate. `usage.window === 0` means
  // "unknown", not "divide by zero" — the popover already resolves it that
  // way, and two different denominators showed the same session twice.
  const window = usage?.window || fallbackWindow || 128000;
  const tokens = usage ? usage.context_tokens : localTokens;
  return Math.min(100, Math.round((tokens / window) * 100));
}
