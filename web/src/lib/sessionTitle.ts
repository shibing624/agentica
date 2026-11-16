/** Local draft titles that mean "not named yet". */
const UNTITLED = new Set(["", "Chat", "New Chat"]);

type TitleSession = {
  title?: string;
  named?: boolean;
};

/** A first request gives an unsent "New Chat" draft a readable local title.
 *  Names the user picked are authoritative; generated titles arrive later from
 *  SessionLog through the session catalog. */
export function titleForFirstMessage(
  sess: TitleSession,
  message: string,
): string {
  if (sess.named) return sess.title || "Chat";
  if (!UNTITLED.has((sess.title || "").trim())) return sess.title || "Chat";
  return (message || "Chat").trim().slice(0, 50) || "Chat";
}

/** Whether a server catalog title may replace a local draft/fallback title. */
export function shouldApplyDisplayTitle(
  sess: TitleSession,
  serverTitle: string,
): boolean {
  if (sess.named) return false;
  const title = (serverTitle || "").trim();
  if (!title || UNTITLED.has(title)) return false;
  return UNTITLED.has((sess.title || "").trim()) || title !== sess.title;
}