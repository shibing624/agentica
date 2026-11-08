/** When the last automatic reload happened (ms since epoch), per tab. */
const RELOAD_KEY = "ag_chunk_reload";

/** One automatic reload per this long. A reload that does not cure the error
 *  (the new build is also missing the file) must end in a message, not a loop. */
export const RELOAD_WINDOW_MS = 60_000;

/**
 * A lazy page failing to load: the gateway was upgraded and this tab still holds
 * the old build's file names, so the chunk it asks for no longer exists.
 */
export function isStaleBundleError(error: unknown): boolean {
  const text = String((error as { message?: string } | null)?.message ?? error ?? "");
  return /Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed|Unable to preload CSS/i.test(text);
}

/**
 * True when this stale-bundle error should reload the page: it is one, and the
 * tab has not already reloaded for it within the last minute.
 *
 * Judged by time, not by "did the page render afterwards": a lazy page that is
 * still loading renders its siblings immediately, so a render-based signal
 * clears the guard before the failure arrives and the page reloads forever.
 */
export function shouldAutoReload(
  error: unknown,
  storage: Pick<Storage, "getItem" | "setItem"> = sessionStorage,
  now: number = Date.now(),
): boolean {
  if (!isStaleBundleError(error)) return false;
  const last = Number(storage.getItem(RELOAD_KEY));
  if (last && now - last < RELOAD_WINDOW_MS) return false;
  storage.setItem(RELOAD_KEY, String(now));
  return true;
}
