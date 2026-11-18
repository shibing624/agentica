/** What the right panel's browser tab can load: http(s) pages only. */

const SCHEME = /^[a-z][a-z0-9+.-]*:\/\//i;
const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]);

/**
 * The address bar's text as a URL to load, or `null` when it is not an
 * address. A scheme-less entry gets one by where it points: `http://` for this
 * machine (a dev server), `https://` for everything else. Nothing is guessed
 * into a search query — an entry that is not an address is refused.
 */
export function normalizeBrowserUrl(input: string): string | null {
  const text = input.trim();
  if (!text || /\s/.test(text)) return null;
  const candidate = SCHEME.test(text) ? text : `https://${text}`;
  let url: URL;
  try {
    url = new URL(candidate);
  } catch {
    return null;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  if (!SCHEME.test(text) && LOOPBACK.has(url.hostname)) url.protocol = "http:";
  if (!url.hostname.includes(".") && !url.hostname.includes(":") && !LOOPBACK.has(url.hostname)) return null;
  return url.toString();
}

/** Storage partition of the browser tab's pages. The shell refuses any other one
 *  (`desktop/util.js`), so a page never shares cookies with the app itself. */
export const BROWSER_PARTITION = "persist:agentica-browser";

/** The Electron shell marks its user agent; only there a `<webview>` exists. */
export function inDesktopShell(): boolean {
  return /AgenticaDesktop/.test(navigator.userAgent);
}

/** Tab label: the page's host, since its title arrives later (or never). */
export function browserTabLabel(url: string, title = ""): string {
  if (title.trim()) return title.trim();
  try {
    return new URL(url).host || url;
  } catch {
    return url;
  }
}
