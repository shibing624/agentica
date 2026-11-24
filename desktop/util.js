"use strict";
/**
 * Pure helpers for the shell. No Electron imports, so they can be run under
 * plain `node` — which is the only way any of this gets tested without a
 * display.
 */

const path = require("node:path");

/** Max automatic gateway restarts before giving up with a dialog. */
const MAX_GATEWAY_RESTARTS = 3;

/** Restart backoff: 1s, 2s, 4s, capped at 8s (attempt is 0-based). */
function restartDelayMs(attempt) {
  return Math.min(1000 * 2 ** attempt, 8000);
}

/**
 * A run that stayed up this long is healthy: the restart budget resets, so a
 * crash days later starts a fresh 1s/2s/4s ladder instead of immediately
 * hitting the cap left over from a bad start hours ago.
 */
const HEALTHY_AFTER_MS = 60_000;

/**
 * Whether a navigation target stays inside the app window.
 *
 * Only this instance's own origin qualifies. Anything else — including another
 * gateway on another port — is the web and belongs in the user's own browser,
 * where their extensions and password manager live. The previous rule was "any
 * loopback host", which quietly made every dev server on the machine part of
 * the app.
 */
function isAppUrl(url, origin) {
  if (!origin) return false;
  try {
    return new URL(url).origin === new URL(origin).origin;
  } catch {
    return false;
  }
}

/** Reads a port out of the shell's own memory file; null when absent/invalid. */
function parsePort(content) {
  const m = /^(\d{1,5})\s*$/.exec(String(content).trim());
  if (!m) return null;
  const port = Number(m[1]);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? port : null;
}

/**
 * The icon file, for runs that need one from disk.
 *
 * The same waving cat as the browser tab and the SPA sidebar — the shell is a
 * window onto that UI, so a different mark here would be a second brand to keep
 * in sync. `make_icon.py` rebuilds the PNGs from `docs/assets/logo.png`.
 * `nativeImage` decodes PNG everywhere, whereas the ICO the favicon ships as
 * decodes **only** on Windows and yields an empty image — which looks identical
 * to having configured no icon at all — anywhere else.
 *
 * Unpackaged macOS uses `icon-dock.png`. `app.dock.setIcon` draws the PNG as-is,
 * so the squircle and its transparent margin are in the file. The installed
 * bundle uses `icon-mac.png`: the same badge flattened opaque to the edges.
 * The dock file's transparent margin is what gets nested inside the system plate.
 */
function iconPath(platform = process.platform) {
  return path.join(__dirname, "build", platform === "darwin" ? "icon-dock.png" : "icon.png");
}

/**
 * The built-in browser tab's pages live in their own storage partition: a page
 * opened there must never see the cookies of the app's own window (the gateway
 * session is a cookie on this very host).
 */
const BROWSER_PARTITION = "persist:agentica-browser";

/** Only web pages open in the browser tab; no file:, no custom schemes. */
function isBrowsableUrl(url) {
  try {
    const { protocol } = new URL(url);
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Decide whether a `<webview>` the page asked for may attach, and strip it of
 * everything that is not a plain web page: no preload, no Node, sandboxed.
 * Returns false to refuse the attach. The storage partition is the page's own
 * attribute and is only *checked* here: Electron has already resolved it by
 * the time this runs, so rewriting `params.partition` changes nothing (a guest
 * "moved" that way still read the app window's cookies).
 */
function hardenWebview(webPreferences, params) {
  if (params.partition !== BROWSER_PARTITION) return false;
  if (params.src && !isBrowsableUrl(params.src)) return false;
  delete webPreferences.preload;
  delete webPreferences.preloadURL;
  webPreferences.nodeIntegration = false;
  webPreferences.nodeIntegrationInSubFrames = false;
  webPreferences.contextIsolation = true;
  webPreferences.sandbox = true;
  webPreferences.webSecurity = true;
  return true;
}

module.exports = {
  BROWSER_PARTITION,
  hardenWebview,
  isBrowsableUrl,
  MAX_GATEWAY_RESTARTS,
  HEALTHY_AFTER_MS,
  restartDelayMs,
  isAppUrl,
  parsePort,
  iconPath,
};
