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
 * in sync. It is the 1024² master electron-builder packages from
 * (`make_icon.py` rebuilds it from `docs/assets/logo.png`), and one file covers
 * every platform: `nativeImage` decodes PNG everywhere, whereas the ICO the
 * favicon ships as decodes **only** on Windows and yields an empty image — which
 * looks identical to having configured no icon at all — anywhere else.
 */
function iconPath() {
  return path.join(__dirname, "build", "icon.png");
}

module.exports = {
  MAX_GATEWAY_RESTARTS,
  HEALTHY_AFTER_MS,
  restartDelayMs,
  isAppUrl,
  parsePort,
  iconPath,
};
