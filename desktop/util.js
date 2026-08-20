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
 * Which icon file this platform can actually use.
 *
 * The same waving cat as the browser tab and the SPA sidebar — the shell is a
 * window onto that UI, so a different mark here would just be a second brand to
 * keep in sync. Two files because one cannot serve both platforms:
 *
 * - Windows wants the multi-size ICO (16/32/48), and `nativeImage` decodes ICO
 *   **only** on Windows, so handing it the ICO anywhere else yields an empty
 *   image and a default-looking window.
 * - Everywhere else takes the 256² transparent PNG. macOS ignores
 *   `BrowserWindow#icon` outright and reads the dock icon from the bundle, so
 *   there it is only good for an unpackaged run (see `installIcon`).
 *
 * Paths point at the repo's own assets rather than copies under `desktop/`:
 * a duplicated binary is a second cat that drifts, and electron-builder can
 * reference these same paths when packaging lands.
 */
function iconPath(platform, repoRoot) {
  return platform === "win32"
    ? path.join(repoRoot, "docs", "assets", "favicon.ico")
    : path.join(repoRoot, "web", "src", "assets", "cat.png");
}

module.exports = {
  MAX_GATEWAY_RESTARTS,
  HEALTHY_AFTER_MS,
  restartDelayMs,
  isAppUrl,
  parsePort,
  iconPath,
};
