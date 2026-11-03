"use strict";
/**
 * Background updates. The shell, the bundled Python and agentica ship as one
 * versioned unit, so an update replaces all three at once.
 *
 * macOS only accepts an update for a signed app (Squirrel.Mac verifies the
 * signature), so an unsigned build skips the check rather than downloading an
 * update it can never install.
 */

const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { autoUpdater } = require("electron-updater");

function appBundle() {
  return path.resolve(process.execPath, "..", "..", "..");
}

function canSelfUpdate() {
  if (process.platform !== "darwin") return true;
  try {
    execFileSync("codesign", ["--verify", "--deep", appBundle()], { stdio: "ignore", timeout: 15_000 });
    return true;
  } catch {
    return false;
  }
}

function startUpdates(app, log) {
  if (!app.isPackaged || process.env.AGENTICA_DESKTOP_NO_UPDATE === "1") return;
  if (!canSelfUpdate()) {
    log("auto-update skipped: this build is not code-signed");
    return;
  }
  autoUpdater.logger = { info: log, warn: log, error: log, debug: () => {} };
  autoUpdater.on("error", (err) => log(`update check failed: ${err.message}`));
  void autoUpdater.checkForUpdatesAndNotify();
}

module.exports = { canSelfUpdate, startUpdates };
