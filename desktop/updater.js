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
const { URLS, compareSemver } = require("./menu");

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

async function checkForUpdatesInteractive({ app, dialog, shell, lang = "zh" }) {
  const zh = /^zh/i.test(String(lang || ""));
  const current = app.getVersion();
  if (app.isPackaged && canSelfUpdate() && process.env.AGENTICA_DESKTOP_NO_UPDATE !== "1") {
    try {
      const res = await autoUpdater.checkForUpdates();
      const latest = res && res.updateInfo ? String(res.updateInfo.version || "") : "";
      if (latest && compareSemver(current, latest) === -1) {
        const { response } = await dialog.showMessageBox({
          type: "info",
          message: zh ? `发现新版本 v${latest}` : `Update available: v${latest}`,
          detail: zh
            ? `当前版本为 v${current}，后台正在准备更新。你也可以打开发布页查看更新说明。`
            : `You are running v${current}. The update is being fetched in the background.`,
          buttons: [zh ? "查看发布页" : "View Releases", zh ? "好" : "OK"],
          defaultId: 0,
        });
        if (response === 0) void shell.openExternal(URLS.releases);
        return;
      }
      await dialog.showMessageBox({
        type: "info",
        message: zh ? "当前已是最新版本" : "Agentica is up to date",
        detail: zh ? `当前版本：v${current}` : `Current version: v${current}`,
      });
      return;
    } catch (err) {
      await dialog.showMessageBox({
        type: "warning",
        message: zh ? "检查更新失败" : "Update check failed",
        detail: err instanceof Error ? err.message : String(err),
      });
      return;
    }
  }

  try {
    const resp = await fetch(URLS.latestReleaseApi, {
      headers: { Accept: "application/vnd.github+json", "User-Agent": "agentica-desktop" },
      signal: AbortSignal.timeout(6000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const tag = String(data.tag_name || "").replace(/^v/i, "");
    const cmp = tag ? compareSemver(current, tag) : null;
    if (cmp === -1) {
      const { response } = await dialog.showMessageBox({
        type: "info",
        message: zh ? `发现新版本 v${tag}` : `New version available: v${tag}`,
        detail: zh
          ? `当前版本：v${current}\n最新发布：v${tag}\n\n请前往 Releases 下载新版，或运行 brew upgrade --cask shibing624/tap/agentica 更新。`
          : `Current: v${current}\nLatest: v${tag}\n\nDownload from Releases or run \`brew upgrade --cask shibing624/tap/agentica\`.`,
        buttons: [zh ? "前往下载" : "Open Releases", zh ? "关闭" : "Close"],
        defaultId: 0,
      });
      if (response === 0) void shell.openExternal(data.html_url || URLS.releases);
      return;
    }
    await dialog.showMessageBox({
      type: "info",
      message: zh ? "当前已是最新版本" : "Agentica is up to date",
      detail: zh ? `当前版本：v${current}` : `Current version: v${current}`,
    });
  } catch (err) {
    const { response } = await dialog.showMessageBox({
      type: "warning",
      message: zh ? "检查更新失败" : "Could not check for updates",
      detail: zh
        ? `当前版本：v${current}\n错误：${err instanceof Error ? err.message : String(err)}`
        : `Current: v${current}\nError: ${err instanceof Error ? err.message : String(err)}`,
      buttons: [zh ? "打开发布页" : "Open Releases", zh ? "关闭" : "Close"],
      defaultId: 1,
    });
    if (response === 0) void shell.openExternal(URLS.releases);
  }
}

module.exports = { canSelfUpdate, startUpdates, checkForUpdatesInteractive };
