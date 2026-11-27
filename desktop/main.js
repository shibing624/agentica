"use strict";
/**
 * The Electron shell.
 *
 * This is the least updatable code in the product: a change here reaches a user
 * only through a new installer, while a change in the SPA or the gateway
 * reaches them on the next `pip install -U`. So it carries mechanism only —
 * process lifecycle, one window, and the native menu items a web page cannot
 * provide. Anything that *could* be delivered over HTTP must be.
 *
 * There is no preload, no IPC and no Node in the renderer. The window is a
 * plain browser pointed at the same SPA a browser would load, so a feature can
 * never land here instead of in the product: if it is not reachable over HTTP,
 * the desktop app cannot do it either.
 *
 * One exception, kept to mechanism: the SPA's built-in browser tab needs a
 * `<webview>` (an iframe cannot show most sites). The shell only allows it and
 * locks it down — see `hardenWebview`; the tab itself is SPA code.
 *
 * What it does:
 *
 *   1. one instance per machine,
 *   2. attach to a running gateway or start one (on last launch's port, so the
 *      window's origin — and its localStorage — survives a restart),
 *   3. open a window on it, already signed in,
 *   4. restart the gateway if it dies, with a backoff and a cap,
 *   5. stop what it started on the way out, gracefully.
 */

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFile } = require("node:child_process");
const { promisify } = require("node:util");
const { app, BrowserWindow, clipboard, dialog, Menu, nativeImage, nativeTheme, Notification, screen, session, shell } = require("electron");
const { GatewayProcess, health } = require("./gateway");
const { bootstrapBlockedPage, resolveExistingCommand, supportRoot } = require("./runtime");
const { inspectCommand, installCommand, uninstallCommand } = require("./cli-shim");
const { buildDockMenuTemplate } = require("./dock-menu");
const {
  URLS,
  buildAppMenuTemplate,
  formatDiagnosticsReport,
  formatResourceUsageReport,
} = require("./menu");
const { canSelfUpdate, checkForUpdatesInteractive, startUpdates } = require("./updater");
const { keepVisiblePosition, readWindowState, writeWindowState } = require("./window-state");
const { desktopStatePaths, waitForChat } = require("./smoke");
const {
  BROWSER_PARTITION, HEALTHY_AFTER_MS, MAX_GATEWAY_RESTARTS, formatTurnNotification, hardenWebview, iconPath, isAppUrl, isBrowsableUrl, restartDelayMs,
} = require("./util");

const log = (msg) => console.log(`[desktop] ${msg}`);
const smoke = process.env.AGENTICA_DESKTOP_SMOKE === "1";
const isolatedRoot = process.env.AGENTICA_DESKTOP_STATE_DIR;
if (isolatedRoot) {
  const state = desktopStatePaths(isolatedRoot);
  fs.mkdirSync(state.profile, { recursive: true });
  app.setPath("userData", state.profile);
  process.env.AGENTICA_HOME = state.home;
  process.env.AGENTICA_CACHE_DIR = path.join(state.home, "cache");
  process.env.AGENTICA_DESKTOP_RUNTIME = state.runtime;
}

/** The app icon, or null when the file is missing or unreadable. Loaded once:
 *  `nativeImage` decoding is disk I/O, and the window and the dock want the
 *  same image. */
const icon = (() => {
  // A packaged build needs none: the installer put the icon where the platform
  // looks for it (the .app bundle, the exe's resources, the .desktop entry).
  // Reading it here would also mean reading out of app.asar, which nativeImage
  // is not guaranteed to do — and the failure looks like a missing file.
  if (app.isPackaged) return null;
  const file = iconPath();
  const image = nativeImage.createFromPath(file);
  if (image.isEmpty()) {
    log(`no app icon at ${file} — using the default`);
    return null;
  }
  return image;
})();

let gateway = null;
let win = null;
/** The origin the window is on; null until boot resolves. */
let origin = null;
let quitting = false;
let stopping = null;
let restarts = 0;

function fatal(context, err) {
  if (smoke) {
    // First line only: the gateway's own explanation often continues with a
    // stderr tail, and that tail can contain the local token.
    const detail = (err instanceof Error ? err.message : String(err || "")).split(/\r?\n/, 1)[0].slice(0, 300);
    process.stdout.write(`DESKTOP-SMOKE-RESULT ${JSON.stringify({ error: context, detail, version: app.getVersion() })}\n`);
    process.exitCode = 1;
    app.quit();
    return;
  }
  const detail = err instanceof Error ? (err.stack || err.message) : String(err);
  log(`${context}: ${detail}`);
  dialog.showErrorBox("Agentica", `${context}\n\n${detail}`);
  app.exit(1);
}

function createWindow() {
  const stateFile = path.join(app.getPath("userData"), "window-state.json");
  const state = keepVisiblePosition(readWindowState(stateFile), screen.getAllDisplays());
  const { maximized, ...bounds } = state;
  const mac = process.platform === "darwin";
  win = new BrowserWindow({
    ...bounds,
    minWidth: 720,
    minHeight: 480,
    title: "Agentica",
    // macOS: no title bar, traffic lights inset into the sidebar, and a
    // translucent sidebar over the system material. The page paints the chat
    // column itself, so the window background has to be transparent there.
    ...(mac
      ? {
          titleBarStyle: "hiddenInset",
          trafficLightPosition: { x: 16, y: 18 },
          vibrancy: "sidebar",
          visualEffectState: "active",
          backgroundColor: "#00000000",
        }
      : { backgroundColor: nativeTheme.shouldUseDarkColors ? "#0a0a0a" : "#ffffff" }),
    show: false,
    ...(icon ? { icon } : {}),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      // For the SPA's browser tab only; `will-attach-webview` below refuses
      // anything that is not a locked-down web page.
      webviewTag: true,
    },
  });
  win.webContents.on("will-attach-webview", (event, webPreferences, params) => {
    if (!hardenWebview(webPreferences, params)) event.preventDefault();
  });
  // The SPA lays out differently inside this shell (drag regions, room for the
  // traffic lights), and the user agent is the one signal it can read without a
  // preload script.
  win.webContents.setUserAgent(`${win.webContents.getUserAgent()} AgenticaDesktop/${app.getVersion()}`);
  if (maximized) win.maximize();

  win.once("ready-to-show", () => win.show());
  // macOS full screen hides the traffic lights, so the page stops leaving room
  // for them. There is no preload; the main process marks <html> itself, and
  // again after each load so a reload in full screen keeps it.
  const markFullScreen = () => {
    if (!win || win.isDestroyed()) return;
    const on = win.isFullScreen();
    void win.webContents
      .executeJavaScript(
        `document.documentElement.toggleAttribute("data-fullscreen", ${on ? "true" : "false"})`,
      )
      .catch(() => {});
  };
  win.on("enter-full-screen", markFullScreen);
  win.on("leave-full-screen", markFullScreen);
  win.webContents.on("did-finish-load", () => {
    // A reload starts a fresh page: whatever it was running is gone.
    lastTurnRunning = false;
    lastTurnSession = "";
    lastTurnTitle = "";
    markFullScreen();
    void refreshDockMenu(true);
  });
  win.webContents.on("page-title-updated", () => {
    void onTitleChanged();
    void refreshDockMenu(true);
  });
  win.on("focus", () => void refreshDockMenu(true));
  win.on("blur", () => void refreshDockMenu(false));
  win.on("close", () => {
    try {
      writeWindowState(stateFile, { ...win.getNormalBounds(), maximized: win.isMaximized() });
    } catch (err) {
      log(`could not save window state: ${err.message}`);
    }
  });
  win.on("closed", () => {
    win = null;
    void refreshDockMenu(false);
  });

  // Only this instance's own origin is the app. Everything else — including
  // another gateway on another port — is the web, and belongs in the user's own
  // browser where their extensions and password manager live.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (isAppUrl(url, origin)) return { action: "allow" };
    void shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (!isAppUrl(url, origin)) {
      event.preventDefault();
      void shell.openExternal(url);
    }
  });

  // A renderer crash leaves a white rectangle with no way back: there is no
  // reload button in a frameless-menu app, and the user's only recourse would
  // be quitting. Reloading is always right — the state is on the server.
  win.webContents.on("render-process-gone", () => win && win.webContents.reload());

  armSmokeProbe(win);
  return win;
}

/**
 * Sign the window in before it loads anything.
 *
 * The gateway accepts `?token=…` on a page load too, but that puts the
 * credential in a URL the renderer can read back out of `location.search`.
 * Redeeming it here means the page is loaded already signed in and never sees
 * the value.
 *
 * The cookie the gateway checks is a *session*, not the token — so this
 * redeems the token against a gated route (any one will do; `/api/status` is
 * the cheapest) and copies the session it hands back into the window's cookie
 * jar. That round trip doubles as the check that the token in runtime.json
 * still belongs to the gateway now answering on that port.
 */
async function installSession(record) {
  if (!record.token) return;
  let cookie = null;
  try {
    const resp = await fetch(`${record.url}/api/status?token=${encodeURIComponent(record.token)}`);
    cookie = (resp.headers.getSetCookie() || [])
      .map((line) => /(?:^|;\s*)agentica_session=([^;]+)/.exec(line))
      .find(Boolean);
  } catch (err) {
    log(`could not redeem the token: ${err.message}`);
  }
  if (!cookie) {
    // Attach mode against a gateway whose token has since changed, or one
    // running with GATEWAY_AUTH=false. Either way the window still loads: it
    // lands on the login page, or straight in.
    log("no session was issued; the window will sign itself in");
    return;
  }
  await session.defaultSession.cookies.set({
    url: record.url,
    name: "agentica_session",
    value: cookie[1],
    httpOnly: true,
    sameSite: "lax",
  });
}

/** Start (or restart) the gateway and point the window at it. */
async function startGatewayAndWindow() {
  const record = await gateway.ensure();
  origin = record.url;
  await installSession(record);

  // A run that survives a minute is healthy: reset the budget so a crash days
  // later gets a fresh ladder rather than the cap left over from this morning.
  const healthy = setTimeout(() => {
    restarts = 0;
  }, HEALTHY_AFTER_MS);
  if (gateway.child) {
    gateway.child.once("exit", (code, signal) => {
      clearTimeout(healthy);
      void handleGatewayExit(code ?? signal);
    });
  }

  if (win === null) createWindow();
  await win.loadURL(`${record.url}/chat`);
  void refreshDockMenu(true);
}

/**
 * The gateway died on its own. Restart it, up to a point.
 *
 * Without this the window sits on a dead port showing whatever it had rendered,
 * and every click fails with no explanation — the worst possible outcome of a
 * crash that a 1s restart would have hidden entirely.
 */
async function handleGatewayExit(code) {
  if (quitting) return;
  if (restarts >= MAX_GATEWAY_RESTARTS) {
    fatal(
      `The gateway keeps exiting (last code ${code}); giving up on restarting it.`,
      "Run `agentica-gateway` in a terminal to see the actual error."
    );
    return;
  }
  const wait = restartDelayMs(restarts);
  restarts += 1;
  log(`gateway exited (${code}); restarting in ${wait}ms`);
  await new Promise((r) => setTimeout(r, wait));
  if (quitting) return;
  try {
    await startGatewayAndWindow();
  } catch (err) {
    fatal("Could not restart the gateway.", err);
  }
}

function showSetupWindow() {
  const w = new BrowserWindow({
    width: 520,
    height: 220,
    title: "Agentica",
    resizable: false,
    minimizable: false,
    maximizable: false,
    backgroundColor: "#ffffff",
    show: true,
    ...(icon ? { icon } : {}),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });
  const html = `<!doctype html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:36px 40px;font:15px/1.45 system-ui,-apple-system,sans-serif;color:#1a1a1a">
  <h1 style="font-size:18px;margin:0 0 10px">Installing Agentica</h1>
  <p style="margin:0;color:#444">No Python gateway on this machine. Downloading a one-time runtime (uv + Python 3.12 + agentica). The window opens when it is ready.</p>
</body></html>`;
  w.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(html));
  return w;
}

async function boot() {
  gateway = new GatewayProcess(log, path.join(app.getPath("userData"), "gateway-port"));
  let splash = null;
  if (!resolveExistingCommand() && process.env.AGENTICA_DESKTOP_NO_BOOTSTRAP !== "1") {
    splash = showSetupWindow();
  }
  try {
    await startGatewayAndWindow();
  } catch (err) {
    // No gateway on the machine, and the one-time download did not finish
    // (no network, or the installer did not land). Leave the window up with
    // what to do next. An error dialog is the wrong surface here: it exits
    // the process, and its wording ("could not start") is what the AppImage
    // catalog's screenshot check rejects when it launches the app offline.
    if (splash && !splash.isDestroyed() && !resolveExistingCommand()) {
      log(`boot stopped before a gateway: ${err.stack || err.message}`);
      splash.setResizable(true);
      splash.setSize(640, 420);
      await splash.loadURL(
        "data:text/html;charset=utf-8," + encodeURIComponent(bootstrapBlockedPage()),
      );
      return;
    }
    if (splash && !splash.isDestroyed()) splash.close();
    fatal("Agentica could not start.", err);
    return;
  }
  if (splash && !splash.isDestroyed()) splash.close();
  startUpdates(app, log);
  // Lets the packaging and notification checks drive one banner without a
  // session: `AGENTICA_DESKTOP_NOTIFY_TEST=1` shows it once the window is up.
  if (process.env.AGENTICA_DESKTOP_NOTIFY_TEST === "1") {
    const { title, body } = formatTurnNotification({ lang: app.getLocale() || "en", isTest: true });
    notifyUser(title, body);
  }
}

/**
 * The native menu.
 *
 * Reload and DevTools are here rather than behind a preload bridge the page
 * could call: a window with no menu cannot be reloaded or inspected at all,
 * which turns any renderer hiccup into "quit and reopen", and a bridge would be
 * a capability that exists only in the desktop build.
 *
 * The shell's own text is English and does not follow the in-app language
 * setting. That setting lives in the renderer's localStorage, and every string
 * here is either a native `role` (which Electron localises itself, from the OS
 * language) or a dialog that fires when the gateway never came up — i.e. when
 * there is no renderer to read a preference from.
 */
const execFileAsync = promisify(execFile);

function currentLang() {
  return lastRendererLang || app.getLocale() || "zh";
}

function isZhLang() {
  return /^zh/i.test(currentLang());
}

function agenticaHomeDir() {
  return process.env.AGENTICA_HOME || path.join(os.homedir(), ".agentica");
}

function logsDirPath() {
  return path.join(agenticaHomeDir(), "logs");
}

function showAboutDialog() {
  const zh = isZhLang();
  void dialog.showMessageBox({
    type: "info",
    title: zh ? "关于 Agentica" : "About Agentica",
    message: `Agentica v${app.getVersion()}`,
    detail: `Electron ${process.versions.electron}\nNode ${process.versions.node}\nChrome ${process.versions.chrome}\n\n${URLS.docs}`,
  });
}

function installMenu() {
  const lang = currentLang();
  lastMenuLang = lang;
  Menu.setApplicationMenu(Menu.buildFromTemplate(buildAppMenuTemplate({
    platform: process.platform,
    lang,
    handlers: {
      onAbout: showAboutDialog,
      onCheckUpdates: () => void checkForUpdatesInteractive({ app, dialog, shell, lang: currentLang() }),
      onManageCli: () => void manageTerminalCommand(),
      onNewChat: () => openFromDock({ action: "new-chat" }),
      onOpenDocs: () => void shell.openExternal(URLS.docs),
      onDiagnostics: () => void showDiagnostics(),
      onResourceUsage: () => void showResourceUsage(),
      onOpenLogs: () => void openLogsFolder(),
      onFeedback: () => void shell.openExternal(URLS.issues),
      onTestNotification: () => {
        const { title, body } = formatTurnNotification({ lang: currentLang(), isTest: true });
        notifyUser(title, body);
      },
    },
  })));
}

/** Menu action: inspect, install, or remove the desktop-managed `agentica` CLI shim. */
async function manageTerminalCommand() {
  const zh = isZhLang();
  const status = inspectCommand();
  if (status.exists && status.managedByApp) {
    const { response } = await dialog.showMessageBox({
      type: "info",
      message: zh ? "已安装终端命令 `agentica`" : "The 'agentica' command is installed",
      detail: zh
        ? `路径：${status.target}\n${status.onPath ? "当前 PATH 已包含该目录，可直接在终端运行 `agentica`。" : `请将 ${path.dirname(status.target)} 加入 PATH。`}`
        : `Path: ${status.target}\n${status.onPath ? "Ready to use in any terminal." : `Add ${path.dirname(status.target)} to your PATH.`}`,
      buttons: [
        zh ? "关闭" : "Close",
        zh ? "重新安装" : "Reinstall",
        zh ? "移除命令" : "Remove",
      ],
      defaultId: 0,
      cancelId: 0,
    });
    if (response === 2) {
      try {
        uninstallCommand();
        await dialog.showMessageBox({
          type: "info",
          message: zh ? "已移除终端命令" : "Removed 'agentica' command",
          detail: status.target,
        });
      } catch (err) {
        await dialog.showMessageBox({
          type: "warning",
          message: zh ? "移除终端命令失败" : "Could not remove the 'agentica' command",
          detail: err.message,
        });
      }
      return;
    }
    if (response !== 1) return;
  }
  try {
    const { target, onPath, restart } = installCommand();
    const hint = restart
      ? (zh ? "请新开一个终端窗口使用。" : "Open a new terminal window to use it.")
      : onPath
        ? (zh ? "现在可以直接在任意终端使用 `agentica`。" : "It is ready to use in any terminal.")
        : (zh ? `请将 ${path.dirname(target)} 加入你的 PATH。` : `Add ${path.dirname(target)} to your PATH to use it.`);
    await dialog.showMessageBox({
      type: "info",
      message: zh ? "已安装 `agentica` 终端命令" : "The 'agentica' command is installed.",
      detail: `${target}\n\n${hint}`,
    });
  } catch (err) {
    await dialog.showMessageBox({
      type: "warning",
      message: zh ? "无法安装 `agentica` 命令" : "Could not install the 'agentica' command.",
      detail: err.message,
    });
  }
}

async function openLogsFolder() {
  const zh = isZhLang();
  const dir = logsDirPath();
  try {
    fs.mkdirSync(dir, { recursive: true });
    const err = await shell.openPath(dir);
    if (err) throw new Error(err);
  } catch (e) {
    await dialog.showMessageBox({
      type: "warning",
      message: zh ? "无法打开日志文件夹" : "Could not open logs folder",
      detail: `${dir}\n\n${e instanceof Error ? e.message : String(e)}`,
    });
  }
}

async function showDiagnostics() {
  const zh = isZhLang();
  const record = gateway && gateway.record ? gateway.record : null;
  let healthOk = false;
  let healthMs = 0;
  let healthError = "";
  if (record && record.port) {
    const t0 = Date.now();
    try {
      healthOk = await health(record, 1500);
      healthMs = Date.now() - t0;
      if (!healthOk) healthError = "Unhealthy";
    } catch (e) {
      healthError = e instanceof Error ? e.message : String(e);
    }
  } else {
    healthError = zh ? "Gateway 未就绪" : "Gateway not ready";
  }
  let cmdSource = "-";
  try {
    const cmd = resolveExistingCommand();
    cmdSource = cmd ? `${cmd.source} (${cmd.command})` : "bootstrap needed";
  } catch (e) {
    cmdSource = e instanceof Error ? e.message : String(e);
  }
  const cli = inspectCommand();
  const report = formatDiagnosticsReport({
    appVersion: app.getVersion(),
    packaged: app.isPackaged,
    electron: process.versions.electron,
    node: process.versions.node,
    chrome: process.versions.chrome,
    platform: process.platform,
    arch: process.arch,
    osRelease: os.release(),
    signed: canSelfUpdate(),
    gatewayRunning: Boolean(gateway && gateway.running),
    gatewayPort: record ? record.port : null,
    gatewayPid: record ? record.pid : null,
    gatewaySpawned: gateway ? gateway.spawned : undefined,
    healthOk,
    healthMs,
    healthError,
    commandSource: cmdSource,
    cliTarget: cli.target,
    cliInstalled: cli.exists,
    agenticaHome: agenticaHomeDir(),
    logsDir: logsDirPath(),
    userDataDir: app.getPath("userData"),
  }, currentLang());

  const { response } = await dialog.showMessageBox({
    type: "info",
    title: zh ? "诊断工具" : "Diagnostics",
    message: zh ? "Agentica 运行诊断" : "Agentica Diagnostics",
    detail: report,
    buttons: [
      zh ? "关闭" : "Close",
      zh ? "复制诊断信息" : "Copy Diagnostics",
      zh ? "打开日志文件夹" : "Open Logs Folder",
    ],
    defaultId: 0,
    cancelId: 0,
  });
  if (response === 1) clipboard.writeText(report);
  else if (response === 2) await openLogsFolder();
}

async function readExtraProcessStats(pid) {
  if (!pid || process.platform === "win32") return [];
  try {
    const { stdout } = await execFileAsync("ps", ["-o", "pid=,rss=,%cpu=", "-p", String(pid)], { timeout: 2000 });
    const line = stdout.trim().split(/\r?\n/)[0] || "";
    const parts = line.trim().split(/\s+/);
    if (parts.length < 3) return [];
    return [{
      pid: Number(parts[0]),
      rssKb: Number(parts[1]),
      cpuPercent: Number(parts[2]),
      label: "Gateway (Python)",
    }];
  } catch {
    return [];
  }
}

async function showResourceUsage() {
  const zh = isZhLang();
  const gwPid = gateway && gateway.record && gateway.record.pid ? gateway.record.pid : null;
  const [metrics, extra] = await Promise.all([
    Promise.resolve(app.getAppMetrics()),
    readExtraProcessStats(gwPid),
  ]);
  const report = formatResourceUsageReport(metrics, extra, currentLang());
  const { response } = await dialog.showMessageBox({
    type: "info",
    title: zh ? "资源占用" : "Resource Usage",
    message: zh ? "Agentica 进程资源占用" : "Agentica Resource Usage",
    detail: report,
    buttons: [
      zh ? "关闭" : "Close",
      zh ? "刷新" : "Refresh",
      zh ? "复制" : "Copy",
    ],
    defaultId: 0,
    cancelId: 0,
  });
  if (response === 1) await showResourceUsage();
  else if (response === 2) clipboard.writeText(report);
}

/**
 * The dock icon, macOS and unpackaged runs only.
 *
 * macOS reads an app's icon from its bundle, so `BrowserWindow#icon` is ignored
 * and a packaged build needs nothing here — but `npm start` runs inside
 * Electron's own bundle, which is why a dev launch otherwise shows the generic
 * Electron atom in the dock and the ⌘-Tab switcher.
 */
function installDockIcon() {
  if (!icon || process.platform !== "darwin" || app.isPackaged) return;
  app.dock.setIcon(icon);
}

let cachedRecentSessions = [];
let lastDockFetchAt = 0;
let lastRendererLang = "";
let lastMenuLang = "";

/** Was the last observed turn running, and on which session?
 *  The page mirrors the active session into `<html data-session-running>`, and
 *  the title carries a "● " prefix while a turn runs — so a title update is
 *  exactly the moment that flag flips. That is the whole trigger: no polling,
 *  no preload and no IPC. */
let lastTurnRunning = false;
let lastTurnSession = "";
let lastTurnTitle = "";
let titleCheckSeq = 0;

/** A system banner, the way every other desktop app announces a finished job.
 *  macOS asks for permission the first time one is shown; before that the app
 *  does not even appear under System Settings → Notifications. */
function notifyUser(title, body, sessionId = "") {
  if (!Notification.isSupported()) return;
  const notification = new Notification({ title, body });
  notification.on("click", () => {
    if (sessionId) {
      openFromDock({ action: "session", sessionId });
      return;
    }
    const { win: target } = focusOrCreateWindow();
    if (target && !target.isDestroyed()) target.focus();
  });
  notification.show();
}

async function onTitleChanged() {
  if (!win || win.isDestroyed() || quitting) return;
  const seq = ++titleCheckSeq;
  const state = await readRendererSession();
  if (!state || seq !== titleCheckSeq || !win || win.isDestroyed() || quitting) return;
  const finished = lastTurnRunning && !state.running && state.id === lastTurnSession;
  const prevTitle = state.id === lastTurnSession ? lastTurnTitle : "";
  lastTurnRunning = state.running === true;
  lastTurnSession = state.id;
  lastTurnTitle = state.title || prevTitle || "";
  if (!finished || win.isFocused()) return;
  let sessionTitle = state.title || prevTitle || "";
  if (!sessionTitle || sessionTitle === "Chat" || sessionTitle === "New Chat") {
    const findName = (rows) => {
      const hit = Array.isArray(rows) ? rows.find((r) => r && r.session_id === state.id) : null;
      return hit && hit.name ? String(hit.name) : "";
    };
    sessionTitle = findName(cachedRecentSessions) || findName(await fetchRecentSessions(true)) || sessionTitle;
  }
  const { title, body } = formatTurnNotification({
    sessionTitle,
    lang: state.lang || lastRendererLang || app.getLocale() || "zh",
  });
  notifyUser(title, body, state.id);
}

function focusOrCreateWindow(targetUrl = null) {
  if (win && !win.isDestroyed()) {
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
    return { win, created: false };
  }
  if (origin !== null) {
    const w = createWindow();
    void w.loadURL(targetUrl || `${origin}/chat`);
    return { win: w, created: true };
  }
  return { win: null, created: false };
}

function openFromDock(detail) {
  const query = detail.action === "new-chat"
    ? "?new=1"
    : detail.sessionId
      ? `?session=${encodeURIComponent(detail.sessionId)}`
      : "";
  const { win: w, created } = focusOrCreateWindow(origin ? `${origin}/chat${query}` : null);
  if (w && !created) {
    void w.webContents
      .executeJavaScript(
        `window.dispatchEvent(new CustomEvent("agentica:desktop-nav", { detail: ${JSON.stringify(detail)} }))`,
      )
      .catch(() => {});
  }
}

async function readRendererSession() {
  if (!win || win.isDestroyed()) return null;
  if (!isAppUrl(win.webContents.getURL(), origin)) return null;
  try {
    const state = await win.webContents.executeJavaScript(
      `({
        lang: document.documentElement.lang || "",
        id: document.documentElement.dataset.sessionId || "",
        title: document.documentElement.dataset.sessionTitle || "",
        running: document.documentElement.dataset.sessionRunning === "1"
      })`,
    );
    if (state && state.lang && state.lang !== lastRendererLang) {
      lastRendererLang = state.lang;
      if (lastRendererLang !== lastMenuLang) installMenu();
    }
    return state;
  } catch {
    return null;
  }
}

async function fetchRecentSessions(force = false) {
  if (!origin) return cachedRecentSessions;
  if (!force && Date.now() - lastDockFetchAt < 3000) return cachedRecentSessions;
  const headers = {};
  try {
    const cookies = await session.defaultSession.cookies.get({ url: origin, name: "agentica_session" });
    if (cookies && cookies.length && cookies[0].value) {
      headers.cookie = `agentica_session=${cookies[0].value}`;
    }
  } catch {
    /* ignore cookie lookup errors */
  }
  if (!headers.cookie && gateway && gateway.record && gateway.record.token) {
    headers.authorization = `Bearer ${gateway.record.token}`;
  }
  try {
    const resp = await fetch(`${origin}/api/sessions?scope=recent`, {
      headers,
      signal: AbortSignal.timeout(2000),
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data && Array.isArray(data.sessions)) {
        cachedRecentSessions = data.sessions;
        lastDockFetchAt = Date.now();
      }
    }
  } catch {
    /* gateway busy or restarting; keep last known list */
  }
  return cachedRecentSessions;
}

async function refreshDockMenu(forceFetch = false) {
  if (process.platform !== "darwin" || !app.dock || quitting) return;
  const [current, recent] = await Promise.all([
    readRendererSession(),
    fetchRecentSessions(forceFetch),
  ]);
  if (quitting) return;
  const lang = (current && current.lang) || lastRendererLang || app.getLocale() || "zh";
  const template = buildDockMenuTemplate({
    current,
    recent,
    lang,
    onNewChat: () => openFromDock({ action: "new-chat" }),
    onSelectSession: (sessionId) => openFromDock({ action: "session", sessionId }),
  });
  app.dock.setMenu(Menu.buildFromTemplate(template));
}

// A second launch must reach the window that already exists, not start a
// second gateway against the same ~/.agentica.
if (!app.requestSingleInstanceLock()) {
  if (smoke) {
    process.stdout.write(`DESKTOP-SMOKE-RESULT ${JSON.stringify({ error: "single-instance lock held", version: app.getVersion() })}\n`);
  }
  app.quit();
} else {
  app.on("second-instance", () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  // Pages in the SPA's browser tab. A link that wants a new window loads in
  // the same tab; anything that leaves http(s) is dropped; no page gets a
  // permission (camera, location, notifications) it was never asked about.
  app.on("web-contents-created", (_event, contents) => {
    if (contents.getType() !== "webview") return;
    contents.setWindowOpenHandler(({ url }) => {
      if (isBrowsableUrl(url)) void contents.loadURL(url);
      return { action: "deny" };
    });
    contents.on("will-navigate", (event, url) => {
      if (!isBrowsableUrl(url)) event.preventDefault();
    });
  });

  app.whenReady().then(() => {
    app.setAboutPanelOptions({
      applicationName: "Agentica",
      applicationVersion: app.getVersion(),
      version: app.getVersion(),
      copyright: "Copyright © 2026 XuMing",
      website: URLS.docs,
      ...(!app.isPackaged && fs.existsSync(iconPath()) ? { iconPath: iconPath() } : {}),
    });
    session.fromPartition(BROWSER_PARTITION).setPermissionRequestHandler((_contents, _permission, done) => done(false));
    installMenu();
    installDockIcon();
    void refreshDockMenu(false);
    return boot();
  });

  app.on("activate", () => {
    if (win === null && origin !== null) void createWindow().loadURL(`${origin}/chat`);
  });

  // macOS keeps the app alive with no windows; on the other platforms closing
  // the window is quitting, and quitting is what stops the gateway.
  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });

  // Stopping the gateway is an await, so the quit has to be held open for it.
  // `will-quit` with a fire-and-forget stop let the app exit first, which took
  // the SIGKILL fallback down with it and left the orphan to the gateway's own
  // parent watchdog two seconds later.
  app.on("before-quit", (event) => {
    quitting = true;
    if (gateway && gateway.running && stopping === null) {
      event.preventDefault();
      stopping = gateway.stop().finally(() => app.quit());
    }
  });
}

// --- smoke hook ------------------------------------------------------------
// Verifying the shell from outside means attaching a debugger to it, which is a
// different process tree and a different quit path from the one users get. So
// the check lives in the app: it reports what the window actually loaded and
// then quits through `before-quit`, exercising the real teardown.

function armSmokeProbe(target) {
  if (!smoke) return;
  target.webContents.once("did-finish-load", () => {
    void (async () => {
      try {
        const probe = await waitForChat(() => target.webContents.executeJavaScript(
          `({ composer: !!document.querySelector(".input-ta"),
              login: !!document.querySelector("input[type=password]"),
              sessions: document.querySelectorAll(".s-item").length })`
        ));
        const result = {
          url: target.webContents.getURL(),
          title: target.webContents.getTitle(),
          origin,
          version: app.getVersion(),
          targetPid: gateway?.record?.pid ?? null,
          spawned: gateway ? gateway.spawned : false,
          gatewayPid: gateway && gateway.child ? gateway.child.pid : null,
          iconSize: icon ? icon.getSize() : null,
          probe,
        };
        const shot = process.env.AGENTICA_DESKTOP_SMOKE_SHOT;
        if (shot) {
          const image = await target.webContents.capturePage();
          fs.writeFileSync(shot, image.toPNG());
        }
        process.stdout.write(`DESKTOP-SMOKE-RESULT ${JSON.stringify(result)}\n`);
      } catch (err) {
        process.stdout.write(`DESKTOP-SMOKE-RESULT ${JSON.stringify({ error: String(err), version: app.getVersion() })}\n`);
        process.exitCode = 1;
      } finally {
        app.quit();
      }
    })();
  });
}
