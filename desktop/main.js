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
 * What it does:
 *
 *   1. one instance per machine,
 *   2. attach to a running gateway or start one (on last launch's port, so the
 *      window's origin — and its localStorage — survives a restart),
 *   3. open a window on it, already signed in,
 *   4. restart the gateway if it dies, with a backoff and a cap,
 *   5. stop what it started on the way out, gracefully.
 */

const path = require("node:path");
const { app, BrowserWindow, dialog, Menu, session, shell } = require("electron");
const { GatewayProcess } = require("./gateway");
const {
  HEALTHY_AFTER_MS, MAX_GATEWAY_RESTARTS, isAppUrl, restartDelayMs,
} = require("./util");

const log = (msg) => console.log(`[desktop] ${msg}`);

let gateway = null;
let win = null;
/** The origin the window is on; null until boot resolves. */
let origin = null;
let quitting = false;
let stopping = null;
let restarts = 0;

function fatal(context, err) {
  const detail = err instanceof Error ? (err.stack || err.message) : String(err);
  log(`${context}: ${detail}`);
  dialog.showErrorBox("Agentica", `${context}\n\n${detail}`);
  app.exit(1);
}

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 720,
    minHeight: 480,
    title: "Agentica",
    backgroundColor: "#ffffff",
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  win.once("ready-to-show", () => win.show());
  win.on("closed", () => {
    win = null;
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
      `gateway 反复退出（最后一次 ${code}），已放弃重启。`,
      "在终端里跑 `agentica-gateway` 看具体报错。"
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
    fatal("gateway 重启失败。", err);
  }
}

async function boot() {
  gateway = new GatewayProcess(log, path.join(app.getPath("userData"), "gateway-port"));
  try {
    await startGatewayAndWindow();
  } catch (err) {
    // A blank window with no explanation is the worst outcome here: the usual
    // causes (gateway not installed, no API key, bad config.yaml) are all
    // fixable, and the message says which one it is.
    fatal("Agentica 启动失败。", err);
  }
}

/**
 * The native menu.
 *
 * Reload and DevTools are here rather than behind a preload bridge the page
 * could call: a window with no menu cannot be reloaded or inspected at all,
 * which turns any renderer hiccup into "quit and reopen", and a bridge would be
 * a capability that exists only in the desktop build.
 */
function installMenu() {
  const isMac = process.platform === "darwin";
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(isMac ? [{ role: "appMenu" }] : []),
    { role: "fileMenu" },
    { role: "editMenu" },
    {
      label: "视图",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    { role: "windowMenu" },
  ]));
}

// A second launch must reach the window that already exists, not start a
// second gateway against the same ~/.agentica.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  app.whenReady().then(() => {
    installMenu();
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

const SMOKE_SETTLE_MS = 2500;

function armSmokeProbe(target) {
  if (process.env.AGENTICA_DESKTOP_SMOKE !== "1") return;
  target.webContents.once("did-finish-load", () => {
    setTimeout(() => {
      void (async () => {
        try {
          const result = {
            url: target.webContents.getURL(),
            title: target.webContents.getTitle(),
            origin,
            spawned: gateway ? gateway.spawned : false,
            gatewayPid: gateway && gateway.child ? gateway.child.pid : null,
            // What the page can see of itself: a signed-in SPA renders the
            // composer, the login page renders a password field.
            probe: await target.webContents.executeJavaScript(
              `({ composer: !!document.querySelector("textarea"),
                  login: !!document.querySelector("input[type=password]"),
                  sessions: document.querySelectorAll(".s-item").length })`
            ),
          };
          const shot = process.env.AGENTICA_DESKTOP_SMOKE_SHOT;
          if (shot) {
            const image = await target.webContents.capturePage();
            require("node:fs").writeFileSync(shot, image.toPNG());
          }
          process.stdout.write(`DESKTOP-SMOKE-RESULT ${JSON.stringify(result)}\n`);
        } catch (err) {
          process.stdout.write(`DESKTOP-SMOKE-RESULT ${JSON.stringify({ error: String(err) })}\n`);
        } finally {
          app.quit();
        }
      })();
    }, SMOKE_SETTLE_MS);
  });
}
