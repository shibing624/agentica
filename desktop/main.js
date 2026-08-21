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
const { app, BrowserWindow, dialog, Menu, nativeImage, session, shell } = require("electron");
const { GatewayProcess } = require("./gateway");
const { resolveExistingCommand } = require("./runtime");
const {
  HEALTHY_AFTER_MS, MAX_GATEWAY_RESTARTS, iconPath, isAppUrl, restartDelayMs,
} = require("./util");

const log = (msg) => console.log(`[desktop] ${msg}`);

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
    ...(icon ? { icon } : {}),
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
    if (splash && !splash.isDestroyed()) splash.close();
    fatal("Agentica could not start.", err);
    return;
  }
  if (splash && !splash.isDestroyed()) splash.close();
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
function installMenu() {
  const isMac = process.platform === "darwin";
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(isMac ? [{ role: "appMenu" }] : []),
    { role: "fileMenu" },
    { role: "editMenu" },
    {
      label: "View",
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
    installDockIcon();
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
            // Whether this platform's icon file decoded. An ICO handed to a
            // non-Windows build decodes to an empty image, which looks exactly
            // like having configured no icon at all.
            iconSize: icon ? icon.getSize() : null,
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
