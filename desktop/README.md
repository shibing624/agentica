# Agentica Desktop

An Electron shell around the gateway. It does not implement any product
feature — it starts a process, opens a window on it, and stops what it started.
Everything you see in the window is the same SPA a browser gets from
`agentica-gateway`, over the same HTTP API.

If a capability is not reachable over HTTP, the desktop app cannot do it
either. That is the point: there is no second UI to keep in sync. It is also
the least updatable code in the product — a change here reaches a user only
through a new installer, while a change in the SPA or the gateway reaches them
on the next `pip install -U` — so anything that *could* ship over HTTP must.

## Run it

```bash
pip install -e ".[gateway]"   # from the repo root, once
cd desktop
npm install
npm start
```

The window is empty of chrome-level features on purpose: no preload script, no
IPC, `nodeIntegration: false`, `sandbox: true`. External links open in your own
browser; only this instance's own origin stays in the window.

The native menu carries reload / force-reload / DevTools. Those are here rather
than behind a bridge the page could call, because a window with no menu cannot
be reloaded at all — and a bridge would be a capability that exists only in the
desktop build.

## What it does on startup

1. **Single instance.** A second launch focuses the existing window.
2. **Attach before spawn.** If a gateway is already running on this
   `~/.agentica` (you started `agentica-gateway` yourself, or another shell
   did), it connects to that one. It never starts a second gateway against the
   same data root, and never stops a gateway it did not start.
3. **Otherwise spawn** `agentica-gateway --port <sticky> --parent-pid <shell
   pid>`, wait for the runtime record and `/api/health`, then load `/chat`.
4. **Sign the window in** by redeeming the machine token for a session cookie
   before the first navigation (see below).
5. **Restart on death.** If the gateway exits on its own the shell restarts it
   after 1s, 2s, 4s, and gives up with a dialog after three tries. A run that
   stays up for a minute resets that budget.
6. **Teardown.** Quitting asks `POST /api/desktop/shutdown` first, then SIGTERM,
   then SIGKILL — and the quit is *held open* until the child is gone.
   `--parent-pid` remains the backstop: if the shell itself is SIGKILLed, the
   gateway notices the pid is gone and exits on its own.

### Why the port is sticky

The SPA keeps the session tree, the active session and the theme in
`localStorage`, which is scoped to the origin — and the origin is
`http://127.0.0.1:<port>`. So a fresh `--port 0` every launch hands the user an
empty sidebar every launch: same sessions on disk, different origin, no memory
of which one they were in.

So the shell remembers the port the gateway actually bound
(`<userData>/gateway-port`) and asks for it again next time, falling back to `0`
when something else has taken it. No fixed port is introduced — a hard-coded
8881 would collide with the user's own `agentica-gateway`, and the collision
would surface as a blank window.

### Why the token becomes a session

The gateway's cookie holds a *session*, not the machine token: the token is per
process, so a cookie holding it would expire on every gateway restart. The
shell redeems the token from `runtime.json` against `/api/status?token=…` and
copies the session it gets back into Electron's cookie jar, so the page loads
already signed in and the renderer never sees the token. That round trip is
also the check that a stale `runtime.json` token still belongs to whatever is
answering on that port.

## Environment

| Variable | Effect |
|---|---|
| `AGENTICA_GATEWAY_BIN` | Path to the gateway executable. Skips shell resolution. |
| `AGENTICA_HOME` | Data root. Must match the one your CLI uses, or the app sees a different set of sessions. |
| `GATEWAY_AUTH=false` | Turns the auth gate off in the spawned gateway. |
| `AGENTICA_DESKTOP_SMOKE=1` | Print one `DESKTOP-SMOKE-RESULT {json}` line describing what the window loaded, then quit through the normal quit path. |
| `AGENTICA_DESKTOP_SMOKE_SHOT` | With the above: also write a PNG of the window there. |

An app launched from the Dock does not inherit your shell's `PATH`, so a conda
or venv `agentica-gateway` is invisible to it. Resolution therefore asks your
login shell (`$SHELL -ilc 'command -v agentica-gateway'`); set
`AGENTICA_GATEWAY_BIN` to skip that.

## The icon

The same waving cat as the browser tab and the SPA sidebar, taken straight from
the repo's assets rather than a copy under `desktop/` — a duplicated binary is
a second cat that drifts. Two files, because one cannot cover both platforms:

| Platform | File | Why |
|---|---|---|
| Windows | `docs/assets/favicon.ico` | Multi-size (16/32/48), and `nativeImage` decodes ICO **only** on Windows — elsewhere it yields an empty image, which looks identical to having set no icon |
| Everything else | `web/src/assets/cat.png` | 256², transparent |

macOS ignores `BrowserWindow#icon` entirely and takes the icon from the app
bundle, so a packaged build needs nothing — but `npm start` runs inside
*Electron's* bundle, which is why `installDockIcon()` sets the dock explicitly
while unpackaged.

Packaging will need an `.icns`; generate it from the same cat, and point
electron-builder at these paths rather than adding copies.

## Not done here

Packaging into an installer, auto-update, tray. The first version requires
`agentica[gateway]` to already be installed on the machine — see
`docs/learn_cc/web_v2.md` §5 for why that ordering is deliberate.

## Verifying a change

Two runnable checks, both from the repo root, both against a temporary
`AGENTICA_HOME` *and* a temporary Electron `userData` (the port memory lives
there, and a stale one would make the stickiness check meaningless):

```bash
python tmp/desktop_prereq_smoke.py   # gateway side: port, auth, login, orphan
python tmp/desktop_shell_smoke.py    # the real shell: window, stickiness, attach, teardown
```

The shell check reads the app's own `AGENTICA_DESKTOP_SMOKE` report rather than
attaching a debugger to it, so the teardown it exercises is the one users get.
