# Agentica Desktop

An Electron shell around the gateway. It does not implement any product
feature — it starts a process, opens a window on it, and stops what it started.
Everything you see in the window is the same SPA a browser gets from
`agentica-gateway`, over the same HTTP API.

If a capability is not reachable over HTTP, the desktop app cannot do it
either. That is the point: there is no second UI to keep in sync. It is also
the least updatable code in the product — a change here reaches a user only
through a new installer, while a change in the SPA or the gateway reaches them
on the next `uv tool upgrade` — so anything that *could* ship over HTTP must.

## Run it

```bash
pip install -e ".[gateway]"   # from the repo root; skip if you want first-launch bootstrap
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
| `AGENTICA_DESKTOP_RUNTIME` | Override the managed-runtime root (tests). Default: macOS `~/Library/Application Support/Agentica`, Linux `~/.local/share/agentica`, Windows `%LOCALAPPDATA%\\Agentica`. |
| `AGENTICA_DESKTOP_PACKAGE` | pip spec for first-launch install (default `agentica[gateway]`). |
| `AGENTICA_DESKTOP_IGNORE_EXISTING=1` | Skip login-shell PATH / python; only env + managed runtime + bootstrap. |
| `AGENTICA_DESKTOP_NO_BOOTSTRAP=1` | Never download uv/Python. Fail if nothing is already installed. |
| `AGENTICA_HOME` | Data root. Must match the one your CLI uses, or the app sees a different set of sessions. |
| `GATEWAY_AUTH=false` | Turns the auth gate off in the spawned gateway. |
| `AGENTICA_DESKTOP_SMOKE=1` | Print one `DESKTOP-SMOKE-RESULT {json}` line describing what the window loaded, then quit through the normal quit path. |
| `AGENTICA_DESKTOP_SMOKE_SHOT` | With the above: also write a PNG of the window there. |

`AGENTICA_DESKTOP_BUNDLED_RUNTIME` points at a runtime directory other than
`<resources>/runtime` (tests, `npm start` against a locally prepared runtime).
`AGENTICA_DESKTOP_NO_UPDATE=1` skips the update check.

Without a bundled runtime (`npm start`), an app launched from the Dock does not
inherit your shell's `PATH`, so a conda or venv `agentica-gateway` is invisible
to it. Resolution asks the login shell first
(`$SHELL -ilc 'command -v agentica-gateway'`), then a managed venv under
Application Support. If neither exists, the first launch downloads uv, installs
CPython 3.12, and `pip install`s `agentica[gateway]` into that venv — once.
`~/.agentica` stays data-only. Set `AGENTICA_GATEWAY_BIN` to skip the search.

## Packaging

```bash
CSC_IDENTITY_AUTO_DISCOVERY=false npm run dist:mac    # or dist:win / dist:linux
```

Output lands in `desktop/dist/` as `agentica-desktop-<platform>-<arch>.<ext>`,
except the AppImage, which is `Agentica-x86_64.AppImage`. The catalog at
appimage.github.io rejects a filename that contains "linux", and the old
FUSE2 runtime (electron-builder's default on the 26.x line) needs libfuse2
on the host. `toolsets.appimage` is pinned to `1.0.3`, the static runtime.
That runtime adds `--no-sandbox` only when user namespaces are missing; the
catalog enables them and then runs under firejail, where the chrome sandbox
still does not start, so the AppImage target passes `--no-sandbox` itself.
The deb does not.
A given OS can only build its own installers — the NSIS exe needs Windows and
the deb/AppImage need Linux — so the full matrix runs in CI
(`.github/workflows/desktop.yml`) and attaches the installers to the release for
whatever `v*` tag was pushed. Builds are unsigned; the root README documents the
one-time unlock each OS asks for.

## Bundled runtime (no system dependencies)

Every installer carries its own CPython 3.12 (python-build-standalone) with
`agentica[gateway]` already installed, under `<resources>/runtime/python`. The
app needs no system Python, Node, uv or network on first launch, and the shell
and the gateway are always the same version: they are built from one commit and
shipped, signed and updated as one unit.

```bash
cd web && npm ci && npm run build          # the wheel carries the SPA
cd ../desktop && npm ci
node scripts/prepare-runtime.js            # host platform; or --target mac-arm64 --target mac-x64 ...
CSC_IDENTITY_AUTO_DISCOVERY=false npm run dist:mac
```

`prepare-runtime.js` builds a wheel of this checkout and installs it with
`pip --platform ... --target`, so one runner can assemble a runtime for another
architecture. The CPython build is pinned (`AGENTICA_PBS_TAG` /
`AGENTICA_PBS_PYTHON` override it). `extraResources` selects
`runtime/${os}-${arch}` per target. The unpacked runtime is roughly 380 MB
(`lark_oapi` alone is ~100 MB) and compresses to about a third in the installer.

Gateway resolution is: `AGENTICA_GATEWAY_BIN`, then the **bundled** interpreter,
then (only for `npm start` and runtime-less builds) a login-shell
`agentica-gateway`, the managed venv, a system Python, and finally the uv
bootstrap. The bundled interpreter runs with `PYTHONNOUSERSITE=1`, so a
`pip install --user` never shadows what shipped. The desktop and the CLI share
the data under `~/.agentica` but never an executable.

`Help > Install 'agentica' Command in Terminal…` writes an `agentica` wrapper
that runs the bundled Python (`/usr/local/bin` on macOS when writable, else
`~/.local/bin`; on Windows `%LOCALAPPDATA%\Agentica\bin` plus the user PATH).
It will not overwrite an `agentica` it did not create.

## Updates and signing

`electron-updater` checks GitHub Releases at startup and installs a new version
that replaces shell, Python and agentica together. CI uploads the
`latest*.yml` and blockmap files next to the installers. macOS only accepts an
update for a signed app, so an unsigned build skips the check
(`AGENTICA_DESKTOP_NO_UPDATE=1` disables it everywhere).

Signing is driven by repository secrets in `.github/workflows/desktop.yml`:

| Platform | Secrets |
|---|---|
| macOS (sign + notarize) | `CSC_LINK`, `CSC_KEY_PASSWORD`, `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID` |
| Windows | `WIN_CSC_LINK`, `WIN_CSC_KEY_PASSWORD` |

Without them the build is unsigned, exactly as before.
