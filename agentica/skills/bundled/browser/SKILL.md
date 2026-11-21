---
name: browser
description: >-
  Drive a real or headless browser to verify web UI changes, click through
  flows, inspect console/network errors, or act inside the user's logged-in
  Chrome/Edge session. Prefers the strong-typed BrowserTool when available;
  otherwise drives browser CLIs from the shell (agent-browser, lightpanda,
  bsk BrowserSkill, playwright-cli).
metadata:
  version: "1.1"
---

# Browser automation

Two surfaces, in this order:

1. **Preferred — `BrowserTool`**: strong-typed tools when the agent has them
   installed. It encodes the observation/ref/action contract so the model does
   not have to reconstruct it from shell text.
2. **Fallback — shell drivers via `execute`**: no schema tokens on turns that
   do not touch a browser, but the model must manage daemon lifecycle, refs and
   cleanup itself.

There is no dedicated browser tool in the **default** tool list on purpose.
Enable one explicitly with `Agent(tools=[BrowserTool()])`. Drive the CLI
fallback through `execute`.

If you only need to read a public static page, use `fetch_url` first. Reach
for a browser when the page needs JavaScript, you are verifying a UI you just
edited, or the task requires clicking, filling forms, screenshots, or the
user's existing login state.

---

## 1. Strong-typed tools (`BrowserTool`)

```python
from agentica import Agent, BrowserTool

browser = BrowserTool()          # one warm `agent-browser` daemon session
agent = Agent(tools=[browser])
try:
    ...
finally:
    await browser.close()        # application-boundary cleanup
```

Registered functions: `browser_observe`, `browser_act`, `browser_press`,
`browser_upload`, `browser_close`. The contract is also injected into the
system prompt via `get_system_prompt()`:

- `browser_observe(url, max_elements)` returns a compact accessibility tree
  with short refs (`[ref=eN]`) plus a generation number; defaults to 240
  interactive elements (use `snapshot -i` semantics).
- Refs belong to one generation. Navigation, any action, or a re-observation
  invalidates them; `browser_act` refuses stale refs with an explicit error.
- `browser_act(ref, action, text, wait_for)` combines the action, an optional
  `wait`, and a URL read in ONE call. Do not split action + verification into
  separate turns.
- `browser_press(text=...)` inserts a whole text block via
  `keyboard inserttext`; never send one character per call.
- `browser_close()` ends the daemon; also call `await browser.close()` in a
  `finally` block at the application boundary.

The driver is replaceable: pass any object implementing `run(*args)` and
`close()` (see `AgentBrowserDriver`). A future in-app `WebContentsView`
backend can be slotted in without changing the tool surface.

For Web/Desktop chat, set `GATEWAY_BROWSER_TOOL=1` in the Gateway environment
and restart it with `agent-browser` on `PATH`. It is off by default. Each
account's chat uses a separate headless tool-browser session without personal
CLI settings or profile restore. The right panel shows URLs reported by
`browser_observe` and `browser_act` in a dedicated tab over the chat's event
stream; it does not load them until the user clicks "Open here". The Gateway
closes each session's daemon when its Agent leaves the cache or the Gateway
shuts down. It is still an independent viewer: cookies, page state and input
actions are not shared, and iframe embedding restrictions still apply on Web.

---

## 2. Pick the right shell driver

Probe what is installed before your first call (`command -v agent-browser bsk lightpanda playwright-cli`):

| Task | Tool to use | Why |
|---|---|---|
| **Local dev preview (`localhost`), UI verification, public interactive pages** | `agent-browser` (default) | Fast Rust CLI + background daemon; `snapshot -i` returns compact `@e1` refs |
| **Fast read-only extraction on JS/SSR pages (no pixels needed)** | `agent-browser --engine lightpanda` or `lightpanda fetch --dump markdown <url>` | Zig engine, ~10x faster cold start and ~16x less memory than Chrome; no screenshots or extensions |
| **Sites requiring the user's existing logins / SSO / intranet (Feishu, GitHub, admin consoles)** | `bsk` (BrowserSkill) | Connects to the user's running Chrome/Edge in a separate orange Agent Window; shares cookies without hijacking their active tab; hands CAPTCHAs/2FA back to the user |
| **Fallback when `agent-browser` is absent, or exporting E2E tests** | `playwright-cli` | Session-persistent (`-s=<name>`) Playwright runner |

---

## 3. Clean / headless browser (`agent-browser`)

Chain commands in a single `execute` call when the steps do not depend on
reading intermediate refs; otherwise follow **open → `snapshot -i` → act on `@ref` → verify → `close`**.

Always pass `-i` (interactive-only) to `snapshot` unless you need static prose
— full snapshots waste tokens and inflate step counts. Refs (`@e1`, `@e2`)
belong to the snapshot that produced them; after any navigation or DOM change,
take a fresh `snapshot -i` before clicking the next ref.

```sh
# Open and read interactive refs
agent-browser open http://127.0.0.1:8881 && agent-browser snapshot -i

# Act on refs from the snapshot, then check console/errors
agent-browser click @e3 && agent-browser fill @e4 "hello" && agent-browser press Enter
agent-browser console && agent-browser errors
agent-browser screenshot /tmp/ui-check.png

# Always close when finished
agent-browser close
```

Command notes verified against agent-browser 0.41.2:
- `get text` requires a selector (`agent-browser get text body`); for a plain
  body read use `agent-browser eval "document.body.innerText"`.
- `snapshot -i` lines look like `- link "Learn more" [ref=e1]`; pass the ref
  back as `@e1`.
- `fill`/`type` take a ref plus text; `keyboard inserttext <text>` inserts a
  whole string without key events.
- Chrome version advisories are printed to stderr; they are not page content.

When `lightpanda` is on `PATH` and you do not need screenshots, visual layout,
or Chrome extensions, set `AGENT_BROWSER_ENGINE=lightpanda` (or run
`lightpanda fetch --dump markdown <url>` for one-shot reading).

### Fallback: `playwright-cli`

```sh
playwright-cli -s=dev open http://127.0.0.1:8881
playwright-cli -s=dev snapshot
playwright-cli -s=dev click e2
playwright-cli -s=dev close
```

---

## 4. Logged-in state — measured 2026-11-21

Measured on macOS with agent-browser 0.41.2 against a throwaway local HTTP
server that sets `sid=abc123; Max-Age=86400`. Every path below was actually
run, not inferred from `--help`.

| Path | Command | Measured | When to use |
|---|---|---|---|
| `--profile <dir>` | `agent-browser --profile /path/to/profile --session X open <url>` | **PASS**: cookie survives `close` + a new daemon | Dedicated persistent profile per workspace |
| `--restore` | `agent-browser --session X --restore open <url>` | **PASS**: cookie restored on reopen (`restore: loaded`) | Easiest default; state in `~/.agent-browser` |
| `state load` | `agent-browser state load ./auth.json` then `reload` | **PASS** | Portable, explicit auth state file |
| `--state <file>` | `agent-browser --state ./auth.json open <url>` | **FAIL**: lands on `about:blank`, URL navigation is ignored; a second `open` still showed an empty body | Do not use for navigation |
| `--auto-connect` | `agent-browser --auto-connect open <url>` | **FAIL**: `No running Chrome instance found. Launch Chrome with --remote-debugging-port or use --cdp.` | Only when Chrome already runs with a debug port |
| `connect <port>` | `agent-browser connect 9222` | **FAIL** when nothing listens on 9222 | Same prerequisite |
| `bsk` | see §5 | **Not installed on this machine** | Needs the BrowserSkill extension |

### 4.1 `--profile` caveat (measured)

A **session cookie** (`Set-Cookie: sid=abc123; Path=/`, no `Max-Age`) was
dropped on `close` even with `--profile`. Adding `Max-Age` made the cookie
survive. Real login cookies are usually persistent, but do not assume it:
verify with `agent-browser cookies` after reopening.

Pick one persistence mechanism at a time — `--profile <dir>` and `--restore`
were verified independently, not combined.

### 4.2 Recommended sequences

```sh
# A) dedicated profile (isolated directory, nothing touches the user's Chrome)
agent-browser --profile /path/to/profile --session work open https://example.com/login
# ... user signs in once ...
agent-browser --profile /path/to/profile --session work close
# Later, same flags: the login state is still there.

# B) automatic restore (lightest)
agent-browser --session work --restore open https://example.com

# C) portable auth state
agent-browser --session work state save ./auth-state.json
agent-browser --session work2 open https://example.com
agent-browser --session work2 state load ./auth-state.json
agent-browser --session work2 reload
agent-browser --session work2 state clear --all   # cleanup when done
```

`auth-state.json` contains raw cookies. Keep it out of the repo, and set
`AGENT_BROWSER_ENCRYPTION_KEY=$(openssl rand -hex 32)` to encrypt saved states.

### 4.3 If you must use the user's own Chrome

`--auto-connect` / `--cdp` require Chrome started with
`--remote-debugging-port=9222`; a normal Chrome launch has no debug port and
both commands fail as shown above. Starting Chrome with a debug port on the
user's real profile is a security decision — ask before doing it. Otherwise
use `bsk` (§5), which goes through an extension instead.

---

## 5. Logged-in user browser (`bsk` / BrowserSkill)

Use `bsk` when the user wants work done inside accounts they are already
signed into in Chrome or Edge. On this machine `bsk` is **not installed**; if
it is missing, fall back to §4.2 A/B with a dedicated profile that the user
signs into once.

### Process-group rule (critical)

Each foreground `execute` call kills its process group when the command
returns. If a `bsk` command auto-starts the `bsk` daemon as a child, that
daemon is killed as soon as the `execute` call finishes.

1. Check status without auto-starting:
   ```sh
   BSK_AUTO_START=0 bsk doctor
   ```
2. If the daemon is not running, start it in a **managed background task** so
   it survives across turns, then wait a moment and re-check:
   ```
   execute(command="bsk daemon start --foreground", background=True)
   ```
3. If `extension connected` is still failing in `bsk doctor`, stop and ask the
   user to open the BrowserSkill extension popup in Chrome/Edge and enable the
   local connection.
4. Run every command with `BSK_AUTO_START=0` and always stop the session in
    cleanup (even on failure) so borrowed tabs return to the user's window:
   ```sh
   export BSK_AUTO_START=0
   bsk session start --json          # -> save session_id, e.g. ab12
   bsk navigate https://example.com --session ab12
   bsk observe --session ab12
   bsk click @e1 --session ab12
   bsk session stop ab12
   ```

Never run `bsk evaluate` on banking, SSO, or password-manager pages to read
tokens or secrets, and never borrow a user's existing tab unless the task
explicitly requires that open tab.

---

## 6. Hygiene

- `agent-browser doctor` on 2026-11-21 reported Chrome for Testing 157 while
  stable was 158 (295 CVEs). Run `agent-browser install` to refresh.
- `close` is mandatory in `finally`: whether the task succeeds or fails
  midway, end the session to avoid zombie daemons and leaked Chromium.
- `wait --load networkidle` can hang on SPAs that never go idle; fall back to
  `wait --load load` or skip waiting and snapshot directly.
- Prefer `snapshot -i` for interaction; do not guess CSS selectors blindly.
- Do not read or replay auth state files from untrusted paths.
