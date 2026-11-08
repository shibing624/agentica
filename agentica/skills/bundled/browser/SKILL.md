---
name: browser
description: >-
  Drive a real or headless browser from the shell to verify web UI changes,
  click through flows, inspect console/network errors, or act inside the
  user's logged-in Chrome/Edge session. Use when fetch_url is not enough —
  SPAs, localhost dev servers, form interactions, visual checks, or
  authenticated sites (via agent-browser, lightpanda, bsk BrowserSkill, or
  playwright-cli).
metadata:
  version: "1.0"
---

# Browser automation from the shell

There is no dedicated browser tool in the schema on purpose: browser CLIs keep
a warm daemon between shell calls and cost zero tool-schema tokens on turns
that do not touch a browser. Drive them through `execute`.

If you only need to read a public static page, use `fetch_url` first. Reach
for a browser when the page needs JavaScript, you are verifying a UI you just
edited, or the task requires clicking, filling forms, screenshots, or the
user's existing login state.

## 1. Pick the right driver

Probe what is installed before your first call (`command -v agent-browser bsk lightpanda playwright-cli`):

| Task | Tool to use | Why |
|---|---|---|
| **Local dev preview (`localhost`), UI verification, public interactive pages** | `agent-browser` (default) | Fast Rust CLI + background daemon; `snapshot -i` returns compact `@e1` refs |
| **Fast read-only extraction on JS/SSR pages (no pixels needed)** | `agent-browser --engine lightpanda` or `lightpanda fetch --dump markdown <url>` | Zig engine, ~10x faster cold start and ~16x less memory than Chrome; no screenshots or extensions |
| **Sites requiring the user's existing logins / SSO / intranet (Feishu, GitHub, admin consoles)** | `bsk` (BrowserSkill) | Connects to the user's running Chrome/Edge in a separate orange Agent Window; shares cookies without hijacking their active tab; hands CAPTCHAs/2FA back to the user |
| **Fallback when `agent-browser` is absent, or exporting E2E tests** | `playwright-cli` | Session-persistent (`-s=<name>`) Playwright runner |

---

## 2. Clean / headless browser (`agent-browser`)

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

## 3. Logged-in user browser (`bsk` / BrowserSkill)

Use `bsk` when the user wants work done inside accounts they are already
signed into in Chrome or Edge.

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
