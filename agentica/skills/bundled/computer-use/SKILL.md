---
name: computer-use
description: >-
  Use when a user asks a local interactive macOS session to inspect or operate
  a native app window, such as a music player, spreadsheet, or document editor.
---

# Computer use on macOS

Use this workflow only for a person’s request in a local interactive desktop
session, not for a remote gateway, IM, headless run, or scheduled job. Prefer a
purpose-built app API, connector, or CLI when it can finish the task without
touching the desktop. Use the `browser` skill for ordinary browser tasks.

## Choose a driver

1. On macOS (`uname -s`), check `/Applications/ChatGPT.app`, then Spotlight
   (`mdfind 'kMDItemCFBundleIdentifier == "com.openai.codex"'`) for an alternate
   install of the official desktop app. The `codex` executable alone is not the
   desktop app. On other platforms, report that this workflow is unavailable.
2. If the desktop app exists, prefer [LCU](https://github.com/amontlabs/lcu).
   Check `~/.local/share/lcu/current/bin/lcu` or `command -v lcu`, then run
   `lcu doctor` using the resolved path if readiness is unknown. Use an already
   configured local LCU adapter only if it preserves the runtime's instructions
   and images, forwards native user approvals, and cleans up at turn end.
   Installation, `doctor`, or a generic MCP connection alone is not an adapter.
   If that adapter is missing, say so and use Peekaboo if installed. Do not
   install or change other harnesses without a request. Do not switch drivers
   to get around a refusal, cancellation, or missing desktop permissions.
3. If the desktop app is absent, use Peekaboo if `command -v peekaboo` finds it.
   Check `peekaboo permissions status --no-remote`. If neither driver is ready,
   report what is missing rather than silently switching to OS input scripts.

## Install only when requested

Do not install dependencies just because a desktop task needs them. If the user
asks to install a missing driver, use its upstream route, then verify readiness:

- **Peekaboo:** If Homebrew is installed, run `brew install openclaw/tap/peekaboo`,
  then `peekaboo --version` and `peekaboo permissions status --no-remote`.
- **LCU:** Follow the [official installation guide](https://github.com/amontlabs/lcu/blob/main/docs/INSTALLATION.md).
  On Apple Silicon with the signed desktop app, verify the release archive's
  SHA-256 before running its `./scripts/install.sh --runtime-only`. Then run
  `~/.local/share/lcu/current/bin/lcu doctor`. This installs only the runtime;
  it does not configure an Agentica adapter. Do not select another harness on
  the user's behalf.

Ask the user to grant missing macOS Accessibility or Screen Recording access
in System Settings; do not change those permissions on their behalf.

## Peekaboo workflow

Discover the app by name and select a specific window. Ask before reading its
contents if the user has not already authorized that target; explain that a
window screenshot sent through `analyze_image` can reach the configured model
service. Authorize a specific window for the current request, not an app-wide
grant. Use fresh snapshots after changes, and don't store screenshots or raw
Base64 in the project or send them to a public file host.

Use existing `execute` calls. For example, to inspect an approved TextEdit
window, use `peekaboo window list --app=com.apple.TextEdit --json --no-remote`,
then `peekaboo see --app=com.apple.TextEdit --window-id=<listed-id> --fresh
--no-screenshot --tree --json --no-remote`. Request a window-only screenshot
with `--path` in a private temporary directory if the tree is insufficient.
For mutations, use the element ID from a fresh, reusable snapshot with
`--input-strategy=actionOnly`; disable automatic focus where supported. For
`set-value` use `--snapshot=<snapshot-id> --on=<element-id> --value=<text>
--input-strategy=actionOnly --no-auto-focus --no-remote --json`. Do not combine
its explicit snapshot with app or window selectors: the snapshot already binds
the target. For other actions, check the installed command’s `--help`. Verify
with a fresh observation before retrying a mutation with an unknown outcome.

If background operation cannot finish, explain which app needs the foreground
and what will happen. Ask once before using shared mouse/keyboard input or
switching focus, including a foreground app launch. Only proceed with that
agreed operation; stop on refusal or cancellation. This skill guides tool use;
it does not change CLI permission modes or sandbox the shell.