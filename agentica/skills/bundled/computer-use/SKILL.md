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

## Check the driver

The driver is [Peekaboo](https://github.com/openclaw/Peekaboo), macOS only
(`uname -s`). Look for it with `command -v peekaboo`, then check
`peekaboo permissions status --no-remote`. If it is missing or lacks a
permission, say what is missing and stop. Do not switch to OS input scripts
(`osascript`, `cliclick`) or another harness, and do not use another driver to
get around a refusal or a cancellation.

Install it only when the user asks: with Homebrew,
`brew install openclaw/tap/peekaboo`, then `peekaboo --version` and the
permission check. Ask the user to grant missing Accessibility or Screen
Recording access in System Settings; do not change those permissions for them.

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
agreed operation; stop on refusal or cancellation.

In the `auto` tier the first command that operates the desktop (click, type,
set-value, close a window) shows an approval card, and that one approval covers
desktop commands until the turn ends. Observing (`see`, `window list`) shows
none. Run `peekaboo` directly in `execute`, not from a script file, so the card
shows what will happen. Stay inside the window the user asked about: the card
is consent for this task, not for other apps.
