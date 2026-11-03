"use strict";
/**
 * Put an `agentica` command on the user's terminal, backed by the interpreter
 * that shipped inside the app. Pure Node (no Electron) so it runs under
 * `node --test`.
 *
 * The wrapper calls the bundled Python directly instead of copying a console
 * script: pip writes the build machine's absolute interpreter path into those.
 */

const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { CLI_ENTRY, bundledPython } = require("./runtime");

const MARKER = "agentica-desktop-shim";

function shimText(python, platform = process.platform) {
  if (platform === "win32") {
    return `@echo off\r\nrem ${MARKER}\r\n"${python}" -c "${CLI_ENTRY}" %*\r\n`;
  }
  return `#!/bin/sh\n# ${MARKER}\nexec "${python}" -c '${CLI_ENTRY}' "$@"\n`;
}

/** `current` PATH value with `dir` appended once, using `sep` as separator. */
function withPathEntry(current, dir, sep = path.delimiter) {
  const entries = current.split(sep).filter(Boolean);
  const same = (a, b) => (process.platform === "win32" ? a.toLowerCase() === b.toLowerCase() : a === b);
  if (entries.some((entry) => same(entry, dir))) return current;
  return [...entries, dir].join(sep);
}

function pathHas(dir) {
  return (process.env.PATH || "").split(path.delimiter).includes(dir);
}

function writableDir(dir) {
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.accessSync(dir, fs.constants.W_OK);
    return true;
  } catch {
    return false;
  }
}

/** Where the command goes: the system location when we may write it, else the user's. */
function shimTarget(platform = process.platform) {
  if (platform === "win32") {
    const local = process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local");
    return path.join(local, "Agentica", "bin", "agentica.cmd");
  }
  if (platform === "darwin" && writableDir("/usr/local/bin")) {
    return "/usr/local/bin/agentica";
  }
  return path.join(os.homedir(), ".local", "bin", "agentica");
}

function addToUserPathWindows(dir) {
  const script = [
    "$p = [Environment]::GetEnvironmentVariable('Path','User')",
    "if ($null -eq $p) { $p = '' }",
    `$d = '${dir.replace(/'/g, "''")}'`,
    "if (-not (($p -split ';') -contains $d)) {",
    "  [Environment]::SetEnvironmentVariable('Path', ($p.TrimEnd(';') + ';' + $d).TrimStart(';'), 'User')",
    "}",
  ].join("\n");
  execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
    stdio: "ignore",
    timeout: 15_000,
  });
}

/**
 * Write the wrapper. Refuses to replace a file it did not write, so a user's
 * own `agentica` (pip, uv tool) is never clobbered.
 */
function installCommand() {
  const python = bundledPython();
  if (!python) {
    throw new Error("This build has no bundled Python; install the CLI with `pip install agentica`.");
  }
  const target = shimTarget();
  if (fs.existsSync(target) && !fs.readFileSync(target, "utf8").includes(MARKER)) {
    throw new Error(`${target} already exists and was not created by Agentica Desktop.`);
  }
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, shimText(python), { mode: 0o755 });
  const dir = path.dirname(target);
  if (process.platform === "win32") {
    addToUserPathWindows(dir);
    return { target, onPath: true, restart: true };
  }
  return { target, onPath: pathHas(dir), restart: false };
}

module.exports = { MARKER, shimText, shimTarget, withPathEntry, installCommand };
