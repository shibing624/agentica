"use strict";
/**
 * Where the desktop app finds a gateway, and how it installs one when the
 * machine has none.
 *
 * The installer is a shell: it does not carry Python. A user who already has
 * `agentica-gateway` on their login-shell PATH (they `pip install`ed, or they
 * started the CLI) keeps that one — attach-before-spawn still holds. A user
 * who double-clicked the dmg with no Python at all gets a managed runtime
 * next to the app's support files, installed once with uv, never inside
 * `~/.agentica` (the agent writes there).
 *
 * Resolution order:
 *
 *   1. AGENTICA_GATEWAY_BIN
 *   2. the interpreter bundled in the installer (`<resources>/runtime/python`),
 *      same version as the shell, no network and no system Python needed
 *   3. login-shell `agentica-gateway` (unless AGENTICA_DESKTOP_IGNORE_EXISTING)
 *   4. managed venv (`<support>/venv/bin/agentica-gateway`)
 *   5. login-shell python that can `import agentica_gateway` (or, for an
 *      install older than the split, `agentica.gateway`)
 *   6. first-launch bootstrap (uv → CPython 3.12 → pip install agentica[gateway])
 *
 * Only 3-6 exist for `npm start` and for installers built without a runtime.
 *
 * Pure Node, no Electron — tests run under `node --test`.
 */

const { execFileSync, spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const PYTHON_VERSION = "3.12";
const BOOTSTRAP_UV_MS = 120_000;
const BOOTSTRAP_PYTHON_MS = 180_000;
const BOOTSTRAP_PIP_MS = 180_000;

function supportRoot() {
  if (process.env.AGENTICA_DESKTOP_RUNTIME) {
    return process.env.AGENTICA_DESKTOP_RUNTIME;
  }
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Application Support", "Agentica");
  }
  if (process.platform === "win32") {
    const local = process.env.LOCALAPPDATA
      || path.join(os.homedir(), "AppData", "Local");
    return path.join(local, "Agentica");
  }
  const data = process.env.XDG_DATA_HOME
    || path.join(os.homedir(), ".local", "share");
  return path.join(data, "agentica");
}

function uvBin(root = supportRoot()) {
  const name = process.platform === "win32" ? "uv.exe" : "uv";
  return path.join(root, "bin", name);
}

function venvDir(root = supportRoot()) {
  return path.join(root, "venv");
}

function venvPython(root = supportRoot()) {
  return process.platform === "win32"
    ? path.join(venvDir(root), "Scripts", "python.exe")
    : path.join(venvDir(root), "bin", "python");
}

function managedGatewayBin(root = supportRoot()) {
  return process.platform === "win32"
    ? path.join(venvDir(root), "Scripts", "agentica-gateway.exe")
    : path.join(venvDir(root), "bin", "agentica-gateway");
}

function managedBinDir(root = supportRoot()) {
  return process.platform === "win32"
    ? path.join(venvDir(root), "Scripts")
    : path.join(venvDir(root), "bin");
}

function isExecutable(file) {
  try {
    fs.accessSync(file, fs.constants.X_OK);
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

function loginShellWhich(command) {
  if (process.platform === "win32") {
    try {
      const out = execFileSync("where", [command], {
        encoding: "utf8",
        timeout: 8_000,
        stdio: ["ignore", "pipe", "ignore"],
      });
      return out.trim().split(/\r?\n/).find(Boolean) || null;
    } catch {
      return null;
    }
  }
  const shell = process.env.SHELL || "/bin/sh";
  try {
    const out = execFileSync(shell, ["-ilc", `command -v ${command}`], {
      encoding: "utf8",
      timeout: 8_000,
      stdio: ["ignore", "pipe", "ignore"],
    });
    return out.trim().split("\n").pop().trim() || null;
  } catch {
    return null;
  }
}

function ignoreExisting() {
  return process.env.AGENTICA_DESKTOP_IGNORE_EXISTING === "1";
}

const GATEWAY_ENTRY = "from agentica_gateway.main import main; main()";
const CLI_ENTRY = "import sys; sys.argv[0] = 'agentica'; from agentica.cli.main import main; sys.exit(main())";

/** Directory the installer unpacked next to the app: `<resources>/runtime`. */
function bundledRoot() {
  if (process.env.AGENTICA_DESKTOP_BUNDLED_RUNTIME) {
    return process.env.AGENTICA_DESKTOP_BUNDLED_RUNTIME;
  }
  return process.resourcesPath ? path.join(process.resourcesPath, "runtime") : null;
}

function bundledPython(root = bundledRoot()) {
  if (!root) return null;
  const python = process.platform === "win32"
    ? path.join(root, "python", "python.exe")
    : path.join(root, "python", "bin", "python3");
  return isExecutable(python) ? python : null;
}

/**
 * The interpreter that shipped inside the installer, with agentica already
 * installed into it. It is the shell's own version of the gateway, so the two
 * never drift; `PYTHONNOUSERSITE` keeps a user's `pip install --user` packages
 * from shadowing what was shipped.
 */
function resolveBundledCommand() {
  const python = bundledPython();
  if (!python) return null;
  return {
    command: python,
    args: ["-c", GATEWAY_ENTRY],
    source: "bundled",
    pathPrefix: path.dirname(python),
    env: { PYTHONNOUSERSITE: "1" },
  };
}

/**
 * Gateway module a Python can run: the `agentica_gateway` distribution, or the
 * `agentica.gateway` subpackage of an agentica older than the split (1.5.0 and
 * earlier). Probed in that order, in one interpreter start.
 */
const GATEWAY_MODULES = ["agentica_gateway.main", "agentica.gateway.main"];

function pythonWithGateway() {
  const py = loginShellWhich("python3") || loginShellWhich("python");
  if (!py) return null;
  // find_spec on the `.main` submodule imports its package, so a package that
  // cannot run (agentica's deprecated alias with no agentica-gateway behind
  // it, or either one without fastapi) raises and is skipped.
  const probe = "import importlib.util, sys\n"
    + `for m in ${JSON.stringify(GATEWAY_MODULES)}:\n`
    + "    try:\n"
    + "        if importlib.util.find_spec(m):\n"
    + "            print(m); sys.exit(0)\n"
    + "    except Exception:\n"
    + "        pass\n"
    + "sys.exit(1)";
  try {
    const out = execFileSync(py, ["-W", "ignore", "-c", probe], { timeout: 8_000, encoding: "utf8" }).trim();
    const module = out.split("\n").pop();
    return GATEWAY_MODULES.includes(module) ? { python: py, module } : null;
  } catch {
    return null;
  }
}

/**
 * A gateway that is already on the machine. Does not download anything.
 * Returns `{ command, args, source, pathPrefix }` or null.
 */
function resolveExistingCommand() {
  if (process.env.AGENTICA_GATEWAY_BIN) {
    return {
      command: process.env.AGENTICA_GATEWAY_BIN,
      args: [],
      source: "env",
      pathPrefix: null,
    };
  }
  const bundled = resolveBundledCommand();
  if (bundled) return bundled;
  if (!ignoreExisting()) {
    const bin = loginShellWhich("agentica-gateway");
    if (bin) {
      return { command: bin, args: [], source: "path", pathPrefix: null };
    }
  }
  const managed = managedGatewayBin();
  if (isExecutable(managed)) {
    return {
      command: managed,
      args: [],
      source: "managed",
      pathPrefix: managedBinDir(),
    };
  }
  if (!ignoreExisting()) {
    const found = pythonWithGateway();
    if (found) {
      return {
        command: found.python,
        args: ["-m", found.module],
        source: "python",
        pathPrefix: null,
      };
    }
  }
  return null;
}

function run(command, args, { cwd, env, timeout, log } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: env || process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let out = "";
    const take = (d) => {
      const text = String(d);
      out += text;
      if (log) {
        const line = text.trimEnd();
        if (line) log(line);
      }
    };
    child.stdout.on("data", take);
    child.stderr.on("data", take);
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`timed out after ${timeout}ms: ${command} ${args.join(" ")}`));
    }, timeout || 60_000);
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on("exit", (code, signal) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve(out);
        return;
      }
      reject(new Error(
        `${command} ${args.join(" ")} exited ${code ?? signal}\n${out.slice(-2500).trim()}`,
      ));
    });
  });
}

// Download the installer to a file, then run it. A pipe (`curl | sh`) reports
// the shell's status: with no network curl writes nothing and exits non-zero,
// and `sh` still exits 0 on empty stdin. The next check then says the
// installer "finished" while the binary was never downloaded.
const UV_UNIX_INSTALL = [
  "set -eu",
  "f=$(mktemp)",
  "trap 'rm -f \"$f\"' EXIT",
  "curl -LsSf https://astral.sh/uv/install.sh -o \"$f\"",
  "sh \"$f\"",
].join("\n");

function uvInstallPlan(root) {
  const dest = uvBin(root);
  const installDir = path.dirname(dest);
  if (process.platform === "win32") {
    return {
      dest,
      command: "powershell.exe",
      args: [
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
        "irm https://astral.sh/uv/install.ps1 | iex",
      ],
      env: {
        UV_INSTALL_DIR: installDir,
        UV_UNMANAGED_INSTALL: installDir,
      },
    };
  }
  // UV_UNMANAGED_INSTALL is the directory the binary is written into.
  // UV_INSTALL_DIR is the parent: the installer appends /bin itself, so
  // setting it to installDir would land uv at <root>/bin/bin/uv.
  return {
    dest,
    command: "/bin/sh",
    args: ["-c", UV_UNIX_INSTALL],
    env: { UV_UNMANAGED_INSTALL: installDir },
  };
}

async function ensureUv(root, log) {
  const plan = uvInstallPlan(root);
  if (isExecutable(plan.dest)) return plan.dest;
  fs.mkdirSync(path.dirname(plan.dest), { recursive: true });
  log(`installing uv into ${path.dirname(plan.dest)}`);
  await run(plan.command, plan.args, {
    timeout: BOOTSTRAP_UV_MS,
    log,
    env: { ...process.env, ...plan.env },
  });
  if (!isExecutable(plan.dest)) {
    throw new Error(`uv installer finished but ${plan.dest} is missing`);
  }
  return plan.dest;
}

// `agentica[gateway]` rather than `agentica-gateway`: it resolves correctly on
// both sides of the 1.5.1 split (before it, the gateway was that extra; after,
// the extra pins the matching agentica-gateway with Feishu, as it always had).
function packageSpec() {
  return process.env.AGENTICA_DESKTOP_PACKAGE || "agentica[gateway]";
}

function maybeLinkShims(root, log) {
  if (process.platform === "win32") return;
  const localBin = path.join(os.homedir(), ".local", "bin");
  try {
    fs.mkdirSync(localBin, { recursive: true });
  } catch (err) {
    log(`could not create ${localBin}: ${err.message}`);
    return;
  }
  for (const name of ["agentica", "agentica-gateway"]) {
    const src = path.join(venvDir(root), "bin", name);
    const dest = path.join(localBin, name);
    if (!isExecutable(src)) continue;
    try {
      if (fs.existsSync(dest)) {
        const st = fs.lstatSync(dest);
        if (!st.isSymbolicLink()) continue;
        const cur = fs.readlinkSync(dest);
        const resolved = path.resolve(path.dirname(dest), cur);
        if (resolved === path.resolve(src)) continue;
        fs.unlinkSync(dest);
      }
      fs.symlinkSync(src, dest);
      log(`linked ${dest} → ${src}`);
    } catch (err) {
      log(`skip ${name} shim: ${err.message}`);
    }
  }
}

async function createVenvAndInstall(root, log) {
  const uv = uvBin(root);
  const venv = venvDir(root);
  const py = venvPython(root);
  const pkg = packageSpec();
  log(`installing CPython ${PYTHON_VERSION}`);
  await run(uv, ["python", "install", PYTHON_VERSION], {
    timeout: BOOTSTRAP_PYTHON_MS,
    log,
  });
  log(`creating venv at ${venv}`);
  await run(uv, ["venv", venv, "--python", PYTHON_VERSION], {
    timeout: BOOTSTRAP_PYTHON_MS,
    log,
  });
  log(`pip install ${pkg}`);
  await run(uv, ["pip", "install", "--python", py, pkg], {
    timeout: BOOTSTRAP_PIP_MS,
    log,
  });
}

/**
 * Install the managed runtime if it is missing. No-op when the gateway
 * binary is already there. Honours AGENTICA_DESKTOP_NO_BOOTSTRAP=1.
 */
async function ensureManagedRuntime({ log = () => {} } = {}) {
  if (process.env.AGENTICA_DESKTOP_NO_BOOTSTRAP === "1") {
    return null;
  }
  const root = supportRoot();
  const gateway = managedGatewayBin(root);
  if (isExecutable(gateway)) {
    return {
      command: gateway,
      args: [],
      source: "managed",
      pathPrefix: managedBinDir(root),
    };
  }
  log("no agentica-gateway on this machine; installing a managed runtime (once)");
  fs.mkdirSync(path.join(root, "bin"), { recursive: true });
  await ensureUv(root, log);
  await createVenvAndInstall(root, log);
  if (!isExecutable(gateway)) {
    throw new Error(
      `Managed runtime installed but ${gateway} is missing. `
      + `Try: pip install "agentica[gateway]"`,
    );
  }
  maybeLinkShims(root, log);
  return {
    command: gateway,
    args: [],
    source: "bootstrap",
    pathPrefix: managedBinDir(root),
  };
}

/**
 * Full resolution, including first-launch bootstrap when nothing else works.
 */
async function resolveCommand({ log = () => {} } = {}) {
  const existing = resolveExistingCommand();
  if (existing) return existing;
  const boot = await ensureManagedRuntime({ log });
  if (boot) return boot;
  return null;
}

/**
 * First launch with no gateway and no network. Shown in the window instead of
 * an error dialog: the AppImage catalog runs the app with `--net=none` and
 * fails the screenshot on phrases like "could not start". The words here are
 * the ones that page is allowed to say.
 */
function bootstrapBlockedPage() {
  return `<!doctype html>
<html><head><meta charset="utf-8"><title>Agentica</title></head>
<body style="margin:0;padding:36px 40px;font:15px/1.5 system-ui,-apple-system,sans-serif;color:#1a1a1a;background:#fff">
  <h1 style="font-size:18px;margin:0 0 12px">One-time setup</h1>
  <p style="margin:0 0 12px">Agentica is a window onto a local Python gateway. The first launch downloads that gateway: uv, Python 3.12, and the agentica package. This download did not finish, so the window has nothing to show yet.</p>
  <p style="margin:0 0 12px">Connect to the network and open Agentica again.</p>
  <p style="margin:0 0 8px">To install the gateway yourself, then open Agentica again, run:</p>
  <p style="margin:0;font-family:ui-monospace,monospace">pip install "agentica[gateway]"</p>
</body></html>`;
}

module.exports = {
  PYTHON_VERSION,
  UV_UNIX_INSTALL,
  CLI_ENTRY,
  bundledRoot,
  bundledPython,
  resolveBundledCommand,
  supportRoot,
  uvBin,
  uvInstallPlan,
  bootstrapBlockedPage,
  venvDir,
  venvPython,
  managedGatewayBin,
  managedBinDir,
  isExecutable,
  resolveExistingCommand,
  resolveCommand,
  ensureManagedRuntime,
};
