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
 *   2. login-shell `agentica-gateway` (unless AGENTICA_DESKTOP_IGNORE_EXISTING)
 *   3. managed venv (`<support>/venv/bin/agentica-gateway`)
 *   4. login-shell python that can `import agentica.gateway`
 *   5. first-launch bootstrap (uv → CPython 3.12 → pip install agentica[gateway])
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

function pythonWithGateway() {
  const py = loginShellWhich("python3") || loginShellWhich("python");
  if (!py) return null;
  try {
    execFileSync(
      py,
      ["-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('agentica.gateway') else 1)"],
      { timeout: 8_000, stdio: "ignore" },
    );
    return py;
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
    const py = pythonWithGateway();
    if (py) {
      return {
        command: py,
        args: ["-m", "agentica.gateway.main"],
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

async function ensureUv(root, log) {
  const dest = uvBin(root);
  if (isExecutable(dest)) return dest;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const installDir = path.dirname(dest);
  log(`installing uv into ${installDir}`);
  if (process.platform === "win32") {
    await run(
      "powershell.exe",
      [
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
        "irm https://astral.sh/uv/install.ps1 | iex",
      ],
      {
        timeout: BOOTSTRAP_UV_MS,
        log,
        env: {
          ...process.env,
          UV_INSTALL_DIR: installDir,
          UV_UNMANAGED_INSTALL: installDir,
        },
      },
    );
  } else {
    await run(
      "/bin/sh",
      ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
      {
        timeout: BOOTSTRAP_UV_MS,
        log,
        env: {
          ...process.env,
          UV_UNMANAGED_INSTALL: installDir,
        },
      },
    );
  }
  if (!isExecutable(dest)) {
    throw new Error(`uv installer finished but ${dest} is missing`);
  }
  return dest;
}

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

module.exports = {
  PYTHON_VERSION,
  supportRoot,
  uvBin,
  venvDir,
  venvPython,
  managedGatewayBin,
  managedBinDir,
  isExecutable,
  resolveExistingCommand,
  resolveCommand,
  ensureManagedRuntime,
};
