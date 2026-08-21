"use strict";
/**
 * Finding, starting and stopping the Python gateway.
 *
 * The shell owns no product logic — it owns a child process and a URL. Every
 * capability in the window arrives over HTTP from that child, exactly as it
 * does for a browser.
 *
 * Two rules shape this file:
 *
 *  - Attach before spawn. A user who already has `agentica-gateway` (or a CLI
 *    session's gateway) running on this data root must not get a second one:
 *    they share ~/.agentica, so two processes mean two cron tickers and two
 *    peers records fighting over the same files.
 *  - Never kill what we did not start. The teardown is gated on `spawned`, so
 *    closing the window cannot take down the terminal's gateway.
 */

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { choosePort, readPreferredPort, rememberPort } = require("./port-memory");
const { resolveCommand } = require("./runtime");

/** How long to wait for a freshly spawned gateway to answer. The first start
 *  loads the model config and builds an Agent, which is not instant. */
const START_TIMEOUT_MS = 45_000;
const POLL_MS = 250;
/** How long the gateway gets to wind itself down after being asked. */
const SHUTDOWN_GRACE_MS = 6_000;
/** Grace between SIGTERM and SIGKILL on the way out. */
const STOP_GRACE_MS = 5_000;

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

function runtimeFile() {
  // Mirrors agentica/config.py: AGENTICA_CACHE_DIR, else $AGENTICA_HOME/cache,
  // else ~/.agentica/cache. The shell has to agree with the Python side on
  // this path or "is one already running" is unanswerable.
  const cache =
    process.env.AGENTICA_CACHE_DIR ||
    path.join(process.env.AGENTICA_HOME || path.join(os.homedir(), ".agentica"), "cache");
  return path.join(cache, "gateway", "runtime.json");
}

function readRuntime() {
  try {
    const raw = JSON.parse(fs.readFileSync(runtimeFile(), "utf8"));
    if (!raw.port || !raw.pid) return null;
    return raw;
  } catch {
    return null;
  }
}

function pidAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e.code === "EPERM";
  }
}

async function health(record, timeoutMs = 1500) {
  // /api/health is deliberately outside the token gate: this probe runs before
  // the shell knows whether the record's token is still valid.
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(`${record.url}/api/health`, { signal: ctrl.signal });
    return resp.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The command that starts a gateway, or null when nothing on this machine can.
 *
 * A GUI app launched from Finder/Dock does not inherit the shell's PATH — it
 * gets a bare `/usr/bin:/bin:/usr/sbin:/sbin`, which is never where a conda or
 * venv `agentica-gateway` lives. Resolution asks the user's login shell, then
 * a managed runtime under Application Support, then (once) bootstraps one
 * with uv. AGENTICA_GATEWAY_BIN skips the search.
 */
// resolveCommand lives in runtime.js so the ladder can be tested without Electron.

class GatewayProcess {
  /**
   * @param log where the child's output goes
   * @param portFile where to remember the port it bound (see port-memory.js);
   *        null disables port stickiness, which only a test wants.
   */
  constructor(log = console.log, portFile = null) {
    this.log = log;
    this.portFile = portFile;
    this.record = null;
    this.child = null;
    this.spawned = false;
  }

  /** Attach to a live gateway, or start one. */
  async ensure() {
    const existing = readRuntime();
    if (existing && pidAlive(existing.pid) && (await health(existing))) {
      this.log(`attaching to gateway pid ${existing.pid} at ${existing.url}`);
      this.record = existing;
      this.spawned = false;
      return existing;
    }
    return this.start();
  }

  /**
   * Start a gateway and wait for it to answer.
   *
   * Called with the remembered port first. If that start dies — which is what
   * a lost race for the port looks like from here — it is retried once on 0:
   * the preferred port is a *preference*, and the shell is the one that
   * expressed it, so the shell is where the fallback belongs. The gateway
   * itself must keep failing loudly on a taken `--port 8881`, because a user
   * who named a port meant it.
   */
  async start() {
    const preferred = this.portFile ? readPreferredPort(this.portFile) : null;
    const port = await choosePort(preferred);
    try {
      return await this._spawnOn(port);
    } catch (err) {
      if (port === 0) throw err;
      this.log(`port ${port} did not work out (${err.message}); retrying on a free one`);
      return this._spawnOn(0);
    }
  }

  async _spawnOn(port) {
    const resolved = await resolveCommand({ log: this.log });
    if (!resolved) {
      throw new Error(
        "Cannot find agentica-gateway. The app installs a managed runtime on "
        + "first launch when the network is available; to install yourself run "
        + "`pip install \"agentica[gateway]\"`, or set AGENTICA_GATEWAY_BIN."
      );
    }

    // --port 0: the OS picks, the gateway publishes what it got. A fixed port
    // would collide with the user's own `agentica-gateway` on 8881, and the
    // collision surfaces as a blank window. A non-zero port here is last
    // launch's, so the window keeps its origin and therefore its localStorage.
    // --parent-pid: if this shell is SIGKILLed there is nobody left to stop
    // the child, and an orphan keeps the port and the session locks.
    const args = [
      ...resolved.args,
      "--port", String(port),
      "--parent-pid", String(process.pid),
    ];
    this.log(`starting ${resolved.command} ${args.join(" ")} (${resolved.source})`);

    const env = { ...process.env, PYTHONUNBUFFERED: "1" };
    if (resolved.pathPrefix) {
      env.PATH = `${resolved.pathPrefix}${path.delimiter}${env.PATH || ""}`;
    }
    const child = spawn(resolved.command, args, {
      // The gateway's default project directory is the cwd it was launched
      // from — which for an app started from the Dock is `/`, so the agent
      // would open on the filesystem root. A GUI launch has no meaningful cwd,
      // so home is the honest default; the user picks a project in the UI.
      cwd: os.homedir(),
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    this.child = child;
    this.spawned = true;

    let stderrTail = "";
    child.stdout.on("data", (d) => this.log(`[gateway] ${String(d).trimEnd()}`));
    child.stderr.on("data", (d) => {
      const text = String(d);
      stderrTail = (stderrTail + text).slice(-4000);
      this.log(`[gateway] ${text.trimEnd()}`);
    });

    let exited = null;
    child.on("exit", (code, signal) => {
      exited = { code, signal };
    });

    const deadline = Date.now() + START_TIMEOUT_MS;
    while (Date.now() < deadline) {
      if (exited) {
        // Dying at startup is the common failure (no API key, bad config), and
        // its explanation is in the child's stderr — not in a timeout message.
        throw new Error(
          `The gateway failed to start (exit ${exited.code ?? exited.signal})\n${stderrTail.trim()}`
        );
      }
      const record = readRuntime();
      // Match on our own pid: an unrelated stale record would otherwise be
      // read as "started" and the window would load a dead port.
      if (record && record.pid === child.pid && (await health(record))) {
        this.log(`gateway ready at ${record.url}`);
        this.record = record;
        if (this.portFile) rememberPort(this.portFile, record.port);
        return record;
      }
      await delay(POLL_MS);
    }
    await this.stop();
    throw new Error(
      `The gateway was not ready within ${START_TIMEOUT_MS / 1000}s\n${stderrTail.trim()}`
    );
  }

  /** Whether the child is still running. */
  get running() {
    return !!this.child && this.child.exitCode === null && this.child.signalCode === null;
  }

  /**
   * Stop the child, if it is ours. Safe to call twice, and safe to await.
   *
   * Three steps, because each one covers a case the next cannot:
   *
   *  1. `POST /api/desktop/shutdown` — the only graceful path on Windows,
   *     where `kill()` is a hard TerminateProcess: no channel disconnect, no
   *     peers record removed, no session flushed.
   *  2. SIGTERM — for a gateway too wedged to serve the endpoint.
   *  3. SIGKILL — for one that ignored the signal.
   *
   * Awaiting matters: `will-quit` used to fire this and return, so the app
   * exited while the child was still winding down and the SIGKILL timer went
   * with it. The `--parent-pid` watchdog was the only thing left, and it takes
   * two seconds to notice.
   */
  async stop() {
    const child = this.child;
    if (!child || !this.spawned || !this.running) return;
    this.log(`stopping gateway pid ${child.pid}`);

    const exited = new Promise((resolve) => child.once("exit", resolve));

    if (this.record) {
      try {
        await fetch(`${this.record.url}/api/desktop/shutdown`, {
          method: "POST",
          headers: this.record.token
            ? { authorization: `Bearer ${this.record.token}` }
            : {},
          signal: AbortSignal.timeout(3000),
        });
      } catch {
        // Unreachable or already gone: the signals below are the fallback.
      }
    }
    if (await raceExit(exited, SHUTDOWN_GRACE_MS)) return;

    try {
      child.kill("SIGTERM");
    } catch {
      return;
    }
    if (await raceExit(exited, STOP_GRACE_MS)) return;

    this.log(`gateway pid ${child.pid} ignored SIGTERM; killing`);
    try {
      child.kill("SIGKILL");
    } catch {
      /* already gone */
    }
    await raceExit(exited, 2000);
  }
}

/** True when the process exited within the budget. */
async function raceExit(exited, ms) {
  let done = false;
  await Promise.race([exited.then(() => { done = true; }), delay(ms)]);
  return done;
}

module.exports = { GatewayProcess, readRuntime, runtimeFile, pidAlive, health, resolveCommand };
