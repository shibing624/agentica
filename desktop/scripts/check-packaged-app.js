"use strict";

const assert = require("node:assert/strict");
const { execFile } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { promisify } = require("node:util");
const { desktopStatePaths } = require("../smoke");
const { GatewayProcess, health, pidAlive } = require("../gateway");
const { bundledPython } = require("../runtime");

async function main(executable, stateRoot, mode) {
  const exe = path.resolve(executable);
  const temporary = !stateRoot;
  const root = stateRoot || fs.mkdtempSync(path.join(os.tmpdir(), "agentica-packaged-smoke-"));
  const state = desktopStatePaths(root);
  const resources = process.platform === "darwin"
    ? path.resolve(path.dirname(exe), "..", "Resources")
    : path.resolve(path.dirname(exe), "resources");
  const runtime = path.join(resources, "runtime");
  assert.ok(fs.existsSync(runtime), `Bundled runtime missing: ${runtime}`);
  for (const key of Object.keys(process.env)) {
    if (/^(AGENTICA_|GATEWAY_|FEISHU_|TELEGRAM_|DISCORD_|QQ_|WECOM_|DINGTALK_|WECHAT_|SLACK_|PYTHONHOME$|PYTHONPATH$)/.test(key)) delete process.env[key];
  }
  Object.assign(process.env, {
    AGENTICA_DESKTOP_STATE_DIR: root,
    AGENTICA_HOME: state.home,
    AGENTICA_CACHE_DIR: path.join(state.home, "cache"),
    AGENTICA_BASE_DIR: state.home,
    AGENTICA_DESKTOP_RUNTIME: state.runtime,
    AGENTICA_DESKTOP_BUNDLED_RUNTIME: runtime,
    AGENTICA_DESKTOP_NO_BOOTSTRAP: "1",
    AGENTICA_DESKTOP_NO_UPDATE: "1",
    AGENTICA_DESKTOP_SMOKE: "1",
    AGENTICA_DESKTOP_IGNORE_EXISTING: "1",
    GATEWAY_AUTH: "true",
    HOST: "127.0.0.1",
  });
  fs.mkdirSync(state.home, { recursive: true });
  // The predecessor may not recognize AGENTICA_DESKTOP_STATE_DIR.
  const args = [`--user-data-dir=${state.profile}`];
  // Ubuntu 26 blocks unprivileged user namespaces, and the unpacked tree's
  // chrome-sandbox is not setuid (the deb postinst does that, the AppImage
  // passes --no-sandbox itself). Launching the raw binary then aborts in
  // about a second. Root cannot use the sandbox either.
  if (process.platform === "linux" && linuxSandboxBlocked(exe)) args.push("--no-sandbox");

  async function launch(spawned) {
    let stdout = "";
    try {
      ({ stdout } = await promisify(execFile)(exe, args, { timeout: 180_000, maxBuffer: 4 * 1024 * 1024 }));
    } catch (error) {
      // A non-zero exit still prints the readiness line on the smoke path.
      // The child's stderr stays out of this log: it can contain the token.
      stdout = typeof error.stdout === "string" ? error.stdout : "";
      if (!readSmokeResult(stdout)) throw error;
    }
    const result = readSmokeResult(stdout);
    assert.ok(result, "expected one readiness report");
    assert.equal(result.error, undefined, result.detail || result.error);
    assert.equal(result.probe.composer, true);
    assert.equal(result.probe.login, false);
    assert.equal(result.spawned, spawned);
    assert.equal(new URL(result.url).pathname, "/chat");
    return result;
  }

  const external = new GatewayProcess(() => {});
  try {
    if (mode === "seed") {
      const python = bundledPython(runtime);
      assert.ok(python, "Predecessor has no bundled interpreter");
      await promisify(execFile)(python, ["-c", [
        "import os",
        "from agentica.session.log import SessionLog",
        'log = SessionLog("desktop-upgrade-fixture", work_dir=os.environ["AGENTICA_BASE_DIR"], user_id="default")',
        'log.append("user", "Desktop upgrade fixture")',
        'log.set_source("gateway")',
      ].join("\n")], { timeout: 30_000 });
    }
    if (mode === "upgrade") {
      assert.ok(fs.existsSync(path.join(state.profile, "window-state.json")), "predecessor renderer state is missing before upgrade launch");
    }
    const owned = await launch(true);
    assert.ok(owned.gatewayPid);
    assert.equal(pidAlive(owned.gatewayPid), false, "desktop left its own gateway running");
    if (mode === "seed") {
      assert.ok(owned.probe.sessions >= 1, "predecessor did not load the fixture session");
      const marker = path.join(state.home, "desktop-upgrade-fixture");
      fs.writeFileSync(marker, owned.version || "predecessor", { flag: "wx" });
      console.log("Predecessor boot and shutdown passed");
      return;
    }
    const record = await external.start();
    const attached = await launch(false);
    assert.equal(attached.origin, record.url);
    assert.equal(attached.targetPid, record.pid);
    assert.ok(await health(record), "desktop stopped an externally owned gateway");
    if (mode === "upgrade") {
      assert.notEqual(fs.readFileSync(path.join(state.home, "desktop-upgrade-fixture"), "utf8"), owned.version,
        "upgrade requires a different predecessor version");
      assert.ok(owned.probe.sessions >= 1, "upgrade did not retain the fixture session");
    }
    console.log(`Desktop ${owned.version}: packaged boot, authenticated chat, owned shutdown and external attach passed`);
  } finally {
    await external.stop();
    if (temporary) fs.rmSync(root, { recursive: true, force: true });
  }
}

function linuxSandboxBlocked(exe) {
  if (typeof process.getuid === "function" && process.getuid() === 0) return true;
  try {
    return (fs.statSync(path.join(path.dirname(exe), "chrome-sandbox")).mode & 0o4000) === 0;
  } catch {
    return true;
  }
}

function readSmokeResult(stdout) {
  const prefix = "DESKTOP-SMOKE-RESULT ";
  const lines = String(stdout || "").split(/\r?\n/).filter((line) => line.startsWith(prefix));
  if (lines.length !== 1) return null;
  return JSON.parse(lines[0].slice(prefix.length));
}

function redact(text) {
  return String(text)
    .replace(/((?:token|password|secret|authorization)[=:]\s*)\S+/gi, "$1<redacted>")
    .replace(/[A-Za-z0-9_-]{40,}/g, "<redacted>");
}

main(process.argv[2], process.argv[3], process.argv[4]).catch((error) => {
  // Do not print child stdout/stderr: the gateway may log its local token.
  const result = readSmokeResult(error.stdout);
  if (result) {
    let pathname = "";
    try {
      pathname = result.url ? new URL(result.url).pathname : "";
    } catch {
      pathname = "";
    }
    console.error(redact(
      `smoke ${result.error || "mismatch"}`
      + ` detail=${result.detail || ""}`
      + ` spawned=${result.spawned} path=${pathname}`
      + ` composer=${result.probe && result.probe.composer}`
      + ` login=${result.probe && result.probe.login}`,
    ));
  } else if (error.stdout || error.stderr) {
    console.error(redact(String(error.message || error).split(/\r?\n/, 1)[0]));
  } else {
    console.error(redact(String(error.message || error).split(/\r?\n/).slice(0, 8).join("\n")));
  }
  process.exitCode = 1;
});