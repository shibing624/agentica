"use strict";

const { test, beforeEach, afterEach } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const {
  supportRoot,
  uvBin,
  venvDir,
  uvInstallPlan,
  UV_UNIX_INSTALL,
  bootstrapBlockedPage,
  managedGatewayBin,
  managedBinDir,
  resolveExistingCommand,
} = require("./runtime");

const KEYS = [
  "AGENTICA_DESKTOP_RUNTIME",
  "AGENTICA_GATEWAY_BIN",
  "AGENTICA_DESKTOP_IGNORE_EXISTING",
  "AGENTICA_DESKTOP_NO_BOOTSTRAP",
];

let saved = {};
let tmp = null;

beforeEach(() => {
  saved = {};
  for (const key of KEYS) {
    saved[key] = process.env[key];
    delete process.env[key];
  }
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agentica-runtime-"));
  process.env.AGENTICA_DESKTOP_RUNTIME = tmp;
});

afterEach(() => {
  for (const key of KEYS) {
    if (saved[key] === undefined) delete process.env[key];
    else process.env[key] = saved[key];
  }
  fs.rmSync(tmp, { recursive: true, force: true });
});

test("supportRoot honours AGENTICA_DESKTOP_RUNTIME", () => {
  assert.equal(supportRoot(), tmp);
  assert.equal(uvBin(), path.join(tmp, "bin", process.platform === "win32" ? "uv.exe" : "uv"));
  assert.equal(venvDir(), path.join(tmp, "venv"));
});

test("AGENTICA_GATEWAY_BIN wins over everything", () => {
  process.env.AGENTICA_GATEWAY_BIN = "/opt/custom/agentica-gateway";
  process.env.AGENTICA_DESKTOP_IGNORE_EXISTING = "1";
  const got = resolveExistingCommand();
  assert.equal(got.source, "env");
  assert.equal(got.command, "/opt/custom/agentica-gateway");
  assert.deepEqual(got.args, []);
});

test("managed venv is used when PATH lookup is skipped", () => {
  process.env.AGENTICA_DESKTOP_IGNORE_EXISTING = "1";
  const bin = managedGatewayBin();
  fs.mkdirSync(path.dirname(bin), { recursive: true });
  fs.writeFileSync(bin, "#!/bin/sh\n");
  fs.chmodSync(bin, 0o755);
  const got = resolveExistingCommand();
  assert.equal(got.source, "managed");
  assert.equal(got.command, bin);
  assert.equal(got.pathPrefix, managedBinDir());
});

test("nothing on the machine returns null (no bootstrap)", () => {
  process.env.AGENTICA_DESKTOP_IGNORE_EXISTING = "1";
  assert.equal(resolveExistingCommand(), null);
});

test("uv install does not treat a failed download as success", () => {
  const plan = uvInstallPlan(tmp);
  assert.equal(plan.dest, uvBin(tmp));
  if (process.platform === "win32") {
    assert.equal(plan.env.UV_INSTALL_DIR, path.dirname(plan.dest));
    return;
  }
  // The installer writes the binary into this directory. UV_INSTALL_DIR
  // would append another /bin.
  assert.equal(plan.env.UV_UNMANAGED_INSTALL, path.dirname(plan.dest));
  assert.equal(plan.env.UV_INSTALL_DIR, undefined);
  assert.equal(plan.args[1], UV_UNIX_INSTALL);
  assert.doesNotMatch(UV_UNIX_INSTALL, /\|/);
  const bin = fs.mkdtempSync(path.join(os.tmpdir(), "agentica-curl-"));
  try {
    fs.writeFileSync(path.join(bin, "curl"), "#!/bin/sh\nexit 7\n");
    fs.chmodSync(path.join(bin, "curl"), 0o755);
    assert.throws(
      () => execFileSync("/bin/sh", ["-c", UV_UNIX_INSTALL], {
        env: { ...process.env, PATH: `${bin}${path.delimiter}${process.env.PATH}` },
        stdio: "pipe",
      }),
    );
  } finally {
    fs.rmSync(bin, { recursive: true, force: true });
  }
});

// AppImageHub's check-screenshot.sh. A hit on HARD fails the catalog run;
// SOFT is only a warning, and the setup page should not trip that either.
const SCREENSHOT_HARD = /traceback|exception|segmentation fault|fatal|error while loading|glibc|not installed|cannot open display|permission denied|no such file|could not (load|find|open|start|initiali)|failed to (load|start|open|initiali|create)|cannot (load|find|open|execute|configure)|unable to (load|find|open|start)|command not found|core dumped/i;
const SCREENSHOT_SOFT = /error|failed|failure|could not|cannot|unable to|not found/i;

test("offline first-launch page does not look like an error dialog", () => {
  const text = bootstrapBlockedPage().replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
  assert.doesNotMatch(text, SCREENSHOT_HARD);
  assert.doesNotMatch(text, SCREENSHOT_SOFT);
  const words = text.toLowerCase().split(/[^a-z]+/).filter((w) => w.length >= 3);
  assert.ok(words.length >= 5, `expected a readable page, got ${words.length} words`);
});
