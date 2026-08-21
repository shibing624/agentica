"use strict";

const { test, beforeEach, afterEach } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {
  supportRoot,
  uvBin,
  venvDir,
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
