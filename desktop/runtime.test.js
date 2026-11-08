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
const {
  keepVisiblePosition, normalizeWindowState, readWindowState, writeWindowState,
} = require("./window-state");

const { shimText, withPathEntry } = require("./cli-shim");
const {
  buildDockMenuTemplate,
  formatSessionLabel,
  selectRecentSessions,
} = require("./dock-menu");
const { TARGETS, parseTargets, sitePackages } = require("./scripts/prepare-runtime");

const KEYS = [
  "AGENTICA_DESKTOP_BUNDLED_RUNTIME",
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

function fakeBundledPython() {
  const bundled = path.join(tmp, "resources", "runtime");
  const python = process.platform === "win32"
    ? path.join(bundled, "python", "python.exe")
    : path.join(bundled, "python", "bin", "python3");
  fs.mkdirSync(path.dirname(python), { recursive: true });
  fs.writeFileSync(python, "#!/bin/sh\n");
  fs.chmodSync(python, 0o755);
  process.env.AGENTICA_DESKTOP_BUNDLED_RUNTIME = bundled;
  return python;
}

test("bundled interpreter beats every system gateway", () => {
  const python = fakeBundledPython();
  const got = resolveExistingCommand();
  assert.equal(got.source, "bundled");
  assert.equal(got.command, python);
  assert.deepEqual(got.args.slice(0, 1), ["-c"]);
  assert.match(got.args[1], /agentica_gateway\.main import main/);
  assert.equal(got.env.PYTHONNOUSERSITE, "1");
});

test("AGENTICA_GATEWAY_BIN still overrides the bundled interpreter", () => {
  fakeBundledPython();
  process.env.AGENTICA_GATEWAY_BIN = "/opt/custom/agentica-gateway";
  assert.equal(resolveExistingCommand().source, "env");
});

test("a bundled runtime directory without python is ignored", () => {
  process.env.AGENTICA_DESKTOP_BUNDLED_RUNTIME = path.join(tmp, "empty");
  process.env.AGENTICA_DESKTOP_IGNORE_EXISTING = "1";
  assert.equal(resolveExistingCommand(), null);
});

test("terminal shim runs the bundled python and tags itself", () => {
  const unix = shimText("/Applications/Agentica.app/Contents/Resources/runtime/python/bin/python3", "darwin");
  assert.match(unix, /^#!\/bin\/sh\n# agentica-desktop-shim\n/);
  assert.match(unix, /exec "\/Applications\/.*python3" -c '.*agentica\.cli\.main import main.*' "\$@"/);
  const win = shimText("C:\\Agentica\\python.exe", "win32");
  assert.match(win, /agentica-desktop-shim/);
  assert.match(win, /%\*/);
});

test("PATH entries are appended once", () => {
  assert.equal(withPathEntry("/a:/b", "/c", ":"), "/a:/b:/c");
  assert.equal(withPathEntry("/a:/b", "/b", ":"), "/a:/b");
  assert.equal(withPathEntry("", "/c", ":"), "/c");
});

test("runtime recipes cover every installer target", () => {
  assert.deepEqual(Object.keys(TARGETS).sort(), ["linux-x64", "mac-arm64", "mac-x64", "win-x64"]);
  assert.deepEqual(parseTargets(["--target", "mac-x64", "--target", "win-x64"]), ["mac-x64", "win-x64"]);
  assert.throws(() => parseTargets(["--target", "plan9-mips"]));
  assert.match(sitePackages("/r/win-x64/python", "win-x64"), /Lib[\\/]site-packages$/);
  assert.match(sitePackages("/r/mac-arm64/python", "mac-arm64"), /lib[\\/]python3\.12[\\/]site-packages$/);
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

test("window state validates bounds and maximized flag", () => {
  assert.deepEqual(normalizeWindowState(null), { width: 1360, height: 880, maximized: false });
  assert.deepEqual(normalizeWindowState({ width: 10, height: 9000, maximized: "yes" }), {
    width: 1360, height: 880, maximized: false,
  });
  assert.deepEqual(normalizeWindowState({ x: -40, y: 80, width: 1200, height: 800, maximized: true }), {
    x: -40, y: 80, width: 1200, height: 800, maximized: true,
  });
});

test("window state survives an atomic round trip", () => {
  const file = path.join(tmp, "state", "window.json");
  const state = { x: 120, y: 90, width: 1440, height: 920, maximized: false };
  writeWindowState(file, state);
  assert.deepEqual(readWindowState(file), state);
  assert.equal(fs.existsSync(`${file}.tmp`), false);
});

test("window state recenters a window left on a disconnected display", () => {
  const displays = [{ workArea: { x: 0, y: 0, width: 1440, height: 900 } }];
  const offscreen = { x: -5000, y: 80, width: 1200, height: 800, maximized: false };
  assert.deepEqual(keepVisiblePosition(offscreen, displays), {
    width: 1200, height: 800, maximized: false,
  });
  const visible = { ...offscreen, x: -40 };
  assert.deepEqual(keepVisiblePosition(visible, displays), visible);
});

test("bundled runtime requirements pin both wheels of one build", () => {
  const { requirements } = require("./scripts/prepare-runtime.js");
  const reqs = requirements({ core: "/w/agentica-1.5.1-py3-none-any.whl", gateway: "/w/agentica_gateway-1.5.1-py3-none-any.whl" });
  assert.equal(reqs.length, 2);
  assert.match(reqs[0], /^agentica @ file:\/\/.*agentica-1\.5\.1-/);
  // Feishu ships in the app, as it did in agentica[gateway] before the split.
  assert.match(reqs[1], /^agentica-gateway\[feishu\] @ file:\/\/.*agentica_gateway-1\.5\.1-/);
});

test("dock menu sorts running sessions first, then newest, and drops archived", () => {
  const rows = [
    { session_id: "old", name: "Older chat", last_timestamp: "2026-11-01T10:00:00Z", work_dir: "/Users/me/proj-a" },
    { session_id: "arch", name: "Archived chat", last_timestamp: "2026-11-08T12:00:00Z", archived: true },
    { session_id: "new", name: "Newer chat", last_timestamp: "2026-11-05T10:00:00Z", work_dir: "/Users/me/proj-b" },
    { session_id: "run", name: "Running task", last_timestamp: "2026-11-02T10:00:00Z", running: true, work_dir: "/Users/me/agentica" },
  ];
  const picked = selectRecentSessions(rows, 8);
  assert.deepEqual(picked.map((r) => r.session_id), ["run", "new", "old"]);
  assert.equal(formatSessionLabel(picked[0]), "● Running task — agentica");
});

test("dock menu template shows current title, recent sessions with checkmark, and triggers callbacks", () => {
  const actions = [];
  const recent = [
    { session_id: "s1", name: "Stale server name", last_timestamp: "2026-11-08T10:00:00Z", work_dir: "/Users/me/agentica" },
    { session_id: "s2", name: "Fix search highlight", last_timestamp: "2026-11-07T10:00:00Z", running: true, work_dir: "/Users/me/web" },
  ];
  const tpl = buildDockMenuTemplate({
    current: { id: "s1", title: "Live renamed title", running: false },
    recent,
    lang: "zh-CN",
    onNewChat: () => actions.push("new"),
    onSelectSession: (id) => actions.push(`select:${id}`),
  });

  assert.equal(tpl[0].label, "新建对话");
  tpl[0].click();
  assert.equal(tpl[2].label, "当前打开：Live renamed title");
  tpl[2].click();
  assert.equal(tpl[4].label, "最近运行");
  assert.equal(tpl[4].enabled, false);

  // Running session s2 sorts first; current session s1 is checked and uses live title
  assert.equal(tpl[5].label, "● Fix search highlight — web");
  assert.equal(tpl[5].checked, false);
  assert.equal(tpl[6].label, "Live renamed title — agentica");
  assert.equal(tpl[6].checked, true);
  tpl[5].click();

  assert.deepEqual(actions, ["new", "select:s1", "select:s2"]);
});

test("dock menu template falls back cleanly when no session is open or recent list is empty", () => {
  const tpl = buildDockMenuTemplate({ current: null, recent: [], lang: "en" });
  assert.deepEqual(
    tpl.map((item) => ({ label: item.label, type: item.type, enabled: item.enabled })),
    [
      { label: "New Chat", type: undefined, enabled: undefined },
      { label: undefined, type: "separator", enabled: undefined },
      { label: "Recent Sessions", type: undefined, enabled: false },
      { label: "No Recent Sessions", type: undefined, enabled: false },
    ],
  );
});

