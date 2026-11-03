"use strict";
/**
 * Assemble the runtime that ships inside an installer:
 *
 *   desktop/runtime/<os>-<arch>/python/   relocatable CPython (python-build-standalone)
 *       .../site-packages/                 agentica[gateway] from THIS checkout
 *
 * `<os>-<arch>` matches electron-builder's `${os}-${arch}` macros, so
 * `extraResources` picks the right directory per target. Dependencies are
 * installed with `pip --platform ... --target`, which downloads wheels for the
 * target instead of running its interpreter: the macOS arm64 runner can build
 * the x64 runtime too.
 *
 *   node scripts/prepare-runtime.js --target mac-arm64 --target mac-x64
 *   node scripts/prepare-runtime.js            # the host platform
 *
 * Needs a host `python3` with pip (PYTHON overrides it) and the web UI already
 * built into agentica/gateway/ui (the wheel carries it).
 */

const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const PYTHON_MINOR = "3.12";
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const OUT_ROOT = path.resolve(__dirname, "..", "runtime");

const TARGETS = {
  "mac-arm64": {
    triple: "aarch64-apple-darwin",
    platforms: ["macosx_11_0_arm64"],
  },
  "mac-x64": {
    triple: "x86_64-apple-darwin",
    platforms: ["macosx_10_9_x86_64", "macosx_11_0_x86_64", "macosx_12_0_x86_64"],
  },
  "linux-x64": {
    triple: "x86_64-unknown-linux-gnu",
    platforms: ["manylinux2014_x86_64", "manylinux_2_17_x86_64", "manylinux_2_28_x86_64"],
  },
  "win-x64": {
    triple: "x86_64-pc-windows-msvc",
    platforms: ["win_amd64"],
  },
};

function hostTarget() {
  const osName = { darwin: "mac", linux: "linux", win32: "win" }[process.platform];
  const arch = { x64: "x64", arm64: "arm64" }[process.arch];
  const key = `${osName}-${arch}`;
  if (!TARGETS[key]) throw new Error(`no runtime recipe for ${key}`);
  return key;
}

function parseTargets(argv) {
  const keys = [];
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--target") keys.push(argv[++i]);
  }
  const chosen = keys.length ? keys : [hostTarget()];
  for (const key of chosen) {
    if (!TARGETS[key]) throw new Error(`unknown target ${key}; known: ${Object.keys(TARGETS).join(", ")}`);
  }
  return chosen;
}

/**
 * Pinned, so every installer of a release ships the same interpreter and a
 * rebuild is reproducible. Bump both together when moving to a newer
 * python-build-standalone release (the asset must exist for all four triples).
 */
const PBS_TAG = process.env.AGENTICA_PBS_TAG || "20261026";
const PBS_PYTHON = process.env.AGENTICA_PBS_PYTHON || "3.12.16";

function pythonAsset(triple) {
  const name = `cpython-${PBS_PYTHON}+${PBS_TAG}-${triple}-install_only_stripped.tar.gz`;
  const url = "https://github.com/astral-sh/python-build-standalone/releases/download/"
    + `${PBS_TAG}/${encodeURIComponent(name)}`;
  return { name, url };
}

async function download(url, dest) {
  const resp = await fetch(url, { headers: { "User-Agent": "agentica-desktop-build" } });
  if (!resp.ok) throw new Error(`download ${url}: HTTP ${resp.status}`);
  fs.writeFileSync(dest, Buffer.from(await resp.arrayBuffer()));
}

function hostPython() {
  return process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function sitePackages(pythonDir, key) {
  return key.startsWith("win-")
    ? path.join(pythonDir, "Lib", "site-packages")
    : path.join(pythonDir, "lib", `python${PYTHON_MINOR}`, "site-packages");
}

function buildWheel(tmp) {
  const dir = path.join(tmp, "wheel");
  fs.mkdirSync(dir, { recursive: true });
  execFileSync(hostPython(), ["-m", "pip", "wheel", "--no-deps", "-w", dir, REPO_ROOT], { stdio: "inherit" });
  const wheel = fs.readdirSync(dir).find((f) => f.startsWith("agentica-") && f.endsWith(".whl"));
  if (!wheel) throw new Error("pip did not produce an agentica wheel");
  return path.join(dir, wheel);
}

async function prepare(key, wheel, tmp) {
  const spec = TARGETS[key];
  const outDir = path.join(OUT_ROOT, key);
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });

  const asset = pythonAsset(spec.triple);
  console.log(`[runtime] ${key}: ${asset.name}`);
  const archive = path.join(tmp, asset.name);
  await download(asset.url, archive);
  execFileSync("tar", ["-xzf", archive, "-C", outDir], { stdio: "inherit" });

  const site = sitePackages(path.join(outDir, "python"), key);
  fs.mkdirSync(site, { recursive: true });
  const platformArgs = spec.platforms.flatMap((p) => ["--platform", p]);
  execFileSync(
    hostPython(),
    [
      "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:",
      "--target", site,
      ...platformArgs,
      "--python-version", PYTHON_MINOR, "--implementation", "cp", "--abi", `cp${PYTHON_MINOR.replace(".", "")}`,
      `agentica[gateway] @ ${pathToFileURL(wheel).href}`,
    ],
    { stdio: "inherit" },
  );
  for (const entry of fs.readdirSync(site, { recursive: true, withFileTypes: true })) {
    if (entry.isDirectory() && entry.name === "__pycache__") {
      fs.rmSync(path.join(entry.parentPath, entry.name), { recursive: true, force: true });
    }
  }
  console.log(`[runtime] ${key}: ready at ${outDir}`);
}

async function main() {
  const keys = parseTargets(process.argv.slice(2));
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "agentica-runtime-"));
  try {
    const wheel = process.env.AGENTICA_WHEEL || buildWheel(tmp);
    for (const key of keys) await prepare(key, wheel, tmp);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err.stack || err.message);
    process.exit(1);
  });
}

module.exports = { TARGETS, parseTargets, sitePackages };

