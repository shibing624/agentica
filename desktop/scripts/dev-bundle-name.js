"use strict";
/**
 * `npm start` runs the stock `node_modules/electron/dist/Electron.app`, whose
 * Info.plist names it "Electron" — the Dock tooltip and the app menu read that,
 * not package.json. A packaged build already carries productName "Agentica".
 *
 * This renames only the dev bundle (CFBundleName / CFBundleDisplayName) and
 * re-registers it with LaunchServices so the Dock picks the name up. It leaves
 * `app.name` alone on purpose: that drives `userData`, and renaming it would
 * move the dev profile onto the managed-runtime root
 * (`~/Library/Application Support/Agentica`).
 *
 * Editing Info.plist invalidates Electron.app's ad-hoc/linker signature, which
 * makes macOS `usernotificationsd` reject local notifications. Re-signing the
 * bundle with `--deep --sign -` binds Info.plist so notifications work in
 * `npm start` too.
 */
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const NAME = "Agentica";
const ENTITLEMENTS = path.resolve(__dirname, "..", "build", "entitlements.mac.plist");

function isBundleSigned(bundle) {
  try {
    execFileSync("/usr/bin/codesign", ["--verify", "--deep", "--strict", bundle], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function signBundle(bundle, entitlements = ENTITLEMENTS) {
  const args = ["--force", "--deep", "--sign", "-"];
  if (entitlements && fs.existsSync(entitlements)) {
    args.push("--entitlements", entitlements);
  }
  args.push(bundle);
  execFileSync("/usr/bin/codesign", args, { stdio: "ignore" });
}

function main() {
  if (process.platform !== "darwin") return;
  const exe = require("electron");
  const bundle = path.resolve(path.dirname(exe), "..", "..");
  const plist = path.join(bundle, "Contents", "Info.plist");
  if (!fs.existsSync(plist)) return;
  const read = (key) => {
    try {
      return execFileSync("/usr/bin/plutil", ["-extract", key, "raw", plist], { encoding: "utf8" }).trim();
    } catch {
      return "";
    }
  };
  const renamed = read("CFBundleName") === NAME && read("CFBundleDisplayName") === NAME;
  if (renamed && isBundleSigned(bundle)) return;
  if (!renamed) {
    for (const key of ["CFBundleName", "CFBundleDisplayName"]) {
      execFileSync("/usr/bin/plutil", ["-replace", key, "-string", NAME, plist]);
    }
  }
  signBundle(bundle);
  const lsregister = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister";
  if (fs.existsSync(lsregister)) execFileSync(lsregister, ["-f", bundle]);
  console.log(`dev bundle prepared (${NAME}, signed): ${bundle}`);
}

if (require.main === module) {
  main();
}

module.exports = { NAME, isBundleSigned, signBundle, main };
