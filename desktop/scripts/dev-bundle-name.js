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
 */
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const NAME = "Agentica";

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
  if (read("CFBundleName") === NAME && read("CFBundleDisplayName") === NAME) return;
  for (const key of ["CFBundleName", "CFBundleDisplayName"]) {
    execFileSync("/usr/bin/plutil", ["-replace", key, "-string", NAME, plist]);
  }
  const lsregister = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister";
  if (fs.existsSync(lsregister)) execFileSync(lsregister, ["-f", bundle]);
  console.log(`dev bundle renamed to ${NAME}: ${bundle}`);
}

main();
