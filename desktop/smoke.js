"use strict";

const path = require("node:path");
const { setTimeout: delay } = require("node:timers/promises");

function desktopStatePaths(root) {
  return {
    profile: path.resolve(root, "electron"),
    home: path.resolve(root, "gateway"),
    runtime: path.resolve(root, "runtime"),
  };
}

async function waitForChat(probe, timeout = 20_000, interval = 125) {
  const expires = Date.now() + timeout;
  let snapshot;
  do {
    snapshot = await probe();
    if (snapshot.composer && !snapshot.login) return snapshot;
    if (Date.now() >= expires) break;
    await delay(interval);
  } while (Date.now() <= expires);
  throw new Error(`Chat did not become ready (${snapshot.login ? "login screen" : "composer missing"})`);
}

module.exports = { desktopStatePaths, waitForChat };