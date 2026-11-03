"use strict";

const fs = require("node:fs");
const path = require("node:path");

const DEFAULT_STATE = Object.freeze({ width: 1360, height: 880, maximized: false });

function normalizeWindowState(value) {
  if (!value || typeof value !== "object") return { ...DEFAULT_STATE };
  const state = { ...DEFAULT_STATE };
  if (Number.isInteger(value.width) && value.width >= 720 && value.width <= 7680) {
    state.width = value.width;
  }
  if (Number.isInteger(value.height) && value.height >= 480 && value.height <= 4320) {
    state.height = value.height;
  }
  if (Number.isInteger(value.x)) state.x = value.x;
  if (Number.isInteger(value.y)) state.y = value.y;
  state.maximized = value.maximized === true;
  return state;
}

function keepVisiblePosition(state, displays) {
  if (!Number.isInteger(state.x) || !Number.isInteger(state.y) || !displays.length) return state;
  const visible = displays.some(({ workArea }) => (
    state.x < workArea.x + workArea.width - 64
    && state.x + state.width > workArea.x + 64
    && state.y < workArea.y + workArea.height - 64
    && state.y + state.height > workArea.y + 64
  ));
  if (visible) return state;
  const { x, y, ...centered } = state;
  return centered;
}

function readWindowState(file) {
  try {
    return normalizeWindowState(JSON.parse(fs.readFileSync(file, "utf8")));
  } catch {
    return { ...DEFAULT_STATE };
  }
}

function writeWindowState(file, value) {
  const state = normalizeWindowState(value);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(state));
  fs.renameSync(tmp, file);
}

module.exports = {
  DEFAULT_STATE, keepVisiblePosition, normalizeWindowState, readWindowState, writeWindowState,
};
