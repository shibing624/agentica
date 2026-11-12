"use strict";
/**
 * macOS Dock right-click menu builder.
 *
 * Shows:
 *   - New Chat action
 *   - Recent sessions from `/api/sessions?scope=recent` (running first, then
 *     most recently active). The checkmark marks the session open in the
 *     window; it is listed even when the recent page does not include it.
 *
 * Kept free of Electron imports so the menu structure, sorting, truncation,
 * and localisation can be unit-tested in plain Node (`runtime.test.js`).
 */

const path = require("node:path");

const DEFAULT_RECENT_LIMIT = 8;
const MAX_TITLE_CHARS = 42;
const MAX_FOLDER_CHARS = 20;

const LABELS = {
  zh: {
    newChat: "新建对话",
    recentHeader: "最近运行",
    noRecent: "暂无最近会话",
  },
  en: {
    newChat: "New Chat",
    recentHeader: "Recent Sessions",
    noRecent: "No Recent Sessions",
  },
};

/** Running marker. U+2022 sits at text height; the black circle U+25CF read as a blob. */
const RUNNING_MARK = "• ";

function labelsFor(lang) {
  return String(lang || "zh").toLowerCase().startsWith("zh") ? LABELS.zh : LABELS.en;
}

/** Clip by code point so surrogate pairs (emoji, CJK extensions) are never split. */
function clipText(text, maxChars = MAX_TITLE_CHARS) {
  const cleaned = String(text || "").replace(/\s+/g, " ").trim();
  const chars = Array.from(cleaned);
  if (chars.length <= maxChars) return cleaned;
  return chars.slice(0, Math.max(1, maxChars - 1)).join("") + "…";
}

/** Trailing directory name of a POSIX or Windows path. */
function folderName(workDir) {
  const raw = String(workDir || "").replace(/[\\/]+$/, "").trim();
  if (!raw) return "";
  const parts = raw.split(/[\\/]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : path.basename(raw);
}

function sessionTimestampMs(row) {
  if (!row || typeof row !== "object") return 0;
  const ts = Date.parse(row.last_timestamp || row.first_timestamp || "");
  return Number.isFinite(ts) ? ts : 0;
}

/**
 * Pick the top non-archived sessions for the Dock menu:
 * running sessions first, then most recently active.
 */
function selectRecentSessions(rows, limit = DEFAULT_RECENT_LIMIT) {
  if (!Array.isArray(rows) || limit <= 0) return [];
  return rows
    .filter((r) => r && typeof r.session_id === "string" && r.session_id && !r.archived)
    .slice()
    .sort((a, b) => {
      const runDiff = Number(Boolean(b.running)) - Number(Boolean(a.running));
      if (runDiff !== 0) return runDiff;
      return sessionTimestampMs(b) - sessionTimestampMs(a);
    })
    .slice(0, limit);
}

function formatSessionLabel(row, maxTitleChars = MAX_TITLE_CHARS) {
  const rawTitle = (row && row.name ? String(row.name) : "").replace(/\s+/g, " ").trim() || "Chat";
  const title = clipText(rawTitle, maxTitleChars);
  const folder = folderName(row && row.work_dir ? row.work_dir : "");
  const prefix = row && row.running ? RUNNING_MARK : "";
  const suffix = folder && folder.toLowerCase() !== rawTitle.toLowerCase()
    ? ` — ${clipText(folder, MAX_FOLDER_CHARS)}`
    : "";
  return `${prefix}${title}${suffix}`;
}

/**
 * Build the template array passed to `Menu.buildFromTemplate` for `app.dock.setMenu`.
 *
 * @param {object} opts
 * @param {{ id?: string, title?: string, running?: boolean } | null} [opts.current]
 * @param {Array<object>} [opts.recent]
 * @param {string} [opts.lang]
 * @param {() => void} [opts.onNewChat]
 * @param {(sessionId: string) => void} [opts.onSelectSession]
 * @param {number} [opts.limit]
 */
function buildDockMenuTemplate({
  current = null,
  recent = [],
  lang = "zh",
  onNewChat = null,
  onSelectSession = null,
  limit = DEFAULT_RECENT_LIMIT,
} = {}) {
  const L = labelsFor(lang);
  const template = [
    {
      label: L.newChat,
      click: () => {
        if (onNewChat) onNewChat();
      },
    },
  ];

  const curTitle = current && current.title ? String(current.title).replace(/\s+/g, " ").trim() : "";
  const curId = current && current.id ? String(current.id) : "";
  const curRunning = Boolean(current && current.running);

  template.push({ type: "separator" });
  template.push({ label: L.recentHeader, enabled: false });

  const items = selectRecentSessions(recent, limit);
  // The open session keeps its checkmark even when it fell off the recent page.
  if (curId && curTitle && !items.some((r) => r.session_id === curId)) {
    items.unshift({ session_id: curId, name: curTitle, running: curRunning });
    if (items.length > limit) items.pop();
  }
  if (items.length === 0) {
    template.push({ label: L.noRecent, enabled: false });
    return template;
  }

  for (const rawRow of items) {
    const isCurrent = Boolean(curId && rawRow.session_id === curId);
    const row = isCurrent && curTitle
      ? { ...rawRow, name: curTitle, running: curRunning || Boolean(rawRow.running) }
      : rawRow;
    template.push({
      label: formatSessionLabel(row),
      type: "checkbox",
      checked: isCurrent,
      click: () => {
        if (onSelectSession) onSelectSession(row.session_id);
      },
    });
  }

  return template;
}

module.exports = {
  DEFAULT_RECENT_LIMIT,
  MAX_TITLE_CHARS,
  clipText,
  folderName,
  selectRecentSessions,
  formatSessionLabel,
  buildDockMenuTemplate,
};
