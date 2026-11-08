import type { FileEdits } from "./fileDiff";

/**
 * Right-panel tabs (dsh's sidebar-right, without split panes or floating).
 * The file browser is the fixed first tab; files, skills and terminals open
 * after it. "+" opens the start page (dsh's guide), a doorway that turns into
 * the page picked on it. Pure, so the rules are testable without React.
 */
export type PanelTab =
  | { key: "files"; kind: "files" }
  | { key: "guide"; kind: "guide" }
  | { key: string; kind: "file"; path: string }
  | { key: string; kind: "skill"; name: string }
  /** One round's edits (dsh's review tab): every file the reply changed,
   *  `path` the one on screen; the header's selector switches between them. */
  | { key: string; kind: "review"; turn: number; files: FileEdits[]; path: string }
  | { key: string; kind: "terminal"; id: string; ordinal: number };

export type TabsState = { tabs: PanelTab[]; active: string };

export type SessionPanelRecord = {
  tabs: PanelTab[];
  active: string;
  open: boolean;
  dirPath: string;
  /** When a reloaded session's active tab was a terminal (`term:...`), remember
   *  it so `adoptTerminals` restores focus once the live shell list arrives. */
  preferredActive?: string;
};

export const PANEL_STORAGE_PREFIX = "ag_panel_v1:";

export const FILES_TAB: PanelTab = { key: "files", kind: "files" };
/** One start page per panel: "+" focuses it when it is already open. */
export const GUIDE_TAB: PanelTab = { key: "guide", kind: "guide" };

export function initialTabs(): TabsState {
  return { tabs: [FILES_TAB], active: FILES_TAB.key };
}

export function initialSessionPanel(): SessionPanelRecord {
  return { tabs: [FILES_TAB], active: FILES_TAB.key, open: false, dirPath: "" };
}

export const fileKey = (path: string) => `file:${path}`;
export const skillKey = (name: string) => `skill:${name}`;
export const reviewKey = (turn: number) => `review:${turn}`;
export const terminalKey = (id: string) => `term:${id}`;

/** Open a tab, or focus it if one with the same key is already open. New tabs
 *  go right after the active one, as in a browser. */
export function openTab(state: TabsState, tab: PanelTab): TabsState {
  if (state.tabs.some((t) => t.key === tab.key)) return { ...state, active: tab.key };
  const at = state.tabs.findIndex((t) => t.key === state.active);
  const tabs = [...state.tabs];
  tabs.splice(at < 0 ? tabs.length : at + 1, 0, tab);
  return { tabs, active: tab.key };
}

/** Close a tab; the fixed file browser cannot be closed. Closing the active
 *  tab activates its right neighbour, else its left. */
export function closeTab(state: TabsState, key: string): TabsState {
  if (key === FILES_TAB.key) return state;
  const at = state.tabs.findIndex((t) => t.key === key);
  if (at < 0) return state;
  const tabs = state.tabs.filter((t) => t.key !== key);
  if (state.active !== key) return { tabs, active: state.active };
  const next = tabs[at] ?? tabs[at - 1] ?? FILES_TAB;
  return { tabs, active: next.key };
}

export function activateTab(state: TabsState, key: string): TabsState {
  return state.tabs.some((t) => t.key === key) ? { ...state, active: key } : state;
}

/** Open a tab, or focus the open one with its content replaced: one review
 *  tab per round, showing whichever of its files was picked last. */
export function upsertTab(state: TabsState, tab: PanelTab): TabsState {
  const at = state.tabs.findIndex((t) => t.key === tab.key);
  if (at < 0) return openTab(state, tab);
  const tabs = [...state.tabs];
  tabs[at] = tab;
  return { tabs, active: tab.key };
}

/** Put `tab` where `key` is (the start page becoming the page picked on it).
 *  A page that is already open elsewhere is revealed instead, and `key` closes. */
export function replaceTab(state: TabsState, key: string, tab: PanelTab): TabsState {
  const at = state.tabs.findIndex((t) => t.key === key);
  if (at < 0) return openTab(state, tab);
  if (state.tabs.some((t) => t.key === tab.key)) {
    return { tabs: state.tabs.filter((t) => t.key !== key), active: tab.key };
  }
  const tabs = [...state.tabs];
  tabs[at] = tab;
  return { tabs, active: tab.key };
}

/** The lowest ordinal not in use, so "Terminal 2" is reused after it closes. */
export function nextTerminalOrdinal(state: TabsState): number {
  const used = new Set(state.tabs.flatMap((t) => (t.kind === "terminal" ? [t.ordinal] : [])));
  let n = 1;
  while (used.has(n)) n++;
  return n;
}

/** Terminals the gateway still runs for this session but the tab list lost
 *  (a page reload): put them back, in creation order. */
export function adoptTerminals(state: TabsState, ids: string[]): TabsState {
  let next = state;
  for (const id of ids) {
    if (next.tabs.some((t) => t.key === terminalKey(id))) continue;
    const tab: PanelTab = { key: terminalKey(id), kind: "terminal", id, ordinal: nextTerminalOrdinal(next) };
    next = { tabs: [...next.tabs, tab], active: next.active };
  }
  return next;
}

/**
 * Serialize one session's right-panel state for `localStorage` (mirrors dsh's
 * `writeSidebarLayout(sessionId, ...)`). Transient `guide` and `terminal` tabs
 * are excluded from storage — live terminals are re-adopted from the gateway.
 */
export function serializeSessionPanel(record: SessionPanelRecord): string {
  const tabs = record.tabs.filter((t) => t.kind !== "terminal" && t.kind !== "guide");
  const active = record.active === GUIDE_TAB.key ? FILES_TAB.key : record.active;
  return JSON.stringify({
    open: Boolean(record.open),
    active,
    dirPath: record.dirPath || "",
    tabs: tabs.length ? tabs : [FILES_TAB],
  });
}

/**
 * Parse and validate one session's saved right-panel state from `localStorage`
 * (mirrors dsh's `readSidebarLayout(sessionId)`).
 */
export function restoreSessionPanel(raw: string | null | undefined): SessionPanelRecord | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const seen = new Set<string>([FILES_TAB.key]);
    const tabs: PanelTab[] = [FILES_TAB];
    if (Array.isArray(parsed.tabs)) {
      for (const item of parsed.tabs) {
        if (!item || typeof item !== "object" || typeof item.kind !== "string") continue;
        if (item.kind === "file" && typeof item.path === "string" && item.path) {
          const key = fileKey(item.path);
          if (!seen.has(key)) {
            seen.add(key);
            tabs.push({ key, kind: "file", path: item.path });
          }
        } else if (item.kind === "skill" && typeof item.name === "string" && item.name) {
          const key = skillKey(item.name);
          if (!seen.has(key)) {
            seen.add(key);
            tabs.push({ key, kind: "skill", name: item.name });
          }
        } else if (
          item.kind === "review" &&
          typeof item.turn === "number" &&
          Array.isArray(item.files) &&
          typeof item.path === "string"
        ) {
          const key = reviewKey(item.turn);
          if (!seen.has(key)) {
            seen.add(key);
            tabs.push({ key, kind: "review", turn: item.turn, files: item.files, path: item.path });
          }
        }
      }
    }
    const rawActive = typeof parsed.active === "string" ? parsed.active : FILES_TAB.key;
    const hasActive = tabs.some((t) => t.key === rawActive);
    const preferredActive = !hasActive && rawActive.startsWith("term:") ? rawActive : undefined;
    return {
      tabs,
      active: hasActive ? rawActive : FILES_TAB.key,
      open: Boolean(parsed.open),
      dirPath: typeof parsed.dirPath === "string" ? parsed.dirPath : "",
      ...(preferredActive ? { preferredActive } : {}),
    };
  } catch {
    return null;
  }
}

export function readSessionPanel(
  sessionKey: string,
  storage: Pick<Storage, "getItem"> | null = typeof localStorage !== "undefined" ? localStorage : null,
): SessionPanelRecord | null {
  if (!sessionKey || !storage) return null;
  try {
    return restoreSessionPanel(storage.getItem(PANEL_STORAGE_PREFIX + sessionKey));
  } catch {
    return null;
  }
}

export function writeSessionPanel(
  sessionKey: string,
  record: SessionPanelRecord,
  storage: Pick<Storage, "setItem"> | null = typeof localStorage !== "undefined" ? localStorage : null,
): void {
  if (!sessionKey || !storage) return;
  try {
    storage.setItem(PANEL_STORAGE_PREFIX + sessionKey, serializeSessionPanel(record));
  } catch {
    /* Storage quota or private mode: in-memory state still works for this window. */
  }
}

/** Tab label for a file: its basename. */
export function baseName(path: string): string {
  const i = path.lastIndexOf("/");
  return i >= 0 ? path.slice(i + 1) : path;
}
