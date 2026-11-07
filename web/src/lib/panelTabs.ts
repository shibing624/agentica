import type { DiffHunk } from "./fileDiff";

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
  /** A reply's edits to one file, one hunk list per tool call. */
  | { key: string; kind: "diff"; path: string; edits: DiffHunk[][] }
  | { key: string; kind: "terminal"; id: string; ordinal: number };

export type TabsState = { tabs: PanelTab[]; active: string };

export const FILES_TAB: PanelTab = { key: "files", kind: "files" };
/** One start page per panel: "+" focuses it when it is already open. */
export const GUIDE_TAB: PanelTab = { key: "guide", kind: "guide" };

export function initialTabs(): TabsState {
  return { tabs: [FILES_TAB], active: FILES_TAB.key };
}

export const fileKey = (path: string) => `file:${path}`;
export const skillKey = (name: string) => `skill:${name}`;
export const diffKey = (path: string) => `diff:${path}`;
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

/** Open a tab, or focus the open one with its content replaced: one diff tab
 *  per file, showing whichever reply's edits were opened last. */
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

/** Tab label for a file: its basename. */
export function baseName(path: string): string {
  const i = path.lastIndexOf("/");
  return i >= 0 ? path.slice(i + 1) : path;
}
