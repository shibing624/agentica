import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api";
import { useAppState, showToast } from "../store";
import { useStrings } from "../i18n";
import {
  activateTab, adoptTerminals, closeTab, fileKey, initialTabs, nextTerminalOrdinal, openTab, skillKey,
  terminalKey, type PanelTab, type TabsState,
} from "../lib/panelTabs";
import { clampWidth, PANEL_MAX_RATIO, PANEL_MIN, preferredWidth, resolvedWidth } from "./panelWidth";

export type FilesPanelState = {
  open: boolean;
  setOpen: (open: boolean) => void;
  width: number;
  startResize: (e: React.MouseEvent) => void;
  /** Double-click on the handle: back to the first-open default. */
  resetWidth: () => void;
  /** Panel covers the whole chat column (and the sidebar). */
  fullscreen: boolean;
  setFullscreen: (on: boolean) => void;
  tabs: PanelTab[];
  active: string;
  activate: (key: string) => void;
  close: (key: string) => void;
  /** Open a workspace file in its own tab (or focus the tab it already has). */
  browsePath: (path: string) => void;
  /** Open an installed skill's SKILL.md in a tab. */
  openSkill: (name: string) => void;
  /** Start a shell in the session's working directory, in a new tab. Admin only. */
  canTerminal: boolean;
  newTerminal: () => void;
  /** The gateway no longer has this shell (closed elsewhere, or the gateway
   *  restarted): drop its tab without asking the gateway to close it. */
  terminalGone: (id: string) => void;
};

// New key: widths saved under the old one were clamped to the old 640px cap,
// and would keep the panel at that size forever.
const WIDTH_KEY = "ag_files_panel_w";
function readSaved(): number | null {
  const raw = localStorage.getItem(WIDTH_KEY);
  if (raw == null) return null;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function useViewportWidth() {
  const [w, setW] = useState(window.innerWidth);
  useEffect(() => {
    const on = () => setW(window.innerWidth);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return w;
}

export function useFilesPanel(sessionId: string | null, workspace: string): FilesPanelState {
  const s = useAppState();
  const S = useStrings();
  const viewport = useViewportWidth();
  const sidebar = s.sidebarCollapsed ? 0 : s.sidebarWidth;
  const [open, setOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [saved, setSaved] = useState(readSaved);
  const width = resolvedWidth(preferredWidth(saved, viewport), viewport, sidebar);
  const canTerminal = s.accountRole === "admin";

  // Tabs belong to a session: switching chats swaps the set, and coming back
  // restores what was open there. Terminals keep running on the gateway either way.
  const bySession = useRef(new Map<string, TabsState>());
  const sessionKey = sessionId || "";
  const [state, setTabs] = useState<TabsState>(initialTabs);
  const stateRef = useRef(state);
  stateRef.current = state;
  const update = useCallback((fn: (st: TabsState) => TabsState) => {
    setTabs((st) => {
      const next = fn(st);
      bySession.current.set(sessionKey, next);
      return next;
    });
  }, [sessionKey]);

  useEffect(() => {
    setTabs(bySession.current.get(sessionKey) ?? initialTabs());
  }, [sessionKey]);

  // After a reload the tab list is gone but the shells are not.
  useEffect(() => {
    if (!canTerminal || !sessionKey) return;
    let cancelled = false;
    void api.fetchTerminals().then(({ ok, data }) => {
      if (cancelled || !ok || !data) return;
      const mine = (data.terminals || [])
        .filter((t) => t.session_id === sessionKey && !t.exited)
        .sort((a, b) => a.created_at - b.created_at)
        .map((t) => t.id);
      if (mine.length) update((st) => adoptTerminals(st, mine));
    });
    return () => { cancelled = true; };
  }, [canTerminal, sessionKey, update]);

  // Closing leaves fullscreen: the next open is the normal column again.
  useEffect(() => { if (!open) setFullscreen(false); }, [open]);
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      // A full-screen program in a terminal owns Esc.
      if (e.key !== "Escape" || (e.target as HTMLElement | null)?.closest(".term-screen")) return;
      setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  const resetWidth = useCallback(() => {
    localStorage.removeItem(WIDTH_KEY);
    setSaved(null);
  }, []);

  const startResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = width;
    const next = (ev: MouseEvent) =>
      clampWidth(startW + (startX - ev.clientX), PANEL_MIN, Math.max(PANEL_MIN, window.innerWidth * PANEL_MAX_RATIO));
    const onMove = (ev: MouseEvent) => setSaved(next(ev));
    const onUp = (ev: MouseEvent) => {
      localStorage.setItem(WIDTH_KEY, String(next(ev)));
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [width]);

  const browsePath = useCallback((path: string) => {
    setOpen(true);
    update((st) => openTab(st, { key: fileKey(path), kind: "file", path }));
  }, [update]);

  const openSkill = useCallback((name: string) => {
    setOpen(true);
    update((st) => openTab(st, { key: skillKey(name), kind: "skill", name }));
  }, [update]);

  const activate = useCallback((key: string) => update((st) => activateTab(st, key)), [update]);

  const close = useCallback((key: string) => {
    const tab = stateRef.current.tabs.find((t) => t.key === key);
    // Closing a terminal tab ends its shell, as in dsh; collapsing the panel does not.
    if (tab?.kind === "terminal") void api.closeTerminal(tab.id);
    update((st) => closeTab(st, key));
  }, [update]);

  const newTerminal = useCallback(() => {
    if (!workspace) { showToast(S.files.noRoot, 2500); return; }
    setOpen(true);
    void api.createTerminal({ cwd: workspace, session_id: sessionKey }).then(({ ok, data }) => {
      if (!ok || !data) { showToast((data as any)?.detail || S.terminal.startFailed, 3000); return; }
      update((st) => openTab(st, { key: terminalKey(data.id), kind: "terminal", id: data.id, ordinal: nextTerminalOrdinal(st) }));
    });
  }, [workspace, sessionKey, update, S.files.noRoot, S.terminal.startFailed]);

  const terminalGone = useCallback((id: string) => update((st) => closeTab(st, terminalKey(id))), [update]);

  return {
    open, setOpen, width, startResize, resetWidth, fullscreen, setFullscreen,
    tabs: state.tabs, active: state.active, activate, close,
    browsePath, openSkill, canTerminal, newTerminal, terminalGone,
  };
}
