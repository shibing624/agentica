import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api";
import { useAppState, showToast } from "../store";
import { useStrings } from "../i18n";
import {
  activateTab, adoptTerminals, closeTab, fileKey, reviewKey, FILES_TAB, GUIDE_TAB, initialSessionPanel,
  nextTerminalOrdinal, openTab, readSessionPanel, replaceTab, skillKey, terminalKey, upsertTab, writeSessionPanel,
  type PanelTab, type SessionPanelRecord, type TabsState,
} from "../lib/panelTabs";
import { isAbsOrHomePath, toWorkspaceRelative } from "../lib/file-path";
import type { FileEdits } from "../lib/fileDiff";
import { clampWidth, PANEL_MAX_RATIO, PANEL_MIN, preferredWidth, resolvedWidth } from "./panelWidth";

export type FilesPanelState = {
  sessionKey: string;
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
  /** Current directory path in the `files` tab for this session. */
  dirPath: string;
  setDirPath: (dirPath: string) => void;
  /** Open a workspace file in its own tab (or focus the tab it already has). */
  browsePath: (path: string) => void;
  /** Reveal `path`'s parent directory in the `files` tab. */
  revealInFilesTab: (path: string) => void;
  /** Open an installed skill's SKILL.md in a tab. */
  openSkill: (name: string) => void;
  /** Review round `turn`'s edits with `path` on screen (one tab per round). */
  openReview: (turn: number, files: FileEdits[], path: string) => void;
  selectReviewFile: (key: string, path: string) => void;
  /** "+": open the start page (or focus it), where the user picks what to open. */
  openGuide: () => void;
  /** Start-page cards: the start page turns into the picked page. */
  pickFiles: () => void;
  /** Start a shell in the session's working directory. Admin only. */
  canTerminal: boolean;
  pickTerminal: () => void;
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
  const [fullscreen, setFullscreen] = useState(false);
  const [saved, setSaved] = useState(readSaved);
  const width = resolvedWidth(preferredWidth(saved, viewport), viewport, sidebar);
  const canTerminal = s.accountRole === "admin";

  // Panel state (open + opened file history tabs + active tab + directory path)
  // belongs to each session (dsh's `dsh.sidebar-right.v1.<sessionId>`):
  // switching sessions swaps the whole record synchronously during render, and
  // localStorage preserves each session's opened file tabs across reloads.
  const bySession = useRef(new Map<string, SessionPanelRecord>());
  const sessionKey = sessionId || "";
  const loadForSession = useCallback((key: string, prevKey?: string): SessionPanelRecord => {
    const cached = bySession.current.get(key);
    if (cached) return cached;
    const persisted = readSessionPanel(key);
    if (persisted) {
      bySession.current.set(key, persisted);
      return persisted;
    }
    // A new chat draft (`""`) that just got its first sessionId inherits any
    // tabs/open state the user opened before sending the first message.
    if (prevKey === "" && key) {
      const draft = bySession.current.get("");
      if (draft && (draft.open || draft.tabs.length > 1 || draft.dirPath)) {
        bySession.current.set("", initialSessionPanel());
        bySession.current.set(key, draft);
        writeSessionPanel(key, draft);
        return draft;
      }
    }
    const fresh = initialSessionPanel();
    bySession.current.set(key, fresh);
    return fresh;
  }, []);

  const [renderedKey, setRenderedKey] = useState(sessionKey);
  const [state, setRecord] = useState<SessionPanelRecord>(() => loadForSession(sessionKey));
  if (renderedKey !== sessionKey) {
    const nextRecord = loadForSession(sessionKey, renderedKey);
    setRenderedKey(sessionKey);
    setRecord(nextRecord);
    if (!nextRecord.open && fullscreen) setFullscreen(false);
  }
  const stateRef = useRef(state);
  stateRef.current = state;
  const updateRecord = useCallback((fn: (rec: SessionPanelRecord) => SessionPanelRecord) => {
    setRecord((rec) => {
      const next = fn(rec);
      bySession.current.set(sessionKey, next);
      writeSessionPanel(sessionKey, next);
      return next;
    });
  }, [sessionKey]);

  const update = useCallback((fn: (st: TabsState) => TabsState) => {
    updateRecord((rec) => {
      const nextTabs = fn({ tabs: rec.tabs, active: rec.active });
      return { ...rec, tabs: nextTabs.tabs, active: nextTabs.active, preferredActive: undefined };
    });
  }, [updateRecord]);

  const open = state.open;
  const setOpen = useCallback((nextOpen: boolean) => {
    updateRecord((rec) => ({ ...rec, open: nextOpen }));
  }, [updateRecord]);

  const dirPath = state.dirPath;
  const setDirPath = useCallback((nextDir: string) => {
    updateRecord((rec) => (rec.dirPath === nextDir ? rec : { ...rec, dirPath: nextDir }));
  }, [updateRecord]);

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
      if (mine.length) {
        updateRecord((rec) => {
          const adopted = adoptTerminals({ tabs: rec.tabs, active: rec.active }, mine);
          const restoreTerm = rec.preferredActive && adopted.tabs.some((t) => t.key === rec.preferredActive)
            ? rec.preferredActive
            : adopted.active;
          return { ...rec, tabs: adopted.tabs, active: restoreTerm, preferredActive: undefined };
        });
      }
    });
    return () => { cancelled = true; };
  }, [canTerminal, sessionKey, updateRecord]);

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
    updateRecord((rec) => {
      const next = openTab({ tabs: rec.tabs, active: rec.active }, { key: fileKey(path), kind: "file", path });
      return { ...rec, open: true, tabs: next.tabs, active: next.active, preferredActive: undefined };
    });
  }, [updateRecord]);

  const revealInFilesTab = useCallback((path: string) => {
    const rel = isAbsOrHomePath(path) ? (toWorkspaceRelative(path, workspace) || "") : path;
    const norm = rel.replace(/\/+$/, "");
    const slash = norm.lastIndexOf("/");
    const parentDir = slash >= 0 && !isAbsOrHomePath(norm) ? norm.slice(0, slash) : "";
    updateRecord((rec) => ({
      ...rec,
      open: true,
      active: FILES_TAB.key,
      dirPath: parentDir,
      preferredActive: undefined,
    }));
  }, [updateRecord, workspace]);

  const openSkill = useCallback((name: string) => {
    updateRecord((rec) => {
      const next = openTab({ tabs: rec.tabs, active: rec.active }, { key: skillKey(name), kind: "skill", name });
      return { ...rec, open: true, tabs: next.tabs, active: next.active, preferredActive: undefined };
    });
  }, [updateRecord]);

  const openReview = useCallback((turn: number, files: FileEdits[], path: string) => {
    updateRecord((rec) => {
      const next = upsertTab({ tabs: rec.tabs, active: rec.active }, { key: reviewKey(turn), kind: "review", turn, files, path });
      return { ...rec, open: true, tabs: next.tabs, active: next.active, preferredActive: undefined };
    });
  }, [updateRecord]);

  const selectReviewFile = useCallback((key: string, path: string) => {
    update((st) => ({ ...st, tabs: st.tabs.map((t) => (t.key === key && t.kind === "review" ? { ...t, path } : t)) }));
  }, [update]);

  const activate = useCallback((key: string) => update((st) => activateTab(st, key)), [update]);

  const close = useCallback((key: string) => {
    const tab = stateRef.current.tabs.find((t) => t.key === key);
    // Closing a terminal tab ends its shell, as in dsh; collapsing the panel does not.
    if (tab?.kind === "terminal") void api.closeTerminal(tab.id);
    update((st) => closeTab(st, key));
  }, [update]);

  const openGuide = useCallback(() => {
    updateRecord((rec) => {
      const next = openTab({ tabs: rec.tabs, active: rec.active }, GUIDE_TAB);
      return { ...rec, open: true, tabs: next.tabs, active: next.active, preferredActive: undefined };
    });
  }, [updateRecord]);

  const pickFiles = useCallback(() => update((st) => replaceTab(st, GUIDE_TAB.key, FILES_TAB)), [update]);

  const pickTerminal = useCallback(() => {
    if (!workspace) { showToast(S.files.noRoot, 2500); return; }
    void api.createTerminal({ cwd: workspace, session_id: sessionKey }).then(({ ok, data }) => {
      if (!ok || !data) { showToast((data as any)?.detail || S.terminal.startFailed, 3000); return; }
      update((st) => replaceTab(st, GUIDE_TAB.key,
        { key: terminalKey(data.id), kind: "terminal", id: data.id, ordinal: nextTerminalOrdinal(st) }));
    });
  }, [workspace, sessionKey, update, S.files.noRoot, S.terminal.startFailed]);

  const terminalGone = useCallback((id: string) => update((st) => closeTab(st, terminalKey(id))), [update]);

  return {
    sessionKey,
    open, setOpen, width, startResize, resetWidth, fullscreen, setFullscreen,
    tabs: state.tabs, active: state.active, activate, close,
    dirPath, setDirPath,
    browsePath, revealInFilesTab, openSkill, openReview, selectReviewFile, openGuide, pickFiles, canTerminal, pickTerminal, terminalGone,
  };
}
