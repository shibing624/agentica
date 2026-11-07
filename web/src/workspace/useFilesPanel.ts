import { useCallback, useEffect, useState } from "react";
import { useAppState } from "../store";
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
  browsePath: (path: string) => void;
  openRequest: { path: string } | null;
  /** Skill whose SKILL.md the panel shows on top of the file browser. */
  skill: string | null;
  openSkill: (name: string) => void;
  closeSkill: () => void;
};

// New key: widths saved under the old one were clamped to the old 640px cap,
// and would keep the panel at that size forever.
const WIDTH_KEY = "ag_files_panel_w";
/** `.files-resize` sits between the chat and the panel and takes its own width. */
const HANDLE_W = 5;

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

export function useFilesPanel(sessionId: string | null): FilesPanelState {
  const s = useAppState();
  const viewport = useViewportWidth();
  const sidebar = (s.sidebarCollapsed ? 0 : s.sidebarWidth) + HANDLE_W;
  const [open, setOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [saved, setSaved] = useState(readSaved);
  const [openRequest, setOpenRequest] = useState<{ path: string } | null>(null);
  const [skill, setSkill] = useState<string | null>(null);
  const width = resolvedWidth(preferredWidth(saved, viewport), viewport, sidebar);

  useEffect(() => {
    setOpenRequest(null);
    setSkill(null);
  }, [sessionId]);

  // Closing leaves fullscreen: the next open is the normal column again.
  useEffect(() => { if (!open) setFullscreen(false); }, [open]);
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setFullscreen(false); };
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
    setSkill(null);
    setOpenRequest({ path });
  }, []);

  const openSkill = useCallback((name: string) => {
    setOpen(true);
    setSkill(name);
  }, []);

  const closeSkill = useCallback(() => setSkill(null), []);

  return {
    open, setOpen, width, startResize, resetWidth, fullscreen, setFullscreen,
    browsePath, openRequest, skill, openSkill, closeSkill,
  };
}
