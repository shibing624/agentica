import { useCallback, useEffect, useState } from "react";

export type FilesPanelState = {
  open: boolean;
  setOpen: (open: boolean) => void;
  width: number;
  startResize: (e: React.MouseEvent) => void;
  browsePath: (path: string) => void;
  openRequest: { path: string } | null;
};

const WIDTH_KEY = "ag_files_w";
const DEFAULT_W = 360;
const MIN_W = 280;
const MAX_W = 640;

function readWidth() {
  const n = Number(localStorage.getItem(WIDTH_KEY) || DEFAULT_W);
  return Number.isFinite(n) ? Math.min(MAX_W, Math.max(MIN_W, n)) : DEFAULT_W;
}

export function useFilesPanel(sessionId: string | null): FilesPanelState {
  const [open, setOpen] = useState(false);
  const [width, setWidth] = useState(readWidth);
  const [openRequest, setOpenRequest] = useState<{ path: string } | null>(null);

  useEffect(() => { setOpenRequest(null); }, [sessionId]);

  const startResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = width;
    const onMove = (ev: MouseEvent) => {
      const next = Math.min(MAX_W, Math.max(MIN_W, startW + (startX - ev.clientX)));
      setWidth(next);
    };
    const onUp = (ev: MouseEvent) => {
      const next = Math.min(MAX_W, Math.max(MIN_W, startW + (startX - ev.clientX)));
      localStorage.setItem(WIDTH_KEY, String(next));
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [width]);

  const browsePath = useCallback((path: string) => {
    setOpen(true);
    setOpenRequest({ path });
  }, []);

  return { open, setOpen, width, startResize, browsePath, openRequest };
}
