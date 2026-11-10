/**
 * Right side of a preview bar: view-mode menu, refresh, download, and an
 * "open with" split button (text editor; the menu adds the default app and
 * "show in folder"). Shared by the workspace file preview and SKILL.md.
 */
import { useEffect, useRef, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { IconChevronDown, IconCopy, IconDownload, IconEditorApp, IconExternal, IconFinder, IconFolder, IconRefresh, IconTick } from "../icons";
import { showToast } from "../store";

export type ViewMode = "markdown" | "web" | "table" | "code" | "text";

export function useDismiss(open: boolean, close: () => void) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) close(); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, close]);
  return ref;
}

export function PreviewActions({
  modes, mode, onMode, onRefresh, refreshing, absPath, downloadHref, downloadName, onBackToTree,
}: {
  modes: ViewMode[];
  mode: ViewMode;
  onMode: (mode: ViewMode) => void;
  onRefresh: () => void;
  refreshing?: boolean;
  /** Absolute path on this machine; enables "open with". */
  absPath?: string;
  downloadHref?: string;
  downloadName?: string;
  /** Switch to the `files` tab at this file's directory (dsh's navigator toggle). */
  onBackToTree?: () => void;
}) {
  const S = useStrings();
  const P = S.files;
  const [modeOpen, setModeOpen] = useState(false);
  const [openMenu, setOpenMenu] = useState(false);
  const modeRef = useDismiss(modeOpen, () => setModeOpen(false));
  const openRef = useDismiss(openMenu, () => setOpenMenu(false));
  const label = (m: ViewMode) => ({ markdown: P.viewMarkdown, web: P.viewWeb, table: P.viewTable, code: P.viewCode, text: P.viewText })[m];

  const open = (app: "editor" | "finder" | "reveal") => {
    setOpenMenu(false);
    if (!absPath) return;
    void api.openPathApi(absPath, app).then(({ ok, data }) => {
      if (!ok) showToast((data as any)?.detail || S.chat.openFailed, 2500);
    });
  };

  return (
    <div className="pv-actions">
      {modes.length > 1 && (
        <div className="pv-menu-wrap" ref={modeRef}>
          <button type="button" className={"pv-mode" + (modeOpen ? " on" : "")} aria-haspopup="menu" aria-expanded={modeOpen}
                  onClick={() => setModeOpen((v) => !v)}>
            {label(mode)}
          </button>
          {modeOpen && (
            <div className="pv-menu" role="menu">
              {modes.map((m) => (
                <button key={m} type="button" role="menuitemradio" aria-checked={m === mode} className="pv-menu-item"
                        onClick={() => { onMode(m); setModeOpen(false); }}>
                  <span>{label(m)}</span>
                  {m === mode && <IconTick />}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      <button type="button" className={"pv-icon" + (refreshing ? " spinning" : "")} title={P.refresh} aria-label={P.refresh}
              onClick={onRefresh}>
        <IconRefresh />
      </button>
      {downloadHref && (
        <a className="pv-icon" href={downloadHref} download={downloadName} title={P.download} aria-label={P.download}>
          <IconDownload />
        </a>
      )}
      {absPath && (
        <div className="pv-menu-wrap" ref={openRef}>
          <div className={"pv-split" + (openMenu ? " on" : "")}>
            <button type="button" className="pv-split-main" title={P.openInEditor} aria-label={P.openInEditor}
                    onClick={() => open("editor")}>
              <IconEditorApp />
            </button>
            <button type="button" className="pv-split-arrow" title={P.openWith} aria-label={P.openWith}
                    aria-haspopup="menu" aria-expanded={openMenu} onClick={() => setOpenMenu((v) => !v)}>
              <IconChevronDown />
            </button>
          </div>
          {openMenu && (
            <div className="pv-menu right" role="menu">
              <button type="button" role="menuitem" className="pv-menu-item" onClick={() => open("editor")}>
                <span className="pv-menu-ico"><IconEditorApp /></span><span>{P.openInEditor}</span>
              </button>
              <button type="button" role="menuitem" className="pv-menu-item" onClick={() => open("finder")}>
                <span className="pv-menu-ico"><IconExternal /></span><span>{P.openDefault}</span>
              </button>
              <button type="button" role="menuitem" className="pv-menu-item"
                      onClick={() => {
                        setOpenMenu(false);
                        void navigator.clipboard.writeText(absPath).then(() => showToast(S.common.copied));
                      }}>
                <span className="pv-menu-ico"><IconCopy /></span><span>{P.copyPath}</span>
              </button>
              <div className="pv-menu-sep" />
              <button type="button" role="menuitem" className="pv-menu-item" onClick={() => open("reveal")}>
                <span className="pv-menu-ico"><IconFinder /></span><span>{P.revealInFolder}</span>
              </button>
            </div>
          )}
        </div>
      )}
      {onBackToTree && (
        <button type="button" className="pv-icon" title={P.backToFiles} aria-label={P.backToFiles}
                onClick={onBackToTree}>
          <IconFolder />
        </button>
      )}
    </div>
  );
}

/** Wrapped plain text with a copy button; no highlighting, no line numbers. */
export function PlainText({ text }: { text: string }) {
  const S = useStrings();
  return (
    <div className="pv-plain">
      <button type="button" className="pv-plain-copy" title={S.common.copy} aria-label={S.common.copy}
              onClick={() => { void navigator.clipboard.writeText(text).then(() => showToast(S.common.copied)); }}>
        <IconCopy />
      </button>
      <pre>{text}</pre>
    </div>
  );
}
