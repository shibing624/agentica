import { lazy, Suspense } from "react";
import { IconClose, IconExitFullscreen, IconFullscreen, IconPlus, IconSidebarRight, IconTerminal } from "../icons";
import { useStrings } from "../i18n";
import { baseName, type PanelTab } from "../lib/panelTabs";
import { SkillPane } from "./SkillPane";
import { FilePreview, WorkspaceBrowser } from "./WorkspaceBrowser";
import type { FilesPanelState } from "./useFilesPanel";

// xterm.js is ~300 kB; nobody who never opens a terminal downloads it.
const TerminalView = lazy(() => import("./TerminalView").then((m) => ({ default: m.TerminalView })));

function TabLabel({ tab }: { tab: PanelTab }) {
  const S = useStrings();
  if (tab.kind === "files") return <>{S.files.title}</>;
  if (tab.kind === "file") return <>{baseName(tab.path)}</>;
  if (tab.kind === "skill") return <>/{tab.name}</>;
  return <><IconTerminal />{S.terminal.title(tab.ordinal)}</>;
}

function tabTitle(tab: PanelTab): string | undefined {
  if (tab.kind === "file") return tab.path;
  if (tab.kind === "skill") return tab.name;
  return undefined;
}

export function FilesPanel({
  root, panel,
}: {
  root: string;
  panel: FilesPanelState;
}) {
  const S = useStrings();
  // Width 0 = the window is too narrow to keep the chat at its minimum; the
  // panel can still be opened fullscreen there.
  const shown = panel.open && (panel.width > 0 || panel.fullscreen);
  return (
    <>
      {shown && !panel.fullscreen && (
        <div className="files-resize" onMouseDown={panel.startResize} onDoubleClick={panel.resetWidth}
             title={S.files.resizeHandle} />
      )}
      <aside
        className={"files-panel" + (shown ? " open" : "") + (panel.fullscreen ? " fullscreen" : "")}
        style={{ width: panel.fullscreen ? undefined : shown ? panel.width : 0 }}
        inert={!shown}
      >
        <div className="files-panel-inner" style={{ width: panel.fullscreen ? "100%" : panel.width }}>
          <div className="files-panel-head">
            <div className="pt-strip" role="tablist">
              {panel.tabs.map((tab, i) => {
                const active = tab.key === panel.active;
                const prevActive = i > 0 && panel.tabs[i - 1].key === panel.active;
                return (
                  <div key={tab.key} className="pt-cell">
                    {i > 0 && <span className={"pt-slot" + (active || prevActive ? " blank" : "")} />}
                    <div className={"pt-tab" + (active ? " active" : "") + (tab.kind === "files" ? " fixed" : "")}
                         role="tab" aria-selected={active} title={tabTitle(tab)}
                         onMouseDown={(e) => { if (e.button === 1 && tab.kind !== "files") { e.preventDefault(); panel.close(tab.key); } }}
                         onClick={() => panel.activate(tab.key)}>
                      <span className="pt-title"><TabLabel tab={tab} /></span>
                      {tab.kind !== "files" && (
                        <button type="button" className="pt-close" aria-label={S.files.closeTab}
                                title={S.files.closeTab}
                                onClick={(e) => { e.stopPropagation(); panel.close(tab.key); }}>
                          <IconClose />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            {panel.canTerminal && (
              <button type="button" className="ib pt-add" onClick={panel.newTerminal}
                      title={S.terminal.newTerminal} aria-label={S.terminal.newTerminal}>
                <IconPlus />
              </button>
            )}
            <span className="pt-fill" />
            <button type="button" className="ib" onClick={() => panel.setFullscreen(!panel.fullscreen)}
                    title={panel.fullscreen ? S.files.exitFullscreen : S.files.fullscreen}
                    aria-label={panel.fullscreen ? S.files.exitFullscreen : S.files.fullscreen}>
              {panel.fullscreen ? <IconExitFullscreen /> : <IconFullscreen />}
            </button>
            <button type="button" className="ib" title={S.files.collapse} aria-label={S.files.collapse}
                    onClick={() => panel.setOpen(false)}>
              <IconSidebarRight />
            </button>
          </div>
          {/* Every tab stays mounted while it is open, so switching does not
              refetch a file or drop a terminal's screen. */}
          {panel.tabs.map((tab) => {
            const active = tab.key === panel.active;
            return (
              <div key={tab.key} className="files-panel-body" hidden={!active} role="tabpanel">
                {tab.kind === "files" && (root
                  ? <WorkspaceBrowser root={root} active={panel.open && active} onOpenFile={panel.browsePath} />
                  : <p className="ws-muted">{S.files.noRoot}</p>)}
                {tab.kind === "file" && root && <FilePreview root={root} path={tab.path} onOpenFile={panel.browsePath} />}
                {tab.kind === "skill" && <SkillPane name={tab.name} />}
                {tab.kind === "terminal" && (
                  <Suspense fallback={<p className="ws-muted">{S.common.loading}</p>}>
                    <TerminalView id={tab.id} visible={shown && active}
                                  onGone={() => panel.terminalGone(tab.id)} />
                  </Suspense>
                )}
              </div>
            );
          })}
        </div>
      </aside>
    </>
  );
}
