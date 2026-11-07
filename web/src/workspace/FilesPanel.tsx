import { lazy, Suspense, useEffect, useLayoutEffect, useRef } from "react";
import {
  IconClose, IconCompass, IconDiff, IconExitFullscreen, IconFolder, IconFullscreen, IconPlus, IconSidebarRight, IconTerminal,
} from "../icons";
import { useStrings } from "../i18n";
import { baseName, type PanelTab } from "../lib/panelTabs";
import { DiffView } from "./DiffView";
import { SkillPane } from "./SkillPane";
import { FilePreview, WorkspaceBrowser } from "./WorkspaceBrowser";
import type { FilesPanelState } from "./useFilesPanel";

// xterm.js is ~300 kB; nobody who never opens a terminal downloads it.
const TerminalView = lazy(() => import("./TerminalView").then((m) => ({ default: m.TerminalView })));

function TabLabel({ tab }: { tab: PanelTab }) {
  const S = useStrings();
  if (tab.kind === "files") return <>{S.files.title}</>;
  if (tab.kind === "guide") return <><IconCompass />{S.files.guideTitle}</>;
  if (tab.kind === "file") return <>{baseName(tab.path)}</>;
  if (tab.kind === "review") return <><IconDiff />{S.files.reviewTitle(tab.turn)}</>;
  if (tab.kind === "skill") return <>/{tab.name}</>;
  return <><IconTerminal />{S.terminal.title(tab.ordinal)}</>;
}

function tabTitle(tab: PanelTab): string | undefined {
  if (tab.kind === "file") return tab.path;
  if (tab.kind === "skill") return tab.name;
  return undefined;
}

/** Width of the strip's fade at a hidden side; mirrors the stylesheet. */
const STRIP_FADE = 24;

/** dsh's chip box: a mouse wheel scrolls the row sideways, the active chip is
 *  kept in view, and only a side that actually hides chips fades. */
function useTabStrip(active: string, tabs: PanelTab[]) {
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const apply = () => {
      const start = el.scrollLeft > 1;
      const end = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
      if (start || end) el.dataset.scroll = [start && "start", end && "end"].filter(Boolean).join(" ");
      else delete el.dataset.scroll;
    };
    // A vertical wheel has nowhere to go in a one-row strip; trackpads already
    // send deltaX and are left alone.
    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX) || el.scrollWidth <= el.clientWidth) return;
      e.preventDefault();
      el.scrollLeft += e.deltaY;
    };
    apply();
    el.addEventListener("scroll", apply, { passive: true });
    el.addEventListener("wheel", onWheel, { passive: false });
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", apply);
      el.removeEventListener("wheel", onWheel);
      ro.disconnect();
    };
  }, []);

  useLayoutEffect(() => {
    const el = box.current;
    const chip = el?.querySelector<HTMLElement>(".pt-tab.active");
    if (!el || !chip) return;
    const bounds = el.getBoundingClientRect();
    const rect = chip.getBoundingClientRect();
    if (rect.left < bounds.left) el.scrollLeft += rect.left - bounds.left - STRIP_FADE;
    else if (rect.right > bounds.right) el.scrollLeft += rect.right - bounds.right + STRIP_FADE;
  }, [active, tabs]);

  return box;
}

/** dsh's guide: a muted compass over one card per page type. Picking a card
 *  turns this tab into that page, so the start page is a doorway, not a page
 *  that stays open. */
function GuidePage({ panel }: { panel: FilesPanelState }) {
  const S = useStrings();
  const cards = [
    { id: "files", icon: <IconFolder />, title: S.files.guideFiles, desc: S.files.guideFilesDesc, pick: panel.pickFiles },
    ...(panel.canTerminal
      ? [{ id: "terminal", icon: <IconTerminal />, title: S.terminal.newTerminal, desc: S.terminal.guideDesc, pick: panel.pickTerminal }]
      : []),
  ];
  return (
    <div className="pg-guide">
      <span className="pg-hero" aria-hidden="true"><IconCompass /></span>
      {cards.map((c) => (
        <button key={c.id} type="button" className="pg-entry" data-guide-entry={c.id} onClick={c.pick}>
          <span className="pg-entry-icon">{c.icon}</span>
          <span className="pg-entry-text">
            <span className="pg-entry-title">{c.title}</span>
            <span className="pg-entry-desc">{c.desc}</span>
          </span>
        </button>
      ))}
    </div>
  );
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
  const strip = useTabStrip(panel.active, panel.tabs);
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
            <div className="pt-strip" role="tablist" ref={strip}>
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
            {/* One start page per panel: the control hides while it is open. */}
            {!panel.tabs.some((t) => t.kind === "guide") && (
              <button type="button" className="ib pt-add" onClick={panel.openGuide}
                      title={S.files.newTab} aria-label={S.files.newTab}>
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
                {tab.kind === "guide" && <GuidePage panel={panel} />}
                {tab.kind === "file" && (root || tab.path.startsWith("/")) && (
                  <FilePreview root={root} path={tab.path} onOpenFile={panel.browsePath} />
                )}
                {tab.kind === "review" && root && (
                  <DiffView root={root} turn={tab.turn} files={tab.files} path={tab.path}
                            onSelect={(path) => panel.selectReviewFile(tab.key, path)} onOpenFile={panel.browsePath} />
                )}
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
