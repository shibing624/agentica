import * as ContextMenu from "@radix-ui/react-context-menu";
import { lazy, Suspense, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import {
  FileTypeIcon, IconClose, IconCompass, IconDiff, IconExitFullscreen, IconFileTree, IconFullscreen, IconGlobe, IconPlus, IconSidebarRight, IconTerminal,
} from "../icons";
import { useStrings } from "../i18n";
import { isAbsOrHomePath, toWorkspaceRelative } from "../lib/filePath";
import { browserTabLabel } from "../lib/browserUrl";
import { baseName, closeTabShortcut, isCloseTabKey, tabsToClose, type PanelTab } from "../lib/panelTabs";
import { BrowserView } from "./BrowserView";
import { DiffView } from "./DiffView";
import { SkillPane } from "./SkillPane";
import { clampWidth, resolveTreeLayout } from "./panelWidth";
import type { FilesPanelState } from "./useFilesPanel";

// xterm.js is ~300 kB; nobody who never opens a terminal downloads it.
const TerminalView = lazy(() => import("./TerminalView").then((m) => ({ default: m.TerminalView })));
// The browser renders Markdown with raw HTML and KaTeX (~600 kB with their
// parsers). Importing it statically made every page load fetch that before the
// panel was ever opened, which defeated the lazy ChatMarkdown in the chat.
const WorkspaceBrowser = lazy(() => import("./WorkspaceBrowser").then((m) => ({ default: m.WorkspaceBrowser })));
const FilePreview = lazy(() => import("./WorkspaceBrowser").then((m) => ({ default: m.FilePreview })));

function TabLabel({ tab }: { tab: PanelTab }) {
  const S = useStrings();
  if (tab.kind === "files") return <><FileTypeIcon kind="folder" size={15} />{S.files.title}</>;
  if (tab.kind === "guide") return <><IconCompass />{S.files.guideTitle}</>;
  if (tab.kind === "file") return <><FileTypeIcon path={tab.path} size={15} />{baseName(tab.path)}</>;
  if (tab.kind === "review") return <><IconDiff />{S.files.reviewTitle(tab.turn)}</>;
  if (tab.kind === "skill") return <><FileTypeIcon kind="markdown" size={15} />/{tab.name}</>;
  if (tab.kind === "browser") return <><IconGlobe />{tab.url || tab.title ? browserTabLabel(tab.url, tab.title) : S.browser.newTab}</>;
  return <><IconTerminal />{S.terminal.title(tab.ordinal)}</>;
}

function tabTitle(tab: PanelTab): string | undefined {
  if (tab.kind === "file") return tab.path;
  if (tab.kind === "skill") return tab.name;
  if (tab.kind === "browser") return tab.url || undefined;
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
    { id: "files", icon: <FileTypeIcon kind="folder" size={22} />, title: S.files.guideFiles, desc: S.files.guideFilesDesc, pick: panel.pickFiles },
    ...(panel.canTerminal
      ? [{ id: "terminal", icon: <IconTerminal />, title: S.terminal.newTerminal, desc: S.terminal.guideDesc, pick: panel.pickTerminal }]
      : []),
    { id: "browser", icon: <IconGlobe />, title: S.files.guideBrowser, desc: S.files.guideBrowserDesc, pick: panel.pickBrowser },
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
  const shown = panel.open;
  const strip = useTabStrip(panel.active, panel.tabs);
  const treeId = useId();
  const treeToggle = useRef<HTMLButtonElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const [contentWidth, setContentWidth] = useState(0);
  const [dragWidth, setDragWidth] = useState<number | null>(null);
  const drag = useRef<{ x: number; width: number; final: number } | null>(null);
  const [showTree, setShowTree] = useState(false);
  const activeTab = panel.tabs.find((tab) => tab.key === panel.active);
  const onPreview = activeTab?.kind === "file";
  const onFiles = panel.active === "files";
  const sideTree = onPreview && showTree;
  useLayoutEffect(() => {
    const el = content.current;
    if (!shown || !sideTree || !el) return;
    const measure = () => setContentWidth(el.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [shown, sideTree]);
  const treeLayout = resolveTreeLayout(contentWidth, dragWidth ?? panel.treeWidth);
  const currentFile = activeTab?.kind === "file" && panel.scope
    ? toWorkspaceRelative(activeTab.path, panel.scope.root.replaceAll("\\", "/")) : null;
  const treeLabel = showTree ? S.files.hideTree : S.files.showTree;
  return (
    <>
      {shown && !panel.fullscreen && !panel.floating && (
        <div className="files-resize" onMouseDown={panel.startResize} onDoubleClick={panel.resetWidth}
             title={S.files.resizeHandle} />
      )}
      <aside
        className={"files-panel" + (shown ? " open" : "") + (panel.fullscreen ? " fullscreen" : panel.floating ? " floating" : "")}
        style={{ width: panel.fullscreen ? undefined : shown ? panel.width : 0 }}
        inert={!shown}
        onKeyDownCapture={(e) => {
          if (!isCloseTabKey(e) || panel.active === "files") return;
          e.preventDefault();
          e.stopPropagation();
          panel.close(panel.active);
        }}
      >
        <div className="files-panel-inner">
          <div className="files-panel-head">
            <div className="pt-strip" role="tablist" ref={strip}>
              {panel.tabs.map((tab, i) => {
                const active = tab.key === panel.active;
                const prevActive = i > 0 && panel.tabs[i - 1].key === panel.active;
                return (
                  <div key={tab.key} className="pt-cell">
                    {i > 0 && <span className={"pt-slot" + (active || prevActive ? " blank" : "")} />}
                    <ContextMenu.Root>
                      <ContextMenu.Trigger asChild>
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
                      </ContextMenu.Trigger>
                      <ContextMenu.Portal>
                        <ContextMenu.Content className="pt-menu">
                          <ContextMenu.Item className="pt-menu-item" disabled={tab.kind === "files"}
                                            onSelect={() => panel.close(tab.key)}>
                            <span>{S.files.closeTab}</span>
                            <kbd>{closeTabShortcut()}</kbd>
                          </ContextMenu.Item>
                          <ContextMenu.Item className="pt-menu-item"
                                            disabled={tabsToClose(panel, tab.key, "others").length === 0}
                                            onSelect={() => panel.closeMany(tab.key, "others")}>
                            <span>{S.files.closeOthers}</span>
                          </ContextMenu.Item>
                          {tabsToClose(panel, tab.key, "right").length > 0 && (
                            <ContextMenu.Item className="pt-menu-item" onSelect={() => panel.closeMany(tab.key, "right")}>
                              <span>{S.files.closeRight}</span>
                            </ContextMenu.Item>
                          )}
                        </ContextMenu.Content>
                      </ContextMenu.Portal>
                    </ContextMenu.Root>
                  </div>
                );
              })}
            </div>
            <div className="pt-actions">
              {/* One start page per panel: the control hides while it is open. */}
              {!panel.tabs.some((t) => t.kind === "guide") && (
                <button type="button" className="ib" onClick={panel.openGuide}
                        title={S.files.newTab} aria-label={S.files.newTab}>
                  <IconPlus />
                </button>
              )}
              {onPreview && panel.scope && (
                <button type="button" className="ib" ref={treeToggle} title={treeLabel} aria-label={treeLabel}
                        aria-controls={treeId} aria-expanded={sideTree} onClick={() => setShowTree(!showTree)}>
                  <IconFileTree />
                </button>
              )}
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
          </div>
          {/* Every tab stays mounted while it is open, so switching does not
              refetch a file or drop a terminal's screen. */}
          <div className="ws-panel-content" ref={content}>
          {panel.tabs.filter((tab) => tab.kind !== "files").map((tab) => {
            const active = tab.key === panel.active;
            const scopedKey = `${panel.sessionKey}:${tab.key}`;
            return (
              <div key={scopedKey} className="files-panel-body" hidden={!active} role="tabpanel">
                {tab.kind === "guide" && <GuidePage panel={panel} />}
                {tab.kind === "file" && (panel.scope || isAbsOrHomePath(tab.path)) && (
                  <Suspense fallback={<p className="ws-muted">{S.common.loading}</p>}>
                    <FilePreview
                      root={panel.scope?.root ?? root}
                      scope={panel.scope ?? undefined}
                      path={tab.path}
                      onOpenFile={panel.browsePath}
                    />
                  </Suspense>
                )}
                {tab.kind === "review" && root && (
                  <DiffView root={root} turn={tab.turn} files={tab.files} path={tab.path}
                            onSelect={(path) => panel.selectReviewFile(tab.key, path)} onOpenFile={panel.browsePath} />
                )}
                {tab.kind === "skill" && <SkillPane name={tab.name} />}
                {tab.kind === "browser" && (
                  <BrowserView url={tab.url} pendingUrl={tab.pendingUrl || ""} visible={shown && active}
                               onChange={(patch) => panel.updateBrowser(tab.key, patch)}
                               onOpen={() => panel.confirmBrowser(tab.key)}
                               onDismiss={() => panel.dismissBrowser(tab.key)} />
                )}
                {tab.kind === "terminal" && (
                  <Suspense fallback={<p className="ws-muted">{S.common.loading}</p>}>
                    <TerminalView id={tab.id} visible={shown && active}
                                  onGone={() => panel.terminalGone(tab.id)} />
                  </Suspense>
                )}
              </div>
            );
          })}
          <div id={treeId} hidden={!onFiles && !sideTree}
               className={sideTree ? `ws-tree-column side ${treeLayout.mode}` : "ws-tree-column"}
               style={sideTree && treeLayout.mode !== "fill" ? { width: treeLayout.width } : undefined}
               onKeyDown={(event) => {
                 if (event.key !== "Escape" || event.defaultPrevented || event.nativeEvent.isComposing || !sideTree) return;
                 event.stopPropagation();
                 setShowTree(false);
                 treeToggle.current?.focus();
               }}>
            {sideTree && treeLayout.mode !== "fill" && (
              <div className={"ws-tree-resize" + (dragWidth !== null ? " dragging" : "")} role="separator"
                   aria-orientation="vertical" aria-label={S.files.resizeTree} aria-valuemin={treeLayout.min}
                   aria-valuemax={treeLayout.max} aria-valuenow={treeLayout.width} title={S.files.resizeHandle} tabIndex={0}
                   onDoubleClick={() => panel.setTreeWidth()}
                   onKeyDown={(event) => {
                     const next = event.key === "ArrowLeft" ? treeLayout.width + 16
                       : event.key === "ArrowRight" ? treeLayout.width - 16
                       : event.key === "Home" ? treeLayout.min : event.key === "End" ? treeLayout.max : null;
                     if (next === null) return;
                     event.preventDefault();
                     panel.setTreeWidth(clampWidth(next, treeLayout.min, treeLayout.max));
                   }}
                   onPointerDown={(event) => {
                     if (!event.isPrimary || event.button !== 0) return;
                     event.preventDefault();
                     event.currentTarget.setPointerCapture(event.pointerId);
                     drag.current = { x: event.clientX, width: treeLayout.width, final: treeLayout.width };
                     setDragWidth(treeLayout.width);
                   }}
                   onPointerMove={(event) => {
                     const start = drag.current;
                     if (!event.isPrimary || !start) return;
                     start.final = clampWidth(start.width + start.x - event.clientX, treeLayout.min, treeLayout.max);
                     setDragWidth(start.final);
                   }}
                   onLostPointerCapture={() => {
                     if (drag.current && drag.current.final !== drag.current.width) panel.setTreeWidth(drag.current.final);
                     drag.current = null;
                     setDragWidth(null);
                   }} />
            )}
            {panel.scope ? (
              <Suspense fallback={<p className="ws-muted">{S.common.loading}</p>}>
                <WorkspaceBrowser key={`${panel.accountId}:${panel.sessionKey}:${panel.scope.binding}`} scope={panel.scope}
                                  accountId={panel.accountId} isAdmin={panel.canTerminal} revision={panel.revision}
                                  active={shown && (onFiles || sideTree)} currentFile={currentFile}
                                  projectDir={panel.projectDir} onOpenFile={sideTree ? panel.replacePreview : panel.browsePath} />
              </Suspense>
            ) : <p className="ws-muted">{panel.scopeError ? S.files.loadFailed : panel.sessionKey && root ? S.common.loading : S.files.noRoot}</p>}
          </div>
          </div>
        </div>
      </aside>
    </>
  );
}
