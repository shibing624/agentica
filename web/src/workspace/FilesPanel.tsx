import { IconExitFullscreen, IconFullscreen, IconSidebarRight } from "../icons";
import { useStrings } from "../i18n";
import { SkillPane } from "./SkillPane";
import { WorkspaceBrowser } from "./WorkspaceBrowser";
import type { FilesPanelState } from "./useFilesPanel";

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
            <h4>{panel.skill ? S.files.skillTitle : S.files.title}</h4>
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
          {panel.skill && <SkillPane name={panel.skill} onBack={panel.closeSkill} />}
          {/* Kept mounted under the skill view so going back restores the
              folder and preview the user was on. */}
          <div className="files-panel-body" hidden={!!panel.skill}>
            {root ? (
              <WorkspaceBrowser root={root} openRequest={panel.openRequest} active={panel.open && !panel.skill} />
            ) : (
              <p className="ws-muted">{S.files.noRoot}</p>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
