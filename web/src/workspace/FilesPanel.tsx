import { IconClose } from "../icons";
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
  return (
    <>
      {panel.open && (
        <div className="files-resize" onMouseDown={panel.startResize} title={S.files.resizeHandle} />
      )}
      <aside
        className={"files-panel" + (panel.open ? " open" : "")}
        style={{ width: panel.open ? panel.width : 0 }}
        inert={!panel.open}
      >
        <div className="files-panel-inner" style={{ width: panel.width }}>
          <div className="files-panel-head">
            <h4>{panel.skill ? S.files.skillTitle : S.files.title}</h4>
            <button type="button" className="ib" title={S.common.close} onClick={() => panel.setOpen(false)}>
              <IconClose />
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
