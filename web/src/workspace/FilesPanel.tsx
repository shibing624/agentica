import { IconClose } from "../icons";
import { useStrings } from "../i18n";
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
            <h4>{S.files.title}</h4>
            <button type="button" className="ib" title={S.common.close} onClick={() => panel.setOpen(false)}>
              <IconClose />
            </button>
          </div>
          {root ? (
            <WorkspaceBrowser root={root} openRequest={panel.openRequest} active={panel.open} />
          ) : (
            <p className="ws-muted">{S.files.noRoot}</p>
          )}
        </div>
      </aside>
    </>
  );
}
