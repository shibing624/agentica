import { useStrings } from "../i18n";
import { IconClose } from "../icons";
import { shortenPath } from "../lib/format";
import { getState, saveSessions, setState, showToast, useAppState } from "../store";
import { DirPicker } from "./DirPicker";

/** Per-session working directory. Kept separate from the settings tab's default
 *  dir: this one only retargets the current (or next) chat. */
export function DirModal() {
  const s = useAppState();
  const S = useStrings();
  const close = () => setState({ dirModal: { ...getState().dirModal, open: false }, dirBrowse: { open: false, path: "", parent: null, dirs: [] } });
  const save = () => {
    const dir = s.dirModal.value.trim();
    if (!dir) { showToast(S.dir.empty); return; }
    if (s.dirModal.forNewSession) {
      setState({ pendingNewChatDir: dir });
    } else if (s.curSess && s.sessions[s.curSess]) {
      s.sessions[s.curSess].dir = dir;
      saveSessions();
    }
    close();
    showToast(S.dir.setTo(shortenPath(dir)));
  };
  return (
    <div className="dir-modal-overlay open" onClick={close}>
      <div className="dir-modal" onClick={(e) => e.stopPropagation()}>
        <h3>{S.dir.title}</h3>
        <p className="dm-desc">{S.dir.desc}</p>
        <DirPicker
          value={s.dirModal.value}
          onChange={(dir) => setState({ dirModal: { ...getState().dirModal, value: dir } })}
        />
        <div className="dm-actions">
          <div className="dm-right">
            <button className="dp-btn" onClick={close}>{S.common.cancel}</button>
            <button className="dp-btn primary" onClick={save}>{S.common.save}</button>
          </div>
        </div>
        <button className="pf-close" onClick={close}><IconClose /></button>
      </div>
    </div>
  );
}
