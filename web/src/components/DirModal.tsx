import { useEffect } from "react";
import { browseDir, loadDirHistory } from "../data";
import { IconClose, IconFolder } from "../icons";
import { shortenPath } from "../lib/format";
import { getState, saveSessions, setState, showToast, useAppState } from "../store";

/** Per-session working directory. Kept separate from the settings tab's default
 *  dir: this one only retargets the current (or next) chat. */
export function DirModal() {
  const s = useAppState();
  useEffect(() => { void loadDirHistory(); }, []);
  const close = () => setState({ dirModal: { ...getState().dirModal, open: false }, dirBrowse: { open: false, path: "", parent: null, dirs: [] } });
  const save = () => {
    const dir = s.dirModal.value.trim();
    if (!dir) { showToast("目录不能为空"); return; }
    if (s.dirModal.forNewSession) {
      setState({ pendingNewChatDir: dir });
    } else if (s.curSess && s.sessions[s.curSess]) {
      s.sessions[s.curSess].dir = dir;
      saveSessions();
    }
    close();
    showToast("工作目录已设置为 " + shortenPath(dir));
  };
  return (
    <div className="dir-modal-overlay open" onClick={close}>
      <div className="dir-modal" onClick={(e) => e.stopPropagation()}>
        <h3>工作目录</h3>
        <p className="dm-desc">这是本会话的项目目录。同目录下的 CLI 会话共享同一棵会话树。</p>
        <div className="dm-row">
          <input value={s.dirModal.value} placeholder="/absolute/path"
                 onChange={(e) => setState({ dirModal: { ...s.dirModal, value: e.target.value } })} />
          <button className="cron-act" onClick={() => void browseDir(s.dirModal.value || s.serverDir)}>浏览</button>
        </div>
        {!!s.dirHistory.length && (
          <div className="dir-history">
            {s.dirHistory.map((d) => (
              <button className="dir-hist-item" key={d} title={d}
                      onClick={() => setState({ dirModal: { ...s.dirModal, value: d } })}>
                <IconFolder /> {shortenPath(d)}
              </button>
            ))}
          </div>
        )}
        {s.dirBrowse.open && (
          <div className="dir-browse">
            <div className="dir-browse-head">
              <code>{s.dirBrowse.path}</code>
              <div className="dir-browse-acts">
                {s.dirBrowse.parent && <button className="cron-act" onClick={() => void browseDir(s.dirBrowse.parent!)}>上一级</button>}
                <button className="cron-act" onClick={() => setState({ dirModal: { ...getState().dirModal, value: s.dirBrowse.path } })}>选此目录</button>
              </div>
            </div>
            <div className="dir-browse-list">
              {!s.dirBrowse.dirs.length && <div className="settings-empty">没有子目录</div>}
              {s.dirBrowse.dirs.map((d) => (
                <button className="dir-browse-item" key={d.path} onDoubleClick={() => void browseDir(d.path)}
                        onClick={() => setState({ dirModal: { ...getState().dirModal, value: d.path } })}>
                  <IconFolder /> {d.name}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="dm-actions">
          <div className="dm-right">
            <button className="dp-btn" onClick={close}>取消</button>
            <button className="dp-btn primary" onClick={save}>保存</button>
          </div>
        </div>
        <button className="pf-close" onClick={close}><IconClose /></button>
      </div>
    </div>
  );
}
