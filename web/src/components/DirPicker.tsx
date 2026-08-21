import { useEffect } from "react";
import type { MouseEvent, ReactNode } from "react";
import * as api from "../api";
import { browseDir, loadDirHistory } from "../data";
import { useStrings } from "../i18n";
import { IconClose, IconFolder } from "../icons";
import { shortenPath } from "../lib/format";
import { getState, setState, showToast, useAppState } from "../store";

/**
 * Absolute-path picker: type it, reuse a recent one, or walk the filesystem the
 * gateway can see.
 *
 * There is deliberately **no native folder dialog**. A browser tab cannot open
 * one that yields an absolute path (`webkitdirectory` reports files under a
 * directory, by relative name), and the only process that could pop one is the
 * gateway — which may be on another machine, where the dialog would appear on a
 * screen nobody is looking at and block the request until someone clicked it.
 * So the listing walks the gateway's filesystem over HTTP, which is the same
 * answer in a browser tab, in the desktop shell, and over the LAN.
 *
 * The browse state lives in the store rather than here so the two callers (the
 * per-session dir modal and the default-dir setting) cannot get out of sync
 * with the path the user is editing.
 */
export function DirPicker({
  value, onChange, onCommit, extraAction,
}: {
  value: string;
  onChange: (dir: string) => void;
  onCommit?: (dir: string) => void;
  extraAction?: ReactNode;
}) {
  const s = useAppState();
  const S = useStrings();
  useEffect(() => { void loadDirHistory(); }, []);

  const browse = s.dirBrowse;
  const toggle = () => {
    if (browse.open) { setState({ dirBrowse: { open: false, path: "", parent: null, dirs: [] } }); return; }
    void browseDir(value || getState().serverDir);
  };

  const removeHistory = async (dir: string, e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const { ok, data } = await api.deleteDirHistoryApi(dir);
    if (!ok) { showToast(S.common.deleteFailed); return; }
    setState({ dirHistory: data?.history || getState().dirHistory.filter((d) => d !== dir) });
  };

  return (
    <>
      <div className="dm-row">
        <input
          value={value}
          placeholder="/absolute/path"
          spellCheck={false}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing) {
              e.preventDefault();
              onCommit?.(value);
            }
          }}
        />
        <button className={"cron-act" + (browse.open ? " active" : "")} onClick={toggle}>
          {browse.open ? S.dir.collapse : S.dir.browse}
        </button>
        {extraAction}
      </div>

      {!!s.dirHistory.length && (
        <div className="dir-history">
          {s.dirHistory.map((d) => (
            <span className="dir-hist-item" key={d} title={d}>
              <button type="button" className="dir-hist-pick" onClick={() => onChange(d)}>
                <IconFolder /> {shortenPath(d)}
              </button>
              <button type="button" className="dir-hist-x" title={S.common.delete} onClick={(e) => void removeHistory(d, e)}>
                <IconClose />
              </button>
            </span>
          ))}
        </div>
      )}

      {browse.open && (
        <div className="dir-browse">
          <div className="dir-browse-head">
            <code title={browse.path}>{browse.path}</code>
            <div className="dir-browse-acts">
              {browse.parent && (
                <button className="cron-act" onClick={() => void browseDir(browse.parent!)}>{S.dir.up}</button>
              )}
              <button className="dp-btn primary" onClick={() => onChange(browse.path)}>{S.dir.pick}</button>
            </div>
          </div>
          <div className="dir-browse-list">
            {!browse.dirs.length && <div className="settings-empty">{S.dir.noSubdirs}</div>}
            {browse.dirs.map((d) => (
              <button className="dir-browse-item" key={d.path} onClick={() => void browseDir(d.path)}>
                <IconFolder /> {d.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
