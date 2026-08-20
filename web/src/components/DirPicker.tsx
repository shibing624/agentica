import { useEffect } from "react";
import { browseDir, loadDirHistory } from "../data";
import { useStrings } from "../i18n";
import { IconFolder } from "../icons";
import { shortenPath } from "../lib/format";
import { getState, setState, useAppState } from "../store";

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
  value, onChange, extraAction,
}: {
  value: string;
  onChange: (dir: string) => void;
  extraAction?: React.ReactNode;
}) {
  const s = useAppState();
  const S = useStrings();
  useEffect(() => { void loadDirHistory(); }, []);

  const browse = s.dirBrowse;
  const toggle = () => {
    if (browse.open) { setState({ dirBrowse: { open: false, path: "", parent: null, dirs: [] } }); return; }
    void browseDir(value || getState().serverDir);
  };

  return (
    <>
      <div className="dm-row">
        <input value={value} placeholder="/absolute/path" onChange={(e) => onChange(e.target.value)} />
        <button className={"cron-act" + (browse.open ? " active" : "")} onClick={toggle}>
          {browse.open ? S.dir.collapse : S.dir.browse}
        </button>
        {extraAction}
      </div>

      {!!s.dirHistory.length && (
        <div className="dir-history">
          {s.dirHistory.map((d) => (
            <button className="dir-hist-item" key={d} title={d} onClick={() => onChange(d)}>
              <IconFolder /> {shortenPath(d)}
            </button>
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
            {/* A click enters the folder. Selecting is what the button above is
                for — the previous split (click to select, double-click to
                enter) hid navigation behind a gesture nothing announced. */}
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
