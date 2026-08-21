import { useEffect, useRef, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { IconChevronDown, IconFolder } from "../icons";
import { projectNameForDir, saveSessions, setState, showToast, useAppState } from "../store";
import { shortenPath } from "../lib/format";

/**
 * Working-directory chip on a brand-new chat. Existing conversations keep
 * the dir they were created with; change it from the workspace panel.
 *
 * The path row is editable: Enter/blur commits a typed absolute path. An
 * invalid directory toasts and reverts to the last successful browse so the
 * user can keep trying. "Use this directory" applies the current browsed path.
 */
export function ComposerDir() {
  const s = useAppState();
  const S = useStrings();
  const cur = s.curSess ? s.sessions[s.curSess] : null;
  const canPick = !cur || !cur.msgs.length;
  const dir = (cur && cur.dir) || s.pendingNewChatDir || s.serverDir || "";
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState(dir);
  const [pathDraft, setPathDraft] = useState(dir);
  const [parent, setParent] = useState<string | null>(null);
  const [dirs, setDirs] = useState<{ name: string; path: string }[]>([]);
  const wrapRef = useRef<HTMLDivElement>(null);
  const pathRef = useRef(path);

  useEffect(() => { setPath(dir); setPathDraft(dir); }, [dir]);
  useEffect(() => { pathRef.current = path; }, [path]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  if (!canPick) return null;

  const load = async (p: string, revertOnError = false): Promise<string | null> => {
    const { ok, data } = await api.fetchFsBrowse(p);
    if (!ok || !data) {
      if (revertOnError) {
        showToast(S.dir.invalid);
        setPathDraft(pathRef.current);
      }
      return null;
    }
    setPath(data.path);
    setPathDraft(data.path);
    setParent(data.parent);
    setDirs(data.dirs || []);
    return data.path;
  };

  const commitPathEdit = () => {
    const p = pathDraft.trim();
    if (!p || p === path) {
      setPathDraft(path);
      return;
    }
    void load(p, true);
  };

  const apply = (p: string) => {
    if (!p) return;
    if (s.curSess && s.sessions[s.curSess]) {
      s.sessions[s.curSess].dir = p;
      saveSessions();
    }
    setState({ pendingNewChatDir: p });
    setOpen(false);
    showToast(S.dir.setTo(shortenPath(p)));
  };

  const applyCurrent = async () => {
    const typed = pathDraft.trim();
    if (!typed) return;
    const resolved = typed === path ? path : await load(typed, true);
    if (resolved) apply(resolved);
  };

  const toggle = () => {
    if (open) { setOpen(false); return; }
    setOpen(true);
    void load(dir || path);
  };

  const useTemp = async () => {
    const { ok, data } = await api.makeTempDirApi();
    if (!ok || !data?.path) { showToast(S.dir.setFailed); return; }
    apply(data.path);
  };

  return (
    <div className="composer-dir" ref={wrapRef}>
      <button type="button" className="foot-btn composer-dir-btn" title={dir || S.dir.title} onClick={toggle}>
        <IconFolder />
        <span>{projectNameForDir(dir) || S.dir.title}</span>
        <IconChevronDown />
      </button>
      {open && (
        <div className="composer-dir-pop">
          <div className="dir-browse-head">
            <input
              className="dir-browse-path"
              value={pathDraft}
              title={pathDraft}
              aria-label={S.dir.title}
              spellCheck={false}
              onChange={(e) => setPathDraft(e.target.value)}
              onBlur={commitPathEdit}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  commitPathEdit();
                } else if (e.key === "Escape") {
                  setPathDraft(path);
                }
              }}
            />
            <div className="dir-browse-acts">
              <button type="button" className="dp-btn primary"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => void applyCurrent()}>{S.dir.pick}</button>
            </div>
          </div>
          <div className="dir-browse-list">
            {parent && (
              <button type="button" className="dir-browse-item" onClick={() => void load(parent)}>
                ↩ {S.dir.up}
              </button>
            )}
            {!dirs.length && <div className="settings-empty">{S.dir.noSubdirs}</div>}
            {dirs.map((d) => (
              <button type="button" className="dir-browse-item" key={d.path} onClick={() => void load(d.path)}>
                <IconFolder /> {d.name}
              </button>
            ))}
          </div>
          <div className="composer-dir-foot">
            <button type="button" className="composer-dir-temp" onClick={() => void useTemp()}>{S.dir.useTemp}</button>
            <p>{S.dir.tempHint}</p>
          </div>
        </div>
      )}
    </div>
  );
}
