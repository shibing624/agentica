import { useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { IconExternal, IconFileCode, IconFinder } from "../icons";
import { fileEditsOf, type FileEdit } from "../lib/fileEdits";
import {
  extractFilePaths, extractToolPaths, toWorkspaceRelative,
} from "../lib/file-path";
import { statWorkspaceFilesBatched } from "../lib/workspaceStat";
import { showToast } from "../store";

const MAX_VISIBLE = 3;
const MAX_VISIBLE_EDITED = 8;

function PathLabel({ path }: { path: string }) {
  const slash = path.lastIndexOf("/");
  const dir = slash >= 0 ? path.slice(0, slash + 1) : "";
  const name = slash >= 0 ? path.slice(slash + 1) : path;
  return (
    <span className="files-path">
      {dir && <span className="files-path-dir">{dir}</span>}
      <span className="files-path-name">{name}</span>
    </span>
  );
}

function absPath(workspace: string, rel: string) {
  return workspace.replace(/\/+$/, "") + "/" + rel;
}

/**
 * The files a reply touched, under the answer: what it edited (with +/- lines)
 * first, then other files it named. A row previews in the side panel; its
 * buttons open the file with the OS default app or show it in its folder.
 */
export function MessageFilesCard({
  text, steps, uploaded, workspace, onOpenFile, live,
}: {
  text: string;
  steps?: Array<Record<string, any>>;
  uploaded?: string[];
  workspace: string | null;
  onOpenFile: (path: string) => void;
  live?: boolean;
}) {
  const S = useStrings();
  const edits = useMemo(() => {
    const out: FileEdit[] = [];
    const seen = new Set<string>();
    for (const e of fileEditsOf(steps)) {
      const rel = toWorkspaceRelative(e.path, workspace);
      if (rel === null || seen.has(rel) || e.deleted) continue;
      seen.add(rel);
      out.push({ ...e, path: rel });
    }
    return out;
  }, [steps, workspace]);
  const candidates = useMemo(() => {
    const raw = [
      ...edits.map((e) => e.path),
      ...extractFilePaths(text),
      ...extractToolPaths(steps),
      ...(uploaded || []),
    ];
    const out: string[] = [];
    const seen = new Set<string>();
    for (const item of raw) {
      const rel = toWorkspaceRelative(item, workspace);
      if (rel === null || seen.has(rel)) continue;
      seen.add(rel);
      out.push(rel);
    }
    return out;
  }, [edits, text, steps, uploaded, workspace]);
  const sig = candidates.join("\0");

  const [existing, setExisting] = useState<string[]>([]);
  useEffect(() => {
    // Streaming tokens rewrite `text` every frame; the extracted path set
    // rarely changes, but a new array still retriggered POST /stat. Wait
    // until the turn settles, then one batched lookup.
    if (live) return;
    if (!workspace || !sig) {
      setExisting([]);
      return;
    }
    let cancelled = false;
    void statWorkspaceFilesBatched(workspace, candidates).then((found) => {
      if (!cancelled) setExisting(found);
    });
    return () => { cancelled = true; };
  }, [live, workspace, sig]);

  const [expanded, setExpanded] = useState(false);
  if (live || !workspace || existing.length === 0) return null;

  const present = new Set(existing);
  const edited = edits.filter((e) => present.has(e.path));
  const editedPaths = new Set(edited.map((e) => e.path));
  const mentioned = existing.filter((p) => !editedPaths.has(p));

  const open = (rel: string, app: "finder" | "reveal") => {
    const path = absPath(workspace, rel);
    void (app === "reveal" ? api.revealPathApi(path) : api.openPathApi(path, "finder"))
      .then(({ ok }) => { if (!ok) showToast(S.chat.openFailed, 2500); });
  };

  const row = (rel: string, edit?: FileEdit) => (
    <div key={rel} className="mfc-row">
      <button type="button" className="mfc-path" title={`${absPath(workspace, rel)}\n${S.chat.openPreview}`}
              onClick={() => onOpenFile(rel)}>
        <PathLabel path={rel} />
      </button>
      {edit && (
        <span className="mfc-diff">
          <span className="add">+{edit.added.toLocaleString()}</span>
          <span className="del">-{edit.removed.toLocaleString()}</span>
        </span>
      )}
      <span className="mfc-actions">
        <button type="button" title={S.chat.openExternal} onClick={() => open(rel, "finder")}><IconExternal /></button>
        <button type="button" title={S.chat.revealInFolder} onClick={() => open(rel, "reveal")}><IconFinder /></button>
      </span>
    </div>
  );

  const editedVisible = expanded ? edited : edited.slice(0, MAX_VISIBLE_EDITED);
  const mentionedVisible = expanded ? mentioned : mentioned.slice(0, MAX_VISIBLE);
  const hidden = edited.length - editedVisible.length + mentioned.length - mentionedVisible.length;

  return (
    <div className="msg-files-card">
      <div className="mfc-head">
        <span className="mfc-icon"><IconFileCode /></span>
        <div className="mfc-head-text">
          <div className="mfc-title">
            {edited.length ? S.chat.filesEdited(edited.length) : S.chat.filesMentioned(mentioned.length)}
          </div>
          <div className="mfc-sub">{S.chat.filesPreviewHint}</div>
        </div>
      </div>
      <div className="mfc-body">
        {editedVisible.map((e) => row(e.path, e))}
        {edited.length > 0 && mentionedVisible.length > 0 && (
          <div className="mfc-section">{S.chat.filesMentioned(mentioned.length)}</div>
        )}
        {mentionedVisible.map((p) => row(p))}
        {(hidden > 0 || expanded) && (edited.length > MAX_VISIBLE_EDITED || mentioned.length > MAX_VISIBLE) && (
          <button type="button" className="msg-files-more" onClick={() => setExpanded((v) => !v)}>
            {expanded ? S.chat.showLess : S.chat.showMoreFiles(hidden)}
          </button>
        )}
      </div>
    </div>
  );
}
