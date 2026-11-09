import { useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { FileTypeIcon, IconChevronDown, IconExternal, IconFileCode, IconFinder } from "../icons";
import { fileEditsOf, type FileEdit } from "../lib/fileEdits";
import { fileDiffsOf, type FileEdits } from "../lib/fileDiff";
import {
  extractFilePaths, extractToolPaths, isAbsOrHomePath, normalizeMentionedPath, toWorkspaceRelative,
} from "../lib/filePath";
import { statWorkspaceFilesBatched } from "../lib/workspaceStat";
import { showToast } from "../store";

const MAX_VISIBLE = 3;
/** Edited rows shown before the fold (dsh's turn-changes card). */
const MAX_VISIBLE_EDITED = 4;

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
  if (isAbsOrHomePath(rel)) return rel;
  return workspace.replace(/\/+$/, "") + "/" + rel;
}

/**
 * The files a reply touched, under the answer: what it edited (with +/- lines)
 * first, then other files it named. An edited row opens its diff in the side
 * panel (a file the reply only named opens its preview); the buttons open the
 * file with the OS default app or show it in its folder.
 */
export function MessageFilesCard({
  text, steps, uploaded, workspace, onOpenFile, onOpenDiff, live, role,
}: {
  text: string;
  steps?: Array<Record<string, any>>;
  uploaded?: string[];
  workspace: string | null;
  onOpenFile: (path: string) => void;
  /** Open this round's review tab on `path`; `files` is every file the round changed. */
  onOpenDiff: (files: FileEdits[], path: string) => void;
  live?: boolean;
  role?: "user" | "assistant";
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
  const diffs = useMemo(() => fileDiffsOf(steps, workspace), [steps, workspace]);
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
      const rel = normalizeMentionedPath(item, workspace);
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

  const row = (rel: string, edit?: FileEdit) => {
    const diff = edit ? diffs.get(rel) : undefined;
    return (
    <div key={rel} className="mfc-row">
      <button type="button" className="mfc-path"
              title={`${absPath(workspace, rel)}\n${diff ? S.chat.openDiff : S.chat.openPreview}`}
              onClick={() => (diff ? onOpenDiff(reviewFiles, rel) : onOpenFile(rel))}>
        <FileTypeIcon path={rel} size={15} />
        <PathLabel path={rel} />
      </button>
      <span className="mfc-actions">
        <button type="button" title={S.chat.openExternal} onClick={() => open(rel, "finder")}><IconExternal /></button>
        <button type="button" title={S.chat.revealInFolder} onClick={() => open(rel, "reveal")}><IconFinder /></button>
      </span>
      {edit && (
        <span className="mfc-diff">
          <span className="add">+{edit.added.toLocaleString()}</span>
          <span className="del">-{edit.removed.toLocaleString()}</span>
        </span>
      )}
    </div>
    );
  };

  // The review tab lists the edited files in the card's order.
  const reviewFiles = edited.flatMap((e) => {
    const d = diffs.get(e.path);
    return d ? [d] : [];
  });

  const editedVisible = expanded ? edited : edited.slice(0, MAX_VISIBLE_EDITED);
  const mentionedVisible = expanded ? mentioned : mentioned.slice(0, MAX_VISIBLE);
  const foldable = edited.length > MAX_VISIBLE_EDITED || mentioned.length > MAX_VISIBLE;
  const total = edited.length + mentioned.length;
  // The round's whole edit volume, so the header says how big the turn was.
  const added = edited.reduce((n, e) => n + e.added, 0);
  const removed = edited.reduce((n, e) => n + e.removed, 0);
  const mentionedTitle = role === "user"
    ? S.chat.filesMentionedQuery(mentioned.length)
    : S.chat.filesMentioned(mentioned.length);
  const headText = (
    <span className="mfc-head-text">
      <span className="mfc-title">
        {edited.length ? S.chat.filesEdited(edited.length) : mentionedTitle}
      </span>
      <span className="mfc-sub">
        {edited.length > 0 && (
          <span className="mfc-total">
            <span className="add">+{added.toLocaleString()}</span>
            <span className="del">-{removed.toLocaleString()}</span>
          </span>
        )}
        <span className="mfc-hint">{S.chat.filesPreviewHint}</span>
      </span>
    </span>
  );

  return (
    <div className="msg-files-card">
      {reviewFiles.length > 0 ? (
        // Like dsh, the header opens the round's review on its first file.
        <button type="button" className="mfc-head mfc-head-btn" title={S.chat.openRoundReview}
                onClick={() => onOpenDiff(reviewFiles, reviewFiles[0].path)}>
          <span className="mfc-icon"><IconFileCode /></span>
          {headText}
        </button>
      ) : (
        <div className="mfc-head">
          <span className="mfc-icon"><IconFileCode /></span>
          {headText}
        </div>
      )}
      <div className="mfc-body">
        {editedVisible.map((e) => row(e.path, e))}
        {edited.length > 0 && mentionedVisible.length > 0 && (
          <div className="mfc-section">{mentionedTitle}</div>
        )}
        {mentionedVisible.map((p) => row(p))}
        {foldable && (
          <button type="button" className={`msg-files-more${expanded ? " open" : ""}`} aria-expanded={expanded}
                  onClick={() => setExpanded((v) => !v)}>
            <span>{expanded ? S.chat.showLess : S.chat.allFiles(total)}</span>
            <IconChevronDown />
          </button>
        )}
      </div>
    </div>
  );
}
