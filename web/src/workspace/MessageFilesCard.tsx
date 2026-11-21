import { useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { useStrings } from "../i18n";
import { FileTypeIcon, IconChevronDown, IconExternal, IconFileCode, IconFinder } from "../icons";
import type { FileEdit } from "../lib/fileEdits";
import { diffCounts, fileDiffsOf, type FileEdits } from "../lib/fileDiff";
import { isAbsOrHomePath } from "../lib/filePath";
import { messageEdits, mentionedFileCandidates } from "../lib/messageFiles";
import { baseName } from "../lib/panelTabs";
import { statWorkspaceFilesBatched } from "../lib/workspaceStat";
import { showToast } from "../store";

const MAX_VISIBLE = 3;
/** Edited rows shown before the fold (dsh's turn-changes card). */
const MAX_VISIBLE_EDITED = 4;

function Counts({ added, removed }: { added: number; removed: number }) {
  return <><span className="add">+{added.toLocaleString()}</span><span className="del">-{removed.toLocaleString()}</span></>;
}

function absPath(workspace: string, rel: string) {
  if (isAbsOrHomePath(rel)) return rel;
  return workspace.replace(/\/+$/, "") + "/" + rel;
}

/**
 * Recorded edits belong to the turn, even if a file isn't there anymore.
 * File mentions are independent and only shown after verifying the path.
 */
export function MessageFilesCard({
  text, steps, uploaded, mentionedFiles, workspace, sessionId, onOpenFile, onOpenDiff, live, role,
}: {
  text: string;
  steps?: Array<Record<string, any>>;
  uploaded?: string[];
  mentionedFiles?: string[];
  workspace: string | null;
  sessionId: string | null;
  onOpenFile: (path: string) => void;
  /** Open this round's review tab on `path`; `files` is every file the round changed. */
  onOpenDiff: (files: FileEdits[], path: string) => void;
  live?: boolean;
  role?: "user" | "assistant";
}) {
  const S = useStrings();
  const edits = useMemo(() => live ? [] : messageEdits(steps, workspace), [live, steps, workspace]);
  const diffs = useMemo(() => edits.length ? fileDiffsOf(steps, workspace) : new Map<string, FileEdits>(), [edits, steps, workspace]);
  const candidates = useMemo(
    () => live || mentionedFiles !== undefined ? [] : mentionedFileCandidates({ content: text, steps, files: uploaded }, workspace, edits),
    [live, mentionedFiles, text, steps, uploaded, workspace, edits],
  );
  const sig = JSON.stringify([sessionId, workspace, candidates]);
  const [fetched, setFetched] = useState<{ sig: string; found: string[] } | null>(null);
  const mentioned = mentionedFiles ?? (fetched?.sig === sig ? fetched.found : []);
  useEffect(() => {
    // Hydrated messages already have a first-paint snapshot. Only newly sent
    // messages need to resolve mentions here, once the turn settles.
    if (live || mentionedFiles !== undefined || !workspace || !candidates.length) return;
    let cancelled = false;
    void (sessionId ? api.fetchWorkspaceScope(sessionId) : Promise.resolve(null)).then((result) => {
      if (cancelled) return [];
      const scope = result?.ok && result.data ? result.data : undefined;
      return statWorkspaceFilesBatched(workspace, candidates, scope);
    }).then((found) => {
      if (!cancelled) setFetched({ sig, found });
    });
    return () => { cancelled = true; };
  }, [live, mentionedFiles, workspace, sessionId, sig, candidates]);

  const [expanded, setExpanded] = useState(false);
  const [mentionsExpanded, setMentionsExpanded] = useState(false);
  if (live || !workspace || (!edits.length && !mentioned.length)) return null;

  const edited = edits.map((edit) => {
    const diff = diffs.get(edit.path);
    return diff ? { ...edit, ...diffCounts(diff.edits) } : edit;
  });
  const reviewFiles = edited.flatMap(({ path }) => {
    const diff = diffs.get(path);
    return diff ? [diff] : [];
  });
  const pick = (path: string) => diffs.has(path) ? onOpenDiff(reviewFiles, path) : onOpenFile(path);

  const open = (rel: string, app: "finder" | "reveal") => {
    const path = absPath(workspace, rel);
    void (app === "reveal" ? api.revealPathApi(path) : api.openPathApi(path, "finder"))
      .then(({ ok }) => { if (!ok) showToast(S.chat.openFailed, 2500); });
  };

  const row = (rel: string, edit?: FileEdit) => {
    const diff = edit ? diffs.get(rel) : undefined;
    return <li key={rel} className="mfc-row">
      <button type="button" className="mfc-path" title={absPath(workspace, rel)}
              aria-label={`${diff ? S.chat.openDiff : S.chat.openPreview}: ${rel}`}
              onClick={() => diff ? onOpenDiff(reviewFiles, rel) : onOpenFile(rel)}>
        {!edit && <FileTypeIcon path={rel} size={15} />}
        <span className="files-path">{rel}</span>
        {edit && <span className="mfc-diff"><Counts added={edit.added} removed={edit.removed} /></span>}
      </button>
      {!edit && <span className="mfc-actions">
        <button type="button" title={S.chat.openExternal} onClick={() => open(rel, "finder")}><IconExternal /></button>
        <button type="button" title={S.chat.revealInFolder} onClick={() => open(rel, "reveal")}><IconFinder /></button>
      </span>}
    </li>;
  };

  const single = edited.length === 1 ? edited[0] : undefined;
  const added = edited.reduce((n, e) => n + e.added, 0);
  const removed = edited.reduce((n, e) => n + e.removed, 0);
  const mentionedTitle = role === "user"
    ? S.chat.filesMentionedQuery(mentioned.length)
    : S.chat.filesMentioned(mentioned.length);
  const fold = (open: boolean, toggle: () => void, count: number) => (
    <button type="button" className={`msg-files-more${open ? " open" : ""}`} aria-expanded={open} onClick={toggle}>
      <span>{open ? S.chat.showLess : S.chat.allFiles(count)}</span><IconChevronDown />
    </button>
  );

  return <>
    {edited.length > 0 && <div className="msg-files-card mfc-edits">
      <button type="button" className="mfc-head mfc-head-btn"
              title={single ? absPath(workspace, single.path) : S.chat.openRoundReview}
              onClick={() => pick(single?.path ?? reviewFiles[0]?.path ?? edited[0].path)}>
        <span className="mfc-icon"><FileTypeIcon path={single?.path} kind={single ? undefined : "code"} size={20} /></span>
        <span className="mfc-head-text">
          <span className="mfc-title">{single ? S.chat.fileEdited(baseName(single.path)) : S.chat.filesEdited(edited.length)}</span>
          <span className="mfc-sub">
            <span className="mfc-total"><Counts added={added} removed={removed} /></span>
            <span className="mfc-hint">{S.chat.filesPreviewHint}</span>
          </span>
        </span>
      </button>
      {!single && <ul className="mfc-body">{(expanded ? edited : edited.slice(0, MAX_VISIBLE_EDITED)).map((e) => row(e.path, e))}</ul>}
      {edited.length > MAX_VISIBLE_EDITED && fold(expanded, () => setExpanded((v) => !v), edited.length)}
    </div>}
    {mentioned.length > 0 && <div className="msg-files-card mfc-mentions">
      <div className="mfc-head"><span className="mfc-icon"><IconFileCode /></span><span className="mfc-title">{mentionedTitle}</span></div>
      <ul className="mfc-body">{(mentionsExpanded ? mentioned : mentioned.slice(0, MAX_VISIBLE)).map((path) => row(path))}</ul>
      {mentioned.length > MAX_VISIBLE && fold(mentionsExpanded, () => setMentionsExpanded((v) => !v), mentioned.length)}
    </div>}
  </>;
}
