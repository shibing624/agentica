import { useEffect, useMemo, useState } from "react";
import { useStrings } from "../i18n";
import {
  extractFilePaths, extractToolPaths, toWorkspaceRelative,
} from "../lib/file-path";
import { statWorkspaceFilesBatched } from "../lib/workspaceStat";

const MAX_VISIBLE = 3;

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
  const candidates = useMemo(() => {
    const raw = [
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
  }, [text, steps, uploaded, workspace]);
  const sig = candidates.join("\0");

  const [paths, setPaths] = useState<string[]>([]);
  useEffect(() => {
    // Streaming tokens rewrite `text` every frame; the extracted path set
    // rarely changes, but a new array still retriggered POST /stat. Wait
    // until the turn settles, then one batched lookup.
    if (live) return;
    if (!workspace || !sig) {
      setPaths([]);
      return;
    }
    let cancelled = false;
    void statWorkspaceFilesBatched(workspace, candidates).then((existing) => {
      if (!cancelled) setPaths(existing);
    });
    return () => { cancelled = true; };
  }, [live, workspace, sig]);

  const [expanded, setExpanded] = useState(false);
  if (live || paths.length === 0) return null;

  const visible = expanded ? paths : paths.slice(0, MAX_VISIBLE);
  const hidden = paths.length - visible.length;

  return (
    <div className="msg-files-card">
      <div className="msg-files-head">{S.chat.filesInMessage(paths.length)}</div>
      {visible.map((path) => (
        <button key={path} type="button" className="msg-files-row" title={path}
                onClick={() => onOpenFile(path)}>
          <PathLabel path={path} />
          <span className="msg-files-preview">{S.chat.openPreview}</span>
        </button>
      ))}
      {(hidden > 0 || expanded) && paths.length > MAX_VISIBLE && (
        <button type="button" className="msg-files-more"
                onClick={() => setExpanded((v) => !v)}>
          {expanded ? S.chat.showLess : S.chat.showMoreFiles(hidden)}
        </button>
      )}
    </div>
  );
}
