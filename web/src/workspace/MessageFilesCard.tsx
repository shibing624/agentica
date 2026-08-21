import { useEffect, useMemo, useState } from "react";
import { useStrings } from "../i18n";
import {
  extractFilePaths, extractToolPaths, toWorkspaceRelative,
} from "../lib/file-path";
import * as api from "../api";

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
  text, steps, uploaded, workspace, onOpenFile,
}: {
  text: string;
  steps?: Array<Record<string, any>>;
  uploaded?: string[];
  workspace: string | null;
  onOpenFile: (path: string) => void;
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

  const [paths, setPaths] = useState<string[] | null>(null);
  useEffect(() => {
    setPaths(null);
    if (!workspace || candidates.length === 0) return;
    let cancelled = false;
    void api.statWorkspaceFiles(workspace, candidates).then(({ ok, data }) => {
      if (cancelled || !ok) return;
      const existing = new Set(data?.existing || []);
      setPaths(candidates.filter((p) => existing.has(p)));
    });
    return () => { cancelled = true; };
  }, [candidates, workspace]);

  const [expanded, setExpanded] = useState(false);
  if (paths === null || paths.length === 0) return null;

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
