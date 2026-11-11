import { fileEditsOf, type FileEdit } from "./fileEdits";
import { extractFilePaths, extractToolPaths, normalizeMentionedPath, toWorkspaceRelative } from "./filePath";

/** Files the reply edited, workspace-relative, first edit of each path, deletions dropped. */
export function messageEdits(steps: Array<Record<string, any>> | undefined, workspace: string | null): FileEdit[] {
  const out: FileEdit[] = [];
  const seen = new Set<string>();
  for (const e of fileEditsOf(steps)) {
    const rel = toWorkspaceRelative(e.path, workspace);
    if (rel === null || seen.has(rel) || e.deleted) continue;
    seen.add(rel);
    out.push({ ...e, path: rel });
  }
  return out;
}

/** Every path a message's files card may list; the card shows the ones that exist. */
export function messageFileCandidates(
  m: { content?: string; steps?: Array<Record<string, any>>; files?: string[] },
  workspace: string | null,
): string[] {
  const raw = [
    ...messageEdits(m.steps, workspace).map((e) => e.path),
    ...extractFilePaths(m.content || ""),
    ...extractToolPaths(m.steps),
    ...(m.files || []),
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
}
