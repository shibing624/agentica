/**
 * Files a reply changed, with +/- line counts, from its tool calls.
 *
 * `apply_patch` carries the exact diff (one `*** Update/Add/Delete File:`
 * section per file, `+`/`-` lines after `@@`). `write_file` replaces the whole
 * file, so only the new line count is known: it reports `+N` and no removals.
 * Failed calls (result starting with "Error") changed nothing and are skipped.
 */
export type FileEdit = { path: string; added: number; removed: number; deleted?: boolean };

const FILE_HEADER_RE = /^\*\*\* (Update|Add|Delete) File: (.+?)\s*$/;
const MOVE_RE = /^\*\*\* Move to: (.+?)\s*$/;

export function fileEditsOf(steps?: Array<Record<string, any>>): FileEdit[] {
  const byPath = new Map<string, FileEdit>();
  const bump = (path: string, added: number, removed: number, deleted = false) => {
    const cur = byPath.get(path) || { path, added: 0, removed: 0 };
    cur.added += added;
    cur.removed += removed;
    if (deleted) cur.deleted = true;
    byPath.set(path, cur);
  };
  for (const st of steps || []) {
    if (st.type !== "tool" || !st.argsStr) continue;
    if (typeof st.result === "string" && /^Error\b/.test(st.result)) continue;
    let args: Record<string, unknown>;
    try { args = JSON.parse(st.argsStr); } catch { continue; }
    if (st.name === "write_file" && typeof args.file_path === "string" && typeof args.content === "string") {
      const content = args.content as string;
      const lines = content === "" ? 0 : content.replace(/\n$/, "").split("\n").length;
      bump(args.file_path as string, lines, 0);
    } else if (st.name === "apply_patch" && typeof args.patch === "string") {
      for (const e of patchEdits(args.patch as string)) bump(e.path, e.added, e.removed, e.deleted);
    }
  }
  return [...byPath.values()];
}

export function patchEdits(patch: string): FileEdit[] {
  const out: FileEdit[] = [];
  let cur: FileEdit | null = null;
  for (const line of patch.split("\n")) {
    const head = FILE_HEADER_RE.exec(line);
    if (head) {
      cur = { path: head[2]!, added: 0, removed: 0 };
      if (head[1] === "Delete") cur.deleted = true;
      out.push(cur);
      continue;
    }
    if (!cur) continue;
    const move = MOVE_RE.exec(line);
    if (move) { cur.path = move[1]!; continue; }
    if (line.startsWith("***") || line.startsWith("@@")) continue;
    if (line.startsWith("+")) cur.added += 1;
    else if (line.startsWith("-")) cur.removed += 1;
  }
  return out;
}
