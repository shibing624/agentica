/**
 * The review side of a reply's edits: the unified diffs its write tools
 * reported (`diff -- path` then `@@` hunks, see `files_unified_diff`), parsed
 * per file and laid out the way dsh's FileDiff draws them. Pure, so the line
 * numbering and pairing rules are testable without React.
 */
import { toWorkspaceRelative } from "./file-path";

export type DiffHunk = {
  oldStart: number; oldLines: number; newStart: number; newLines: number;
  /** Each line keeps its ` ` / `+` / `-` prefix. */
  lines: string[];
};
export type FileDiff = { path: string; hunks: DiffHunk[] };

const FILE_RE = /^diff -- (.+)$/;
const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;

/** `diff -- path` blocks of one tool call. A clipped tail ("... (N more
 *  chars)") and anything else outside a hunk is dropped. */
export function parseFileDiffs(text: string): FileDiff[] {
  const out: FileDiff[] = [];
  let file: FileDiff | null = null;
  let hunk: DiffHunk | null = null;
  for (const line of text.split("\n")) {
    const head = FILE_RE.exec(line);
    if (head) {
      file = { path: head[1]!, hunks: [] };
      out.push(file);
      hunk = null;
      continue;
    }
    const h = HUNK_RE.exec(line);
    if (h && file) {
      // difflib omits the count when it is 1; an empty side is written `N,0`.
      hunk = {
        oldStart: Number(h[1]), oldLines: h[2] === undefined ? 1 : Number(h[2]),
        newStart: Number(h[3]), newLines: h[4] === undefined ? 1 : Number(h[4]),
        lines: [],
      };
      file.hunks.push(hunk);
      continue;
    }
    if (hunk && (line.startsWith(" ") || line.startsWith("+") || line.startsWith("-"))) hunk.lines.push(line);
  }
  return out.filter((f) => f.hunks.length > 0);
}

/** One file's edits in a reply, one entry per tool call that touched it, in order. */
export type FileEdits = { path: string; edits: DiffHunk[][] };

export function fileDiffsOf(steps: Array<Record<string, any>> | undefined, workspace: string | null): Map<string, FileEdits> {
  const out = new Map<string, FileEdits>();
  for (const st of steps || []) {
    if (st.type !== "tool" || typeof st.diff !== "string" || !st.diff) continue;
    for (const f of parseFileDiffs(st.diff)) {
      const rel = toWorkspaceRelative(f.path, workspace);
      if (rel === null) continue;
      const cur = out.get(rel) || { path: rel, edits: [] };
      cur.edits.push(f.hunks);
      out.set(rel, cur);
    }
  }
  return out;
}

export function diffCounts(edits: DiffHunk[][]): { added: number; removed: number } {
  let added = 0;
  let removed = 0;
  for (const hunks of edits) {
    for (const h of hunks) {
      for (const l of h.lines) {
        if (l[0] === "+") added++;
        else if (l[0] === "-") removed++;
      }
    }
  }
  return { added, removed };
}

export type RowKind = "add" | "del" | "context";
export type DiffRow = { kind: RowKind; old?: number; new?: number; text: string };

/** Number a hunk's lines: context counts on both sides, deletions on the old
 *  side, additions on the new side. */
export function hunkRows(h: DiffHunk): DiffRow[] {
  let oldNo = h.oldStart;
  let newNo = h.newStart;
  return h.lines.map((line) => {
    const text = line.slice(1);
    if (line[0] === "+") return { kind: "add", new: newNo++, text };
    if (line[0] === "-") return { kind: "del", old: oldNo++, text };
    return { kind: "context", old: oldNo++, new: newNo++, text };
  });
}

export type SplitCell = { no: number; text: string; kind: RowKind };
export type SplitRow = { left?: SplitCell; right?: SplitCell };

/** Side-by-side rows: each run of deletions lines up with the run of
 *  additions after it, row by row; context sits on both sides. */
export function splitRows(h: DiffHunk): SplitRow[] {
  const rows: SplitRow[] = [];
  let dels: SplitCell[] = [];
  let adds: SplitCell[] = [];
  const flush = () => {
    for (let i = 0; i < Math.max(dels.length, adds.length); i++) {
      const row: SplitRow = {};
      if (dels[i]) row.left = dels[i];
      if (adds[i]) row.right = adds[i];
      rows.push(row);
    }
    dels = [];
    adds = [];
  };
  for (const r of hunkRows(h)) {
    if (r.kind === "del") dels.push({ no: r.old!, text: r.text, kind: "del" });
    else if (r.kind === "add") adds.push({ no: r.new!, text: r.text, kind: "add" });
    else {
      flush();
      rows.push({ left: { no: r.old!, text: r.text, kind: "context" }, right: { no: r.new!, text: r.text, kind: "context" } });
    }
  }
  flush();
  return rows;
}

export function hunkHeader(h: DiffHunk): string {
  return `@@ -${h.oldStart},${h.oldLines} +${h.newStart},${h.newLines} @@`;
}

/** Only additions, or only deletions: a second column would be empty. */
export function oneSided(edits: DiffHunk[][]): boolean {
  const { added, removed } = diffCounts(edits);
  return added === 0 || removed === 0;
}

/** Cap on drawn lines per file (dsh's MAX_RENDERED_LINES). */
export const MAX_RENDERED_LINES = 5000;

/** The edits to draw, cut at `max` lines in total. */
export function renderedEdits(edits: DiffHunk[][], max = MAX_RENDERED_LINES): { edits: DiffHunk[][]; truncated: boolean } {
  let budget = max;
  let truncated = false;
  const kept: DiffHunk[][] = [];
  for (const hunks of edits) {
    if (budget === 0) { truncated = true; break; }
    const keptHunks: DiffHunk[] = [];
    for (const h of hunks) {
      if (budget === 0) { truncated = true; break; }
      if (h.lines.length <= budget) keptHunks.push(h);
      else { keptHunks.push({ ...h, lines: h.lines.slice(0, budget) }); truncated = true; }
      budget -= Math.min(budget, h.lines.length);
    }
    kept.push(keptHunks);
  }
  return { edits: kept, truncated };
}
