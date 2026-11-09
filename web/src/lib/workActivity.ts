/**
 * What a run of thinking + tool rows did, in words (deepseek-harness style):
 * each tool maps to a category, the folded group header names the top three
 * categories ("Ran commands and read files, searched code"), a running group
 * names what is happening now. Pure, so the chat rows and tests share it.
 */
import type { MsgPart } from "../store";

export type Category =
  | "thinking" | "read" | "search" | "edit" | "write" | "commands" | "webSearch" | "webFetch"
  | "subagents" | "plan" | "questions" | "tools";

const BY_NAME: Record<string, Category> = {
  read_file: "read", read_image: "read", ls: "read", list_dir: "read",
  glob: "search", grep: "search", search_session: "search", search_files: "search",
  apply_patch: "edit", edit_file: "edit", multi_edit: "edit",
  write_file: "write",
  execute: "commands", shell: "commands", bash: "commands", run_python: "commands", python: "commands",
  web_search: "webSearch", fetch_url: "webFetch", web_fetch: "webFetch",
  task: "subagents", delegate: "subagents", send_message: "subagents", list_agents: "subagents",
  write_todos: "plan", goal: "plan",
  ask_user_question: "questions",
};

export function toolCategory(name: string): Category {
  return BY_NAME[name] || "tools";
}

export type ActivityStrings = {
  /** Running label per category, e.g. "Running commands". */
  actRunning: Record<Category, string>;
  /** Done label per category, e.g. "Ran commands". */
  actDone: Record<Category, string>;
  /** Join two / three labels: "a and b" / "a, b, c" / overflow suffix. */
  actJoin2: (a: string, b: string) => string;
  actJoinMore: (labels: string[], more: boolean) => string;
};

/** Categories this group used, most used first, ties in first-seen order. */
export function groupCategories(items: MsgPart[]): Category[] {
  const count = new Map<Category, number>();
  for (const p of items) {
    if (p.kind !== "tool") continue;
    const c = toolCategory(p.name);
    count.set(c, (count.get(c) || 0) + 1);
  }
  if (!count.size && items.some((p) => p.kind === "think")) return ["thinking"];
  return [...count.entries()].sort((a, b) => b[1] - a[1]).map(([c]) => c);
}

/** Folded header for a finished group: top three done labels, no numbers. */
export function doneTitle(items: MsgPart[], S: ActivityStrings): string {
  const cats = groupCategories(items);
  const labels = cats.slice(0, 3).map((c) => S.actDone[c]);
  if (labels.length <= 1) return labels[0] || S.actDone.thinking;
  if (labels.length === 2 && cats.length === 2) return S.actJoin2(labels[0], labels[1]);
  return S.actJoinMore(labels, cats.length > 3);
}

/** The newest unfinished tool, else null (only thinking is running). */
export function runningTool(items: MsgPart[]): Extract<MsgPart, { kind: "tool" }> | null {
  for (let i = items.length - 1; i >= 0; i--) {
    const p = items[i];
    if (p.kind === "tool" && p.result == null) return p;
  }
  return null;
}

/** First non-empty line, markdown emphasis stripped, for one-line previews. */
export function firstLine(text: string): string {
  for (const raw of text.split("\n")) {
    const line = raw.replace(/\*\*|__|`/g, "").replace(/^#+\s*/, "").trim();
    if (line) return line;
  }
  return "";
}

/** Tools whose collapsed chat row already says enough without inline output. */
const QUIET_RESULT_TOOLS = new Set([
  "read_file", "glob", "grep", "write_file", "apply_patch", "edit_file", "multi_edit",
  "write_todos", "task", "ask_user_question",
]);

/**
 * Compact one-line result preview for a settled tool row (`command → output`),
 * capped at 512 chars so CSS ellipsis does the visual clipping while copy/paste
 * still captures the text after the ellipsis.
 */
export function toolResultPreview(name: string, resultText: string, hasDiff = false): string {
  if (!resultText || hasDiff || QUIET_RESULT_TOOLS.has(name)) return "";
  const flat = resultText.replace(/\s+/g, " ").trim();
  if (!flat) return "";
  return flat.length > 512 ? flat.slice(0, 512) + "…" : flat;
}

/** `+added -removed` of a unified diff (file headers excluded). */
export function diffStats(diff: string): { added: number; removed: number } | null {
  if (!diff) return null;
  let added = 0;
  let removed = 0;
  for (const line of diff.split("\n")) {
    if (line.startsWith("+++") || line.startsWith("---")) continue;
    if (line.startsWith("+")) added += 1;
    else if (line.startsWith("-")) removed += 1;
  }
  return added || removed ? { added, removed } : null;
}

export type TaskResult = {
  success: boolean;
  status: string;
  subagentName: string;
  subagentType: string;
  model?: string;
  modelTier?: string;
  executionMs?: number;
  toolCount: number;
  steps: Array<{ name: string; info: string }>;
  stepsTruncated?: number;
  result: string;
  error: string;
  runId?: string;
};

/** The `task` tool's JSON result; null when it is not one (e.g. truncated). */
export function parseTaskResult(text: string | undefined): TaskResult | null {
  if (!text) return null;
  let data: any;
  try { data = JSON.parse(text); } catch { return null; }
  if (!data || typeof data !== "object" || !("success" in data)) return null;
  const steps = Array.isArray(data.tool_calls_summary)
    ? data.tool_calls_summary.filter((s: any) => s && typeof s === "object")
      .map((s: any) => ({ name: String(s.name || ""), info: String(s.info || "") }))
    : [];
  return {
    success: !!data.success,
    status: String(data.status || (data.success ? "completed" : "error")),
    subagentName: String(data.subagent_name || data.subagent_type || ""),
    subagentType: String(data.subagent_type || ""),
    model: data.model ? String(data.model) : undefined,
    modelTier: data.model_tier ? String(data.model_tier) : undefined,
    executionMs: typeof data.execution_time === "number" ? Math.round(data.execution_time * 1000) : undefined,
    toolCount: Number(data.tool_count || steps.length) || 0,
    steps,
    result: String(data.result || ""),
    error: String(data.error || ""),
    runId: data.run_id ? String(data.run_id) : undefined,
  };
}

export type Todo = { content: string; status: "pending" | "in_progress" | "completed" };

export function parseTodos(args: Record<string, unknown>): Todo[] {
  const raw = args.todos;
  if (!Array.isArray(raw)) return [];
  return raw.map((t) => {
    if (t && typeof t === "object") {
      const o = t as Record<string, unknown>;
      const st = String(o.status || "pending");
      return {
        content: String(o.content || ""),
        status: st === "completed" ? "completed" : st === "in_progress" ? "in_progress" : "pending",
      } as Todo;
    }
    return { content: String(t), status: "pending" } as Todo;
  });
}

export function todoProgress(todos: Todo[]): { done: number; total: number; active: Todo | null } {
  return {
    done: todos.filter((t) => t.status === "completed").length,
    total: todos.length,
    active: todos.find((t) => t.status === "in_progress") || null,
  };
}

/**
 * The plan of the latest turn: the newest `write_todos` list written after the
 * last real user query (steer chips and automated reports do not start a new
 * question). A plan left unfinished by an interrupted turn stays with that
 * turn instead of riding along under every later question.
 */
export function latestTodos(
  msgs: Array<{ role?: string; steer?: boolean; automated?: boolean; parts?: MsgPart[] }>,
): Todo[] {
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m.role === "user" && !m.steer && !m.automated) break;
    const parts = m.parts || [];
    for (let j = parts.length - 1; j >= 0; j--) {
      const p = parts[j];
      if (p.kind !== "tool" || p.name !== "write_todos") continue;
      if (p.result != null && String(p.result).startsWith("Error: ")) continue;
      try {
        const args = JSON.parse(p.argsStr || "{}");
        if (args && typeof args === "object" && !Array.isArray(args)) return parseTodos(args);
      } catch { /* not JSON yet: still streaming */ }
    }
  }
  return [];
}

/**
 * The plan above the composer: only while a turn is running, as dsh clears
 * its dock when the next turn starts. A finished turn's plan, often left with
 * its last step `in_progress` because the model answered instead of writing
 * the list once more, stays in the transcript's task card and leaves the dock.
 */
export function dockTodos(
  msgs: Array<{ role?: string; steer?: boolean; automated?: boolean; parts?: MsgPart[] }>,
  running: boolean,
): Todo[] {
  return running ? latestTodos(msgs) : [];
}
