/** CLI-aligned one-liner / brief for a tool call. Port of
 *  ``agentica.cli.display.tool_format.format_tool_display``. Trace pages keep
 *  raw JSON; the chat row is what users read. */

export function parseToolArgs(argsStr: string): Record<string, unknown> {
  if (!argsStr) return {};
  try {
    const parsed = JSON.parse(argsStr);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch { /* raw text */ }
  return {};
}

export type ToolDisplayLayout = {
  header: string;
  body: string;
  bodyKind: "args" | "call";
};

const BODY_ONLY_DISPLAYS = new Set(["write_todos", "list_agents", "ask_user_question"]);

export function layoutToolDisplay(name: string, display: string): ToolDisplayLayout {
  if (name === "execute") {
    return { header: "", body: display, bodyKind: "call" };
  }
  if (BODY_ONLY_DISPLAYS.has(name)) {
    return { header: "", body: display, bodyKind: "args" };
  }
  const [header = "", ...rest] = display.split("\n");
  return { header, body: rest.join("\n"), bodyKind: "args" };
}

export function formatToolDisplay(name: string, args: Record<string, unknown>, cwd = ""): string {
  if (name === "read_file") {
    const filePath = String(args.file_path ?? "");
    if (args.tail != null) return `${shortenPath(filePath, cwd)} (tail ${Number(args.tail)})`;
    const offset = Number(args.offset ?? 0) || 0;
    const limit = args.limit == null ? 500 : Number(args.limit) || 500;
    return `${shortenPath(filePath, cwd)} (${lineRange(offset, limit)})`;
  }
  if (name === "write_file") {
    return basename(String(args.file_path ?? ""));
  }
  if (name === "apply_patch") {
    const patch = String(args.patch ?? "");
    const count = (patch.match(/^\*\*\* (?:Add|Update|Delete) File: /gm) || []).length;
    return count ? `${count} ${count === 1 ? "file" : "files"}` : "";
  }
  if (name === "execute") {
    return shortenPathsInCommand(String(args.command ?? ""), cwd);
  }
  if (name === "write_todos") {
    const todos = args.todos;
    if (Array.isArray(todos) && todos.length) {
      return todos.map((todo) => {
        if (todo && typeof todo === "object") {
          const item = todo as Record<string, unknown>;
          const status = String(item.status ?? "pending");
          const icon = status === "completed" ? "✓" : status === "pending" ? "○" : "◐";
          return `${icon} ${String(item.content ?? "")}`;
        }
        return `○ ${String(todo)}`;
      }).join("\n    ");
    }
    return Array.isArray(todos) ? `${todos.length} items` : "";
  }
  if (name === "web_search") {
    const queries = args.queries;
    if (Array.isArray(queries)) {
      return queries.slice(0, 3).map((q) => String(q).slice(0, 40)).join(", ");
    }
    return String(queries ?? "").slice(0, 80);
  }
  if (name === "fetch_url") {
    const url = String(args.url ?? "");
    return url.length > 60 ? url.slice(0, 57) + "..." : url;
  }
  if (name === "glob") {
    const pattern = String(args.pattern ?? "*");
    const path = String(args.path ?? ".");
    return `${pattern} in ${shortenPath(path, cwd)}`;
  }
  if (name === "grep") {
    const pattern = String(args.pattern ?? "");
    const path = String(args.path ?? ".");
    const include = String(args.include ?? "");
    let display = `'${pattern.slice(0, 40)}' in ${shortenPath(path, cwd)}`;
    if (include) display += ` (${include})`;
    return display;
  }
  if (name === "task") {
    return formatHandoff(args, "description", ["subagent_type", "timeout", "max_turns", "resume_from_run_id"]);
  }
  if (name === "delegate") {
    return formatHandoff(args, "task", ["label", "work_dir", "model"]);
  }
  if (name === "send_message") {
    const target = String(args.target ?? "");
    const message = String(args.message ?? "");
    if (!message) return target ? `→ ${target}` : "";
    return target ? `→ ${target}\n    ${message}` : message;
  }
  if (name === "list_agents" || name === "ask_user_question") {
    return "";
  }

  const brief: string[] = [];
  for (const [key, value] of Object.entries(args)) {
    if (typeof value === "string") {
      brief.push(`${key}=${reprStr(value.length > 40 ? value.slice(0, 37) + "..." : value)}`);
    } else if (typeof value === "number" || typeof value === "boolean") {
      brief.push(`${key}=${value}`);
    } else if (Array.isArray(value)) {
      brief.push(`${key}=[${value.length} items]`);
    } else if (value && typeof value === "object") {
      brief.push(`${key}={...}`);
    }
  }
  const shown = brief.slice(0, 3).join(", ");
  return brief.length > 3 ? shown + ", ..." : shown;
}

function formatHandoff(args: Record<string, unknown>, bodyKey: string, metaKeys: string[]): string {
  const body = String(args[bodyKey] ?? "");
  const meta: string[] = [];
  for (const key of metaKeys) {
    const value = args[key];
    if (value == null || value === "") continue;
    meta.push(typeof value === "string" ? `${key}=${reprStr(value)}` : `${key}=${value}`);
  }
  if (body) {
    const indented = body.split("\n").join("\n    ");
    return meta.length ? meta.join(", ") + "\n    " + indented : indented;
  }
  return meta.join(", ");
}

function reprStr(value: string): string {
  return `'${value.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
}

function basename(filePath: string): string {
  const s = filePath.replace(/\\/g, "/");
  const i = s.lastIndexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}

function lineRange(offset: number, limit: number): string {
  if (offset < 0) {
    const keep = Math.abs(offset);
    const take = limit || keep;
    return take >= keep ? `last ${keep}` : `oldest ${take} of last ${keep}`;
  }
  const start = offset ? offset + 1 : 1;
  const end = start + (limit || 500) - 1;
  return `L${start}-${end}`;
}

function shortenPath(filePath: string, cwd: string): string {
  if (!filePath || filePath === ".") return ".";
  const cwdN = cwd.replace(/\\/g, "/").replace(/\/$/, "");
  const p = filePath.replace(/\\/g, "/");
  if (cwdN && (p === cwdN || p.startsWith(cwdN + "/"))) {
    return p === cwdN ? "." : p.slice(cwdN.length + 1);
  }
  return p;
}

function shortenPathsInCommand(command: string, cwd: string): string {
  if (!cwd) return command;
  const cwdN = cwd.replace(/\\/g, "/").replace(/\/$/, "");
  if (cwdN && command.includes(cwdN)) {
    return command.split(cwdN + "/").join("").split(cwdN).join(".");
  }
  return command;
}
