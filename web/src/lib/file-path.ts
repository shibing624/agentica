/**
 * Whether inline code in a message looks like a file path (penguin's heuristic).
 * No whitespace, no URL scheme, ends with a known extension.
 */
const KNOWN_EXTENSIONS = new Set([
  "txt", "md", "json", "js", "mjs", "cjs", "ts", "tsx", "jsx", "py", "sh", "bash",
  "yaml", "yml", "toml", "css", "html", "htm", "csv", "log", "xml", "ini", "conf",
  "rs", "go", "java", "c", "h", "cpp", "hpp", "sql", "rb", "php",
  "png", "jpg", "jpeg", "gif", "webp", "svg", "pdf",
]);

const URL_SCHEME_RE = /^[a-z][a-z0-9+.-]*:\/\//i;
const MAX_PATH_LEN = 512;
const FILE_ARG_KEYS = ["file_path", "path", "target_file", "file"];

export function isFilePathLike(text: string): boolean {
  const s = text.trim();
  if (s.length === 0 || s.length > 200) return false;
  if (/\s/.test(s)) return false;
  if (URL_SCHEME_RE.test(s) || s.startsWith("www.")) return false;
  const m = /\.([A-Za-z0-9]{1,8})$/.exec(s);
  if (!m) return false;
  const ext = m[1]!.toLowerCase();
  if (/^\d+$/.test(ext)) return false;
  return KNOWN_EXTENSIONS.has(ext);
}

export function joinWorkspacePath(base: string, name: string): string {
  return base === "" ? name : `${base}/${name}`;
}

export function extractFilePaths(markdown: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const m of markdown.matchAll(/`([^`\n]+)`/g)) {
    const text = m[1]!.trim();
    if (!isFilePathLike(text) || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

export function extractToolPaths(steps?: Array<Record<string, any>>): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const st of steps || []) {
    if (st.type !== "tool" || !st.argsStr) continue;
    let args: Record<string, unknown>;
    try { args = JSON.parse(st.argsStr); } catch { continue; }
    for (const key of FILE_ARG_KEYS) {
      const v = args[key];
      if (typeof v !== "string" || !v || seen.has(v)) continue;
      seen.add(v);
      out.push(v);
    }
  }
  return out;
}

/** Normalize a mentioned path to workspace-relative, or null if it escapes. */
export function toWorkspaceRelative(path: string, workspace: string | null): string | null {
  const s = path.trim();
  if (s.length === 0 || s.length > MAX_PATH_LEN) return null;
  if (s.startsWith("~")) return null;
  const ws = workspace && workspace.length > 0 ? workspace.replace(/\/+$/, "") : null;
  const absolute = s.startsWith("/");
  let rel = s;
  if (absolute) {
    if (ws === null || !s.startsWith(ws + "/")) return null;
    rel = s.slice(ws.length + 1);
  }
  const stack: string[] = [];
  for (const seg of rel.split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      if (stack.length === 0) return null;
      stack.pop();
      continue;
    }
    stack.push(seg);
  }
  return stack.length > 0 ? stack.join("/") : null;
}
