/**
 * Whether inline code in a message looks like a file path (penguin's heuristic).
 * No whitespace, no URL scheme, ends with a known extension.
 */
const KNOWN_EXTENSIONS = new Set([
  "txt", "md", "json", "js", "mjs", "cjs", "ts", "tsx", "jsx", "py", "sh", "bash",
  "yaml", "yml", "toml", "css", "html", "htm", "csv", "log", "xml", "ini", "conf",
  "rs", "go", "java", "c", "h", "cpp", "hpp", "sql", "rb", "php",
  "png", "jpg", "jpeg", "gif", "webp", "svg", "pdf",
  "tex", "latex", "bib", "sty", "cls", "rst", "adoc", "mdx",
  "jsonl", "ndjson", "jsonc", "ipynb", "mts", "cts", "pyi", "pyw",
  "kt", "kts", "swift", "scss", "less", "vue", "svelte", "diff", "patch",
  "mk", "cfg", "properties", "tsv",
]);

export const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "ico"]);
export const HTML_EXTS = new Set(["html", "htm"]);
export const MD_EXTS = new Set(["md", "markdown", "mdx"]);
const CODE_TEXT_EXTS = new Set([
  "py", "pyi", "pyw", "js", "mjs", "cjs", "jsx", "ts", "tsx", "mts", "cts",
  "json", "jsonc", "jsonl", "ndjson", "ipynb", "yaml", "yml", "toml",
  "sh", "bash", "zsh", "fish", "css", "scss", "sass", "less", "xml", "vue",
  "rs", "go", "java", "kt", "kts", "swift", "c", "h", "cpp", "hpp", "cc", "cxx", "hh", "hxx",
  "cs", "rb", "php", "sql", "ini", "conf", "cfg", "properties", "env", "gitignore",
  "diff", "patch", "tex", "latex", "sty", "cls", "bib", "mk", "makefile", "dockerfile",
  "csv", "tsv", "log", "rst", "adoc", "lua", "r", "jl", "dart", "scala",
  "clj", "cljs", "edn", "erl", "hrl", "ex", "exs", "hs", "fs", "fsi", "fsx",
  "pl", "pm", "v", "sv", "svh", "graphql", "gql", "proto", "tf", "tfvars",
  "hcl", "nix", "svelte", "cmake", "gradle", "groovy", "bat", "cmd", "ps1",
]);

/**
 * Known binary container suffixes without a text representation (mirrors dsh's
 * `UNVIEWABLE_BINARY_EXTENSIONS`). Any suffix not listed here falls back to a
 * NUL-checked plain-text preview so `.tex` and unknown extensions open instead
 * of dead-ending on "unsupported".
 */
export const UNVIEWABLE_BINARY_EXTS = new Set([
  // video
  "mp4", "mov", "avi", "mkv", "webm", "flv", "wmv", "m4v",
  // audio
  "mp3", "wav", "flac", "ogg", "m4a", "aac", "wma", "opus",
  // archives
  "zip", "gz", "tgz", "bz2", "xz", "zst", "7z", "rar", "tar", "jar", "whl",
  // office documents
  "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp", "pages", "numbers",
  // executables and compiled objects
  "exe", "dll", "so", "dylib", "bin", "o", "a", "class", "pyc", "pyo", "pyd", "wasm",
  // fonts
  "ttf", "otf", "woff", "woff2", "eot",
  // disk images and databases
  "dmg", "iso", "img", "sqlite", "sqlite3", "db",
  // design documents, unsupported raster formats, and binary tensor/data blobs
  "psd", "ai", "sketch", "tiff", "tif", "heic", "heif", "avif",
  "npy", "npz", "pt", "pth", "ckpt", "safetensors", "onnx", "parquet", "feather", "arrow", "pkl", "pickle",
]);

export type PreviewTargetKind = "image" | "pdf" | "html" | "md" | "code" | "text" | "unsupported";

/**
 * Classify how the side-panel preview should open `name`:
 * - known code/markup suffixes (including `.tex`, `.bib`, `.sty`, `.cls`) -> `"code"` (Code view + Plain text option)
 * - `.txt` and unknown or dotless filenames -> `"text"` (Plain text default + Code option)
 * - known binary containers -> `"unsupported"`
 */
export function classifyPreviewTarget(name: string): PreviewTargetKind {
  const base = name.includes("/") ? name.slice(name.lastIndexOf("/") + 1) : name;
  const lower = base.toLowerCase();
  const dot = lower.lastIndexOf(".");
  const ext = dot > 0 ? lower.slice(dot + 1) : "";
  if (IMAGE_EXTS.has(ext)) return "image";
  if (ext === "pdf") return "pdf";
  if (HTML_EXTS.has(ext)) return "html";
  if (MD_EXTS.has(ext)) return "md";
  if (UNVIEWABLE_BINARY_EXTS.has(ext)) return "unsupported";
  if (
    lower === "dockerfile" || lower.startsWith("dockerfile.") ||
    lower === "makefile" || lower === "gnumakefile" || lower === "cmakelists.txt" ||
    (dot === 0 && CODE_TEXT_EXTS.has(lower.slice(1))) ||
    CODE_TEXT_EXTS.has(ext)
  ) {
    return "code";
  }
  return "text";
}

const URL_SCHEME_RE = /^[a-z][a-z0-9+.-]*:\/\//i;
const MAX_PATH_LEN = 512;
const FILE_ARG_KEYS = ["file_path", "path", "target_file", "file"];

/** True when `path` is an absolute path (`/a/b`) or a home-relative path (`~` / `~/a/b`). */
export function isAbsOrHomePath(path: string): boolean {
  const s = path.trim();
  return s.startsWith("/") || s === "~" || s.startsWith("~/");
}

/**
 * Resolve `filePath` into `{ targetRoot, targetRel, absPath, isExternal }` for
 * `/api/workspace/content` and `/api/open`. Both `/abs/path` and `~/.agentica/...`
 * resolve against their own parent directory instead of the project `root`.
 */
export function splitExternalTarget(filePath: string, workspaceRoot: string): {
  targetRoot: string;
  targetRel: string;
  absPath: string;
  isExternal: boolean;
} {
  const trimmed = filePath.trim();
  const isExternal = isAbsOrHomePath(trimmed);
  if (isExternal) {
    const slash = trimmed.lastIndexOf("/");
    const parent = slash > 0 ? trimmed.slice(0, slash) : (trimmed.startsWith("/") ? "/" : "~");
    const name = slash >= 0 ? trimmed.slice(slash + 1) : trimmed;
    return {
      targetRoot: parent,
      targetRel: name,
      absPath: trimmed,
      isExternal: true,
    };
  }
  const rootClean = workspaceRoot.replace(/\/+$/, "");
  return {
    targetRoot: workspaceRoot,
    targetRel: trimmed,
    absPath: rootClean ? `${rootClean}/${trimmed}` : trimmed,
    isExternal: false,
  };
}

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
  const add = (raw: string) => {
    const text = raw.trim().replace(/[),.;:!?，。；：！？]+$/, "");
    if (!isFilePathLike(text) || seen.has(text)) return;
    seen.add(text);
    out.push(text);
  };
  for (const m of markdown.matchAll(/`([^`\n]+)`/g)) {
    add(m[1]!);
  }
  for (const m of markdown.matchAll(/(?:^|[\s(（"'“‘])((?:~\/|\/Users\/|\/home\/|\/tmp\/|\/var\/|\/etc\/|\/opt\/)[^\s`"'”’<>()（）]+)/gm)) {
    add(m[1]!);
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

/**
 * Like `toWorkspaceRelative`, but keeps external `~/...` and `/...` file paths
 * intact (instead of returning `null`) when they have no `..` traversal, so
 * built-in files under `~/.agentica/...` appear in `MessageFilesCard` and open
 * in the side panel.
 */
export function normalizeMentionedPath(path: string, workspace: string | null): string | null {
  const rel = toWorkspaceRelative(path, workspace);
  if (rel !== null) return rel;
  const s = path.trim();
  if (s.length === 0 || s.length > MAX_PATH_LEN || !isAbsOrHomePath(s)) return null;
  const isHome = s === "~" || s.startsWith("~/");
  const body = isHome ? s.slice(1) : s;
  const stack: string[] = [];
  for (const seg of body.split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") return null;
    stack.push(seg);
  }
  if (stack.length === 0) return null;
  return isHome ? `~/${stack.join("/")}` : `/${stack.join("/")}`;
}
