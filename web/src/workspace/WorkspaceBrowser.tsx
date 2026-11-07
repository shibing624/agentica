/**
 * Workspace file browser: directory list <-> file preview (penguin's drill-down).
 * Text over 256 KiB is truncated; source highlighting stops at 64 KiB.
 * Markdown has rendered/source views. Source views (and fenced blocks inside
 * rendered Markdown) use CodeFrame.
 */
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import Markdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import rehypeKatex from "rehype-katex";
import * as api from "../api";
import { CodeFrame, MarkdownCode } from "../components/CodeFrame";
import { useStrings } from "../i18n";
import { fmtFileSize, formatDateTime } from "../lib/format";
import { languageFromFilename, HIGHLIGHT_LIMIT } from "../lib/highlight";
import { joinWorkspacePath } from "../lib/file-path";
import { GFM_PLUGINS } from "../lib/markdownPlugins";
import { askConfirm, showToast } from "../store";
import { IconChevronDown, IconCopy, IconFinder, IconFolder, IconRefresh, IconTerminal, IconUpload } from "../icons";
import { PlainText, PreviewActions, type ViewMode } from "./PreviewActions";

const TEXT_EXTS = new Set([
  "txt", "md", "json", "js", "mjs", "cjs", "ts", "tsx", "jsx", "py", "sh", "bash",
  "yaml", "yml", "toml", "css", "html", "htm", "csv", "log", "xml", "ini", "conf",
  "rs", "go", "java", "c", "h", "cpp", "hpp", "sql", "rb", "php", "gitignore", "env",
]);
const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "webp", "svg"]);
const HTML_EXTS = new Set(["html", "htm"]);
const EXTERNAL_REF_RE = /^[a-z][a-z0-9+.-]*:/i;

type Entry = { name: string; kind: "dir" | "file"; sizeBytes: number; mtime?: string };
type Preview = {
  path: string;
  name: string;
  kind: "text" | "md" | "image" | "html" | "pdf" | "unsupported";
  content?: string;
  truncated?: boolean;
};

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : name.toLowerCase();
}

function dirOf(filePath: string): string {
  return filePath.includes("/") ? filePath.slice(0, filePath.lastIndexOf("/")) : "";
}

function resolveRelative(baseDir: string, ref: string): string {
  const out = ref.startsWith("/") || baseDir === "" ? [] : baseDir.split("/");
  for (const seg of ref.split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") out.pop();
    else out.push(seg);
  }
  return out.join("/");
}

/**
 * One file in its own panel tab: loads on mount and on refresh, keeps its view
 * mode. A relative link in rendered Markdown opens the target in another tab.
 */
/** A workspace Markdown file. Memoized on its inputs: `img` / `a` are component
 *  types, and fresh ones on every app-state tick (each keystroke in the
 *  composer) remounted every image and link, which reloaded and jumped. */
const MarkdownDoc = memo(function MarkdownDoc({
  root, path: docPath, content, onOpenFile,
}: {
  root: string;
  path: string;
  content: string;
  onOpenFile: (path: string) => void;
}) {
  const components = useMemo(() => ({
    img: ({ src: href, alt }: { src?: unknown; alt?: string }) => (
      <img
        src={typeof href === "string" && !EXTERNAL_REF_RE.test(href)
          ? api.workspaceContentUrl(root, resolveRelative(dirOf(docPath), href))
          : (href as string | undefined)}
        alt={alt || ""}
      />
    ),
    a: ({ href, children }: { href?: string; children?: React.ReactNode }) => {
      if (typeof href !== "string" || href.startsWith("#") || EXTERNAL_REF_RE.test(href)) {
        return <a href={href} target={EXTERNAL_REF_RE.test(href || "") ? "_blank" : undefined} rel="noreferrer">{children}</a>;
      }
      const target = resolveRelative(dirOf(docPath), href);
      return <a href={api.workspaceContentUrl(root, target)} onClick={(e) => { e.preventDefault(); onOpenFile(target); }}>{children}</a>;
    },
    table: ({ children }: { children?: React.ReactNode }) => <div className="md-table"><table>{children}</table></div>,
    pre: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    code: MarkdownCode,
  }), [root, docPath, onOpenFile]);
  return (
    <div className="md-doc">
      <Markdown remarkPlugins={MD_REMARK} rehypePlugins={MD_REHYPE} components={components}>{content}</Markdown>
    </div>
  );
});

const MD_REMARK = [...GFM_PLUGINS, remarkMath];
const MD_REHYPE = [rehypeRaw, rehypeKatex];

export function FilePreview({
  root, path: filePath, onOpenFile,
}: {
  root: string;
  path: string;
  onOpenFile: (path: string) => void;
}) {
  const S = useStrings();
  const [preview, setPreview] = useState<Preview | null>(null);
  const [mode, setMode] = useState<ViewMode>("markdown");
  const [refreshing, setRefreshing] = useState(false);
  const seq = useRef(0);

  /** `keepMode` is a refresh: same file, same view. */
  const load = useCallback(async (keepMode: boolean) => {
    const name = filePath.includes("/") ? filePath.slice(filePath.lastIndexOf("/") + 1) : filePath;
    const ext = extOf(name);
    const nonce = ++seq.current;
    const present = (p: Preview) => { if (nonce === seq.current) setPreview(p); };
    if (IMAGE_EXTS.has(ext)) { present({ path: filePath, name, kind: "image" }); return; }
    if (ext === "pdf") { present({ path: filePath, name, kind: "pdf" }); return; }
    const isHtml = HTML_EXTS.has(ext);
    const isMd = ext === "md";
    if (!isHtml && !TEXT_EXTS.has(ext)) {
      present({ path: filePath, name, kind: "unsupported" });
      return;
    }
    const { ok, data } = await api.fetchWorkspacePreview(root, filePath);
    if (nonce !== seq.current) return;
    if (!ok || !data) { present({ path: filePath, name, kind: "unsupported" }); return; }
    const content = data.content || "";
    if (!keepMode) {
      setMode(isMd ? (content.length > HIGHLIGHT_LIMIT ? "code" : "markdown") : isHtml ? "web" : "code");
    }
    present({
      path: filePath,
      name,
      kind: isHtml ? "html" : isMd ? "md" : "text",
      content,
      truncated: !!data.truncated,
    });
  }, [root, filePath]);

  useEffect(() => { void load(false); }, [load]);

  if (!preview) return <div className="ws-browser"><p className="ws-muted">{S.common.loading}</p></div>;

  const src = api.workspaceContentUrl(root, preview.path);
  const dl = api.workspaceContentUrl(root, preview.path, true);
  const hasText = preview.content !== undefined;
  const modes: ViewMode[] = preview.kind === "md" ? ["markdown", "code", "text"]
    : preview.kind === "html" ? ["web", "code", "text"]
    : preview.kind === "text" ? ["code", "text"]
    : [];
  const view: ViewMode | null = modes.includes(mode) ? mode : (modes[0] ?? null);
  const refresh = () => {
    setRefreshing(true);
    void load(true).finally(() => setRefreshing(false));
  };
  return (
    <div className="ws-browser">
      <div className="ws-preview-bar">
        <span className="ws-preview-name" title={preview.path}>{preview.path}</span>
        <PreviewActions
          modes={modes}
          mode={view ?? "code"}
          onMode={setMode}
          onRefresh={refresh}
          refreshing={refreshing}
          absPath={root.replace(/\/+$/, "") + "/" + preview.path}
          downloadHref={dl}
          downloadName={preview.name}
        />
      </div>
      <div className={"ws-preview-body" + (view === "code" ? " source" : "")}>
        {preview.kind === "image" && <img src={src} alt={preview.name} />}
        {preview.kind === "pdf" && <iframe src={src} title={preview.name} />}
        {view === "web" && hasText && (
          <iframe srcDoc={preview.content} title={preview.name} sandbox="allow-scripts" />
        )}
        {view === "markdown" && (
          <MarkdownDoc root={root} path={preview.path} content={preview.content || ""} onOpenFile={onOpenFile} />
        )}
        {view === "code" && (
          <CodeFrame
            code={preview.content || ""}
            language={languageFromFilename(preview.name)}
            highlight={(preview.content || "").length <= HIGHLIGHT_LIMIT}
          />
        )}
        {view === "text" && <PlainText text={preview.content || ""} />}
        {preview.truncated && view && <p className="ws-trunc">{S.files.previewTruncated}</p>}
        {preview.kind === "unsupported" && <p className="ws-muted">{S.files.previewUnsupported}</p>}
      </div>
    </div>
  );
}

/** The panel's fixed first tab: the directory list. A file opens in a tab of its own. */
export function WorkspaceBrowser({
  root, active, onOpenFile,
}: {
  root: string;
  active?: boolean;
  onOpenFile: (path: string) => void;
}) {
  const S = useStrings();
  const [path, setPath] = useState("");
  const [data, setData] = useState<{ base: string; entries: Entry[] } | null>(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [reloadTick, setReloadTick] = useState(0);

  const [renderedRoot, setRenderedRoot] = useState(root);
  if (renderedRoot !== root) {
    setRenderedRoot(root);
    setPath("");
    setData(null);
  }
  useEffect(() => {
    if (!root) return;
    let cancelled = false;
    setError("");
    void api.fetchWorkspaceFiles(root, path).then(({ ok, data: body, status }) => {
      if (cancelled) return;
      if (!ok) { setError(S.files.loadFailed + (status ? ` (${status})` : "")); return; }
      setData({ base: path, entries: body?.entries || [] });
    });
    return () => { cancelled = true; };
  }, [root, path, reloadTick, S.files.loadFailed]);

  const prevActive = useRef(active);
  useEffect(() => {
    if (active && !prevActive.current) setReloadTick((t) => t + 1);
    prevActive.current = active;
  }, [active]);

  const doUpload = (files: File[]) => {
    setUploading(true);
    void (async () => {
      try {
        for (const file of files) {
          const up = await api.uploadWorkspaceFile(file, root, path);
          if (!up.ok) showToast(S.chat.uploadFailed(file.name), 3000);
        }
        showToast(S.files.uploaded);
        setReloadTick((t) => t + 1);
      } finally {
        setUploading(false);
      }
    })();
  };

  const onUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? [...e.target.files] : [];
    e.target.value = "";
    if (!files.length) return;
    const existing = new Set((data?.entries || []).map((x) => x.name));
    const clashes = files.filter((f) => existing.has(f.name)).map((f) => f.name);
    if (clashes.length) {
      askConfirm({
        title: S.files.overwriteTitle,
        msg: S.files.overwriteConfirm(clashes.length) + "\n" + clashes.join("\n"),
        okLabel: S.files.upload,
        onOk: () => doUpload(files),
      });
    } else doUpload(files);
  };

  const crumbs = path === "" ? [] : path.split("/");
  const busy = error === "" && data !== null && data.base !== path;

  return (
    <div className="ws-browser">
      <div className="ws-toolbar">
        <button type="button" className="ws-crumb" onClick={() => setPath("")}>{S.files.root}</button>
        {crumbs.map((seg, i) => (
          <span key={i} className="ws-crumb-wrap">
            <span className="ws-sep">/</span>
            <button type="button" className="ws-crumb" onClick={() => setPath(crumbs.slice(0, i + 1).join("/"))}>{seg}</button>
          </span>
        ))}
        <span className="ws-toolbar-grow" />
        <button type="button" className="ws-tool" title={S.files.refresh} aria-label={S.files.refresh}
                onClick={() => setReloadTick((t) => t + 1)}>
          <IconRefresh />
        </button>
        <label className={"ws-tool" + (uploading ? " busy" : "")} title={uploading ? S.files.uploading : S.files.upload}
               aria-label={S.files.upload}>
          <input type="file" multiple hidden disabled={uploading} onChange={onUpload} />
          <IconUpload />
        </label>
        <OpenFolderButton dir={path ? root.replace(/\/+$/, "") + "/" + path : root} />
      </div>
      <div className="ws-list">
        {error ? <p className="ws-err">{error}</p>
          : data === null ? <p className="ws-muted">{S.common.loading}</p>
          : data.entries.length === 0 ? <p className="ws-muted">{S.files.empty}</p>
          : (
            <ul className={busy ? "busy" : ""}>
              {data.entries.map((entry) => {
                const target = joinWorkspacePath(data.base, entry.name);
                return (
                  <li key={entry.name}>
                    <button type="button" className="ws-row" disabled={busy} title={entry.name}
                            onClick={() => { if (entry.kind === "dir") setPath(target); else onOpenFile(target); }}>
                      <span className="ws-row-icon">{entry.kind === "dir" ? <IconFolder /> : <span className="ws-file-dot" />}</span>
                      <span className="ws-row-name">{entry.name}</span>
                      <span className="ws-row-meta">{entry.kind === "file" ? fmtFileSize(entry.sizeBytes) : ""}</span>
                      <span className="ws-row-meta">{entry.mtime ? formatDateTime(entry.mtime) : ""}</span>
                    </button>
                    {entry.kind === "file" && (
                      <a className="ws-row-dl" href={api.workspaceContentUrl(root, target, true)} download={entry.name} title={S.files.download}>↓</a>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
      </div>
    </div>
  );
}

const APP_CHOICE_KEY = "ag_open_in_app";
let appsCache: Promise<api.OpenApp[]> | null = null;
function loadApps(): Promise<api.OpenApp[]> {
  appsCache ??= api.fetchOpenApps().then(({ ok, data }) => (ok && data ? data.apps : []));
  return appsCache;
}

/** The app's own icon from the gateway; a glyph by kind when it has none. */
function AppIcon({ app }: { app: api.OpenApp }) {
  const [failed, setFailed] = useState(false);
  if (failed) return app.kind === "terminal" ? <IconTerminal /> : app.kind === "files" ? <IconFinder /> : <IconFolder />;
  return <img className="ws-app-icon" src={api.openAppIconUrl(app.id)} alt="" draggable={false} onError={() => setFailed(true)} />;
}

/**
 * dsh's open-in-app split button for the folder on screen: the main half opens
 * it in the last app picked (Finder at first); the chevron lists every
 * installed file manager, editor and terminal, with "copy path" below them.
 */
function OpenFolderButton({ dir }: { dir: string }) {
  const S = useStrings();
  const [apps, setApps] = useState<api.OpenApp[]>([]);
  const [choice, setChoice] = useState(() => localStorage.getItem(APP_CHOICE_KEY) || "finder");
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { void loadApps().then(setApps); }, []);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, [open]);
  const preferred = apps.find((a) => a.id === choice) ?? apps[0];
  const launch = (app: api.OpenApp) => {
    setOpen(false);
    void api.openPathApi(dir, app.id).then(({ ok }) => {
      if (!ok) { showToast(S.chat.openFailed, 2500); return; }
      localStorage.setItem(APP_CHOICE_KEY, app.id);
      setChoice(app.id);
    });
  };
  const copy = () => {
    setOpen(false);
    void navigator.clipboard.writeText(dir).then(() => showToast(S.common.copied));
  };
  const icon = (a: api.OpenApp) => <AppIcon app={a} />;
  const groups = (["files", "editor", "terminal"] as const)
    .map((kind) => apps.filter((a) => a.kind === kind))
    .filter((g) => g.length);
  return (
    <div className="ws-open" ref={ref}>
      <button type="button" className="ws-open-main" disabled={!preferred} title={preferred ? S.files.openIn(preferred.name) : ""}
              aria-label={preferred ? S.files.openIn(preferred.name) : ""} onClick={() => preferred && launch(preferred)}>
        {preferred ? icon(preferred) : <IconFinder />}
      </button>
      <button type="button" className={"ws-open-more" + (open ? " on" : "")} aria-haspopup="menu" aria-expanded={open}
              title={S.files.openInMore} aria-label={S.files.openInMore} onClick={() => setOpen((v) => !v)}>
        <IconChevronDown />
      </button>
      {open && (
        <div className="ws-open-menu" role="menu">
          <p className="ws-open-path" title={dir}><bdi dir="ltr">{dir}</bdi></p>
          {groups.map((g, gi) => (
            <div key={gi} className="ws-open-group">
              {g.map((a) => (
                <button key={a.id} type="button" role="menuitem" className={"ws-open-item" + (a.id === preferred?.id ? " on" : "")}
                        onClick={() => launch(a)}>
                  {icon(a)}<span>{a.name}</span>
                </button>
              ))}
            </div>
          ))}
          <div className="ws-open-group">
            <button type="button" role="menuitem" className="ws-open-item" onClick={copy}>
              <IconCopy /><span>{S.files.copyFolderPath}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
