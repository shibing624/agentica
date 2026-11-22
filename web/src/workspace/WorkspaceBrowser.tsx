/**
 * Workspace file browser: scoped directory tree and file previews.
 * Text over 256 KiB is truncated; source highlighting stops at 64 KiB.
 * Markdown has rendered/source views. Source views (and fenced blocks inside
 * rendered Markdown) use CodeFrame.
 */
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import Markdown, { type ExtraProps } from "react-markdown";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import rehypeKatex from "rehype-katex";
import * as api from "../api";
import { CodeFrame, MarkdownCode } from "../components/CodeFrame";
import { useStrings } from "../i18n";
import { fmtFileSize } from "../lib/format";
import { languageFromFilename, HIGHLIGHT_LIMIT } from "../lib/highlight";
import { classifyPreviewTarget, joinWorkspacePath, splitExternalTarget, toWorkspaceRelative } from "../lib/filePath";
import { GFM_PLUGINS, localMarkdownPath, markdownUrlTransform } from "../lib/markdownPlugins";
import { expandWorkspaceAncestors, toggleWorkspaceDirectory } from "../lib/panelTabs";
import { MarkdownLink } from "../components/ChatMarkdown";
import { FileContextMenu } from "../components/FileContextMenu";
import { askConfirm, showToast } from "../store";
import { FileTypeIcon, IconChevronDown, IconChevronRight, IconClose, IconCopy, IconFinder, IconFolder, IconRefresh, IconSearch, IconTerminal, IconUpload } from "../icons";
import { PlainText, PreviewActions, type ViewMode } from "./PreviewActions";
import { SheetView } from "./SheetView";
import { PreviewableImage } from "../components/ImageLightbox";

const EXTERNAL_REF_RE = /^[a-z][a-z0-9+.-]*:/i;

type Entry = { name: string; kind: "dir" | "file"; sizeBytes: number; mtime?: string };
type Preview = {
  path: string;
  name: string;
  kind: "code" | "text" | "md" | "table" | "workbook" | "image" | "html" | "pdf" | "unsupported";
  content?: string;
  truncated?: boolean;
};

function dirOf(filePath: string): string {
  return filePath.includes("/") ? filePath.slice(0, filePath.lastIndexOf("/")) : "";
}

function resolveRelative(baseDir: string, ref: string): string {
  const trimmed = ref.trim();
  if (trimmed === "~" || trimmed.startsWith("~/")) return trimmed;
  const out = ref.startsWith("/") || baseDir === "" ? [] : baseDir.split("/");
  for (const seg of ref.split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") out.pop();
    else out.push(seg);
  }
  return (ref.startsWith("/") ? "/" : "") + out.join("/");
}

/**
 * One file in its own panel tab: loads on mount and on refresh, keeps its view
 * mode. A relative link in rendered Markdown opens the target in another tab.
 */
/** A workspace Markdown file. Memoized on its inputs: `img` / `a` are component
 *  types, and fresh ones on every app-state tick (each keystroke in the
 *  composer) remounted every image and link, which reloaded and jumped. */
const MarkdownDoc = memo(function MarkdownDoc({
  root, scope, path: docPath, content, isExternal, onOpenFile,
}: {
  root: string;
  scope?: api.WorkspaceScope;
  path: string;
  content: string;
  isExternal: boolean;
  onOpenFile: (path: string) => void;
}) {
  const components = useMemo(() => ({
    img: ({ src, alt }: { src?: string; alt?: string }) => {
      if (!src) return <img alt={alt || ""} />;
      return (
        <img
          src={EXTERNAL_REF_RE.test(src)
            ? src
            : api.workspaceContentUrl(scope ?? root, resolveRelative(dirOf(docPath), src))}
          alt={alt || ""}
        />
      );
    },
    a: ({ href, title, children, node }: { href?: string; title?: string; children?: React.ReactNode } & ExtraProps) => {
      const localPath = localMarkdownPath(node);
      if (localPath !== undefined) {
        return <MarkdownLink localPath={localPath} title={title} onOpenFile={onOpenFile}>{children}</MarkdownLink>;
      }
      if (!href) return <>{children}</>;
      if (href.startsWith("#") || EXTERNAL_REF_RE.test(href)) {
        return <a href={href} title={title} target={EXTERNAL_REF_RE.test(href) ? "_blank" : undefined} rel="noreferrer">{children}</a>;
      }
      const target = resolveRelative(dirOf(docPath), href);
      const openPath = isExternal ? splitExternalTarget(target, root).absPath : target;
      return <a href={api.workspaceContentUrl(scope ?? root, target)} title={title} onClick={(e) => { e.preventDefault(); onOpenFile(openPath); }}>{children}</a>;
    },
    table: ({ children }: { children?: React.ReactNode }) => <div className="md-table"><table>{children}</table></div>,
    pre: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    code: MarkdownCode,
  }), [root, scope, docPath, isExternal, onOpenFile]);
  return (
    <div className="md-doc">
      <Markdown remarkPlugins={MD_REMARK} rehypePlugins={MD_REHYPE} components={components} urlTransform={markdownUrlTransform}>{content}</Markdown>
    </div>
  );
});

const MD_REMARK = [...GFM_PLUGINS, remarkMath];
const MD_REHYPE = [rehypeRaw, rehypeKatex];

export function FilePreview({
  root, scope, path: filePath, onOpenFile, allowNativeOpen = true, showFileMenu = false,
}: {
  root: string;
  scope?: api.WorkspaceScope;
  path: string;
  onOpenFile: (path: string) => void;
  allowNativeOpen?: boolean;
  showFileMenu?: boolean;
}) {
  const S = useStrings();
  const [preview, setPreview] = useState<Preview | null>(null);
  const [mode, setMode] = useState<ViewMode>("markdown");
  // dsh `TextTabState.wrap`: per-tab line-wrap preference, enabled by default.
  const [wrap, setWrap] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  // Workbooks fetch their own bytes; a refresh remounts the table to refetch.
  const [refreshKey, setRefreshKey] = useState(0);
  const seq = useRef(0);
  const relative = scope && toWorkspaceRelative(filePath, scope.root.replaceAll("\\", "/"));
  const { targetRoot, targetRel, absPath: resolvedAbsPath, isExternal } = splitExternalTarget(relative || filePath, root);
  const target = !isExternal && scope ? scope : targetRoot;

  /** `keepMode` is a refresh: same file, same view. */
  const load = useCallback(async (keepMode: boolean) => {
    const name = filePath.includes("/") ? filePath.slice(filePath.lastIndexOf("/") + 1) : filePath;
    const targetKind = classifyPreviewTarget(name);
    const nonce = ++seq.current;
    const present = (p: Preview) => { if (nonce === seq.current) setPreview(p); };
    if (targetKind === "image") { present({ path: filePath, name, kind: "image" }); return; }
    if (targetKind === "pdf") { present({ path: filePath, name, kind: "pdf" }); return; }
    if (targetKind === "workbook") { present({ path: filePath, name, kind: "workbook" }); return; }
    if (targetKind === "unsupported") { present({ path: filePath, name, kind: "unsupported" }); return; }
    const { ok, data } = await api.fetchWorkspacePreview(target, targetRel);
    if (nonce !== seq.current) return;
    if (!ok || !data) { present({ path: filePath, name, kind: "unsupported" }); return; }
    const content = data.content || "";
    if (!keepMode) {
      setMode(
        targetKind === "md"
          ? (content.length > HIGHLIGHT_LIMIT ? "code" : "markdown")
          : targetKind === "html"
            ? "web"
            : targetKind === "table"
              ? "table"
            : targetKind === "code"
              ? "code"
              : "text",
      );
    }
    present({
      path: filePath,
      name,
      kind: targetKind,
      content,
      truncated: !!data.truncated,
    });
  }, [filePath, target, targetRel]);

  useEffect(() => { void load(false); }, [load]);

  if (!preview) return <div className="ws-browser"><p className="ws-muted">{S.common.loading}</p></div>;

  const src = api.workspaceContentUrl(target, targetRel);
  const dl = api.workspaceContentUrl(target, targetRel, true);
  const absPath = isExternal ? preview.path : resolvedAbsPath;
  const hasText = preview.content !== undefined;
  const modes: ViewMode[] = preview.kind === "md" ? ["markdown", "code", "text"]
    : preview.kind === "html" ? ["web", "code", "text"]
    : preview.kind === "table" ? ["table", "code", "text"]
    : preview.kind === "code" ? ["code", "text"]
    : preview.kind === "text" ? ["text", "code"]
    : [];
  const view: ViewMode | null = modes.includes(mode) ? mode : (modes[0] ?? null);
  const supportsWrap = view === "code" || view === "text";
  const refresh = () => {
    setRefreshing(true);
    setRefreshKey((k) => k + 1);
    void load(true).finally(() => setRefreshing(false));
  };
  const slashIdx = preview.path.lastIndexOf("/");
  const pathDir = slashIdx >= 0 ? preview.path.slice(0, slashIdx + 1) : "";
  const pathFile = slashIdx >= 0 ? preview.path.slice(slashIdx + 1) : preview.path;
  const fileLabel = <span className="ws-preview-name" title={preview.path} tabIndex={showFileMenu ? 0 : undefined}>
    {pathDir && <span className="ws-preview-dir">{pathDir}</span>}
    <span className="ws-preview-base">{pathFile}</span>
  </span>;
  return (
    <div className="ws-browser">
      <div className="ws-preview-bar">
        <FileTypeIcon path={preview.name} size={15} className="ws-preview-type-icon" />
        {showFileMenu && (typeof target !== "string" || allowNativeOpen) ? (
          <FileContextMenu path={filePath} source={{ target, path: targetRel }} canOpenHost={allowNativeOpen}
                           onOpenSidebar={() => onOpenFile(filePath)}>{fileLabel}</FileContextMenu>
        ) : fileLabel}
        <PreviewActions
          modes={modes}
          mode={view ?? "code"}
          onMode={setMode}
          wrap={supportsWrap ? wrap : undefined}
          onToggleWrap={supportsWrap ? () => setWrap((v) => !v) : undefined}
          onRefresh={refresh}
          refreshing={refreshing}
          absPath={allowNativeOpen ? absPath : undefined}
          downloadHref={dl}
          downloadName={preview.name}
        />
      </div>
      <div className={"ws-preview-body" + (view === "code" || view === "text" ? " source" : "")
        + (view === "table" || preview.kind === "workbook" ? " sheet" : "")
        + (view === "web" || preview.kind === "pdf" ? " frame" : "")}>
        {preview.kind === "image" && <PreviewableImage className="ws-preview-img" src={src} alt={preview.name} />}
        {preview.kind === "pdf" && <iframe src={src} title={preview.name} />}
        {preview.kind === "workbook" && <SheetView key={refreshKey} name={preview.name} src={src} />}
        {view === "table" && hasText && (
          <SheetView name={preview.name} text={preview.content} truncated={preview.truncated} />
        )}
        {view === "web" && hasText && (
          <iframe srcDoc={preview.content} title={preview.name} sandbox="allow-scripts" />
        )}
        {view === "markdown" && (
          <MarkdownDoc root={targetRoot} scope={!isExternal ? scope : undefined} path={targetRel} content={preview.content || ""} isExternal={isExternal} onOpenFile={onOpenFile} />
        )}
        {view === "code" && (
          <CodeFrame
            code={preview.content || ""}
            language={languageFromFilename(preview.name)}
            highlight={(preview.content || "").length <= HIGHLIGHT_LIMIT}
            lineNumbers
            wrap={wrap}
            pane
          />
        )}
        {view === "text" && <PlainText text={preview.content || ""} wrap={wrap} />}
        {preview.truncated && view && <p className="ws-trunc">{S.files.previewTruncated}</p>}
        {preview.kind === "unsupported" && <p className="ws-muted">{S.files.previewUnsupported}</p>}
      </div>
    </div>
  );
}

type DirectoryProps = {
  scope: api.WorkspaceScope;
  isAdmin: boolean;
  path: string;
  active: boolean;
  expanded: string[];
  selected: string;
  currentFile: string | null;
  onCurrentRow: (row: HTMLButtonElement | null) => void;
  revision: string;
  toggle: (path: string) => void;
  onOpenFile: (path: string) => void;
};

function DirectoryChildren(props: DirectoryProps) {
  const { scope, isAdmin, path, active, expanded, selected, currentFile, onCurrentRow, revision, toggle, onOpenFile } = props;
  const S = useStrings();
  const [entries, setEntries] = useState<Entry[] | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!active) return;
    const request = new AbortController();
    void api.fetchWorkspaceFiles(scope, path, request.signal).then(({ ok, data }) => {
      if (request.signal.aborted) return;
      setFailed(!ok);
      if (ok) setEntries(data.entries);
    });
    return () => request.abort();
  }, [scope, path, active, revision]);
  if (failed) return <p className="ws-err">{S.files.loadFailed}</p>;
  if (!entries) return <p className="ws-muted">{S.common.loading}</p>;
  if (!entries.length) return <p className="ws-muted">{S.files.empty}</p>;
  return (
    <ul className="ws-tree-level">
      {entries.map((entry) => {
        const child = joinWorkspacePath(path, entry.name);
        const directory = entry.kind === "dir";
        const open = directory && expanded.includes(child);
        const current = !directory && child === currentFile;
        return (
          <li key={child}>
            <div className={"ws-tree-row" + (current || directory && selected === child ? " selected" : "")}>
              <FileContextMenu path={child} source={{ target: scope, path: child }} canOpenHost={isAdmin}
                               directory={directory} onOpenSidebar={() => onOpenFile(child)}>
                <button type="button" className="ws-row" ref={active && current ? onCurrentRow : undefined}
                        title={child} aria-expanded={directory ? open : undefined} aria-current={current ? true : undefined}
                        onClick={() => directory ? toggle(child) : onOpenFile(child)}>
                  <span className="ws-tree-caret" aria-hidden="true">
                    {directory && (open ? <IconChevronDown /> : <IconChevronRight />)}
                  </span>
                  <span className="ws-row-icon"><FileTypeIcon path={entry.name} kind={directory ? "folder" : undefined} size={16} /></span>
                  <span className="ws-row-name">{entry.name}</span>
                  {!directory && <span className="ws-row-meta">{fmtFileSize(entry.sizeBytes)}</span>}
                </button>
              </FileContextMenu>
              {!directory && <a className="ws-row-dl" href={api.workspaceContentUrl(scope, child, true)} download={entry.name}
                                title={S.files.download} aria-label={`${S.files.download}: ${entry.name}`}>↓</a>}
            </div>
            {open && <DirectoryChildren {...props} path={child} />}
          </li>
        );
      })}
    </ul>
  );
}

/** One tree for a scoped session, shared by the Files page and its previews. */
export function WorkspaceBrowser({
  scope, accountId, isAdmin, active, revision, currentFile, projectDir, onOpenFile,
}: {
  scope: api.WorkspaceScope;
  accountId: string;
  isAdmin: boolean;
  active: boolean;
  revision: string;
  currentFile: string | null;
  projectDir: string;
  onOpenFile: (path: string) => void;
}) {
  const S = useStrings();
  const { root } = scope;
  const storageKey = "ag_workspace_tree:" + JSON.stringify([accountId, scope.session_id, scope.binding]);
  const [expanded, setExpanded] = useState<string[]>(() => {
    try {
      const value = JSON.parse(localStorage.getItem(storageKey) ?? "[]");
      return Array.isArray(value) ? value.filter((path): path is string => typeof path === "string").slice(0, 128) : [];
    } catch { return []; }
  });
  const [path, setPath] = useState(() => {
    try { return localStorage.getItem(`${storageKey}:selection`) || ""; } catch { return ""; }
  });
  const [uploading, setUploading] = useState(false);
  const [reloadTick, setReloadTick] = useState(0);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ paths: string[]; truncated: boolean; failed?: boolean } | null>(null);
  const [reveal, setReveal] = useState<string | null>(null);
  const currentRow = useRef<HTMLButtonElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(expanded)); } catch { /* in-memory tree still works */ }
  }, [storageKey, expanded]);
  useEffect(() => {
    try { localStorage.setItem(`${storageKey}:selection`, path); } catch { /* in-memory selection still works */ }
  }, [storageKey, path]);
  useEffect(() => {
    const list = scroll.current;
    const tree = content.current;
    if (!active || !list || !tree || query.trim() || currentFile) return;
    let offset = 0;
    try { offset = Number(localStorage.getItem(`${storageKey}:scroll`)) || 0; } catch { return; }
    if (!offset) return;
    const observer = new ResizeObserver(() => {
      if (list.scrollHeight - list.clientHeight < offset) return;
      list.scrollTop = offset;
      observer.disconnect();
    });
    observer.observe(tree);
    const release = () => observer.disconnect();
    list.addEventListener("wheel", release, { once: true, passive: true });
    list.addEventListener("pointerdown", release, { once: true });
    return () => {
      release();
      list.removeEventListener("wheel", release);
      list.removeEventListener("pointerdown", release);
    };
  }, [storageKey, active, query, currentFile]);
  const locateFile = useCallback((file: string) => {
    setExpanded((previous) => expandWorkspaceAncestors(previous, file));
    setReveal(file);
  }, []);
  useLayoutEffect(() => {
    if (!active || !currentFile) return;
    setQuery("");
    setResults(null);
    locateFile(currentFile);
  }, [active, currentFile, locateFile]);
  // Lazy ancestor requests may finish after the tab change. Scroll only when
  // the selected row actually mounts, and only once for this reveal.
  const onCurrentRow = useCallback((row: HTMLButtonElement | null) => {
    currentRow.current = row;
    if (!row || reveal !== currentFile || !reveal) return;
    row.scrollIntoView({ block: "nearest", inline: "nearest" });
    setReveal(null);
  }, [reveal, currentFile]);
  useEffect(() => {
    const list = scroll.current;
    if (!list || !active || !currentFile || query.trim()) return;
    let width = list.clientWidth;
    let height = list.clientHeight;
    const observer = new ResizeObserver(() => {
      const nextWidth = list.clientWidth;
      const nextHeight = list.clientHeight;
      if (nextWidth === width && nextHeight === height) return;
      width = nextWidth;
      height = nextHeight;
      currentRow.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
    observer.observe(list);
    return () => observer.disconnect();
  }, [active, currentFile, query]);
  const clearSearch = () => {
    setQuery("");
    setResults(null);
    if (currentFile) locateFile(currentFile);
  };
  useEffect(() => {
    if (!active || !query.trim()) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void api.searchWorkspaceFiles(scope, query.trim(), controller.signal).then(({ ok, data }) => {
        if (!controller.signal.aborted) setResults(ok && data ? data : { paths: [], truncated: false, failed: true });
      });
    }, 180);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [active, scope, query, revision, reloadTick]);

  const doUpload = (files: File[], destination: string) => {
    setUploading(true);
    void (async () => {
      try {
        for (const file of files) {
          const up = await api.uploadWorkspaceFile(file, scope, destination);
          if (!up.ok) showToast(S.chat.uploadFailed(file.name, up.data?.detail), 5000);
        }
        showToast(S.files.uploaded);
        setReloadTick((t) => t + 1);
      } finally {
        setUploading(false);
      }
    })();
  };

  const onUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? [...e.target.files] : [];
    e.target.value = "";
    if (!files.length) return;
    const target = path;
    const { ok, data } = await api.fetchWorkspaceFiles(scope, target);
    if (!ok) { showToast(S.files.loadFailed); return; }
    const existing = new Set((data.entries as Entry[]).map((x) => x.name));
    const clashes = files.filter((f) => existing.has(f.name)).map((f) => f.name);
    if (clashes.length) {
      askConfirm({
        title: S.files.overwriteTitle,
        msg: S.files.overwriteConfirm(clashes.length) + "\n" + clashes.join("\n"),
        okLabel: S.files.upload,
        onOk: () => doUpload(files, target),
      });
    } else doUpload(files, target);
  };
  const cleanRoot = root.replace(/[\\/]+$/, "");
  const label = cleanRoot.split(/[\\/]/).at(-1) || root;
  const parent = cleanRoot.slice(0, -(label.length + 1));

  return (
    <div className="ws-browser">
      <div className="ws-toolbar">
        <button type="button" className="ws-crumb ws-crumb-root ws-scope-title" title={root} onClick={() => setPath("")}>
          <FileTypeIcon kind="folder" size={14} />
          <span><strong>{label}{projectDir && projectDir !== root ? ` · ${S.files.boundWorktree}` : ""}</strong>
            <small>{parent && (parent + (path ? ` · ${path}` : ""))}</small></span>
        </button>
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
        {isAdmin && <OpenFolderButton dir={path ? cleanRoot + "/" + path : root} />}
      </div>
      <div className="ws-search-field">
        <span className="ws-search-icon" aria-hidden="true"><IconSearch /></span>
        <input ref={searchInput} type="search" className="ws-search" placeholder={S.files.searchNames} aria-label={S.files.searchNames}
               value={query} maxLength={160} autoComplete="off" spellCheck={false}
               onChange={(event) => { setQuery(event.currentTarget.value); setResults(null); }}
               onKeyDown={(event) => {
                 if (event.key !== "Escape" || !query || event.nativeEvent.isComposing) return;
                 event.preventDefault();
                 event.stopPropagation();
                 clearSearch();
               }} />
        {query && <button type="button" className="ws-tool ws-search-clear" aria-label={S.files.clearSearch} title={S.files.clearSearch}
                          onClick={() => { clearSearch(); searchInput.current?.focus(); }}><IconClose /></button>}
      </div>
      <div className="ws-list" ref={scroll} onScroll={(e) => {
        if (!query.trim()) try { localStorage.setItem(`${storageKey}:scroll`, String(e.currentTarget.scrollTop)); } catch { /* session remains usable */ }
      }}>
        <div ref={content} hidden={!!query.trim()}>
          <DirectoryChildren scope={scope} isAdmin={isAdmin} path="" active={active && !query.trim()} expanded={expanded} selected={path}
                             currentFile={currentFile} onCurrentRow={onCurrentRow}
                             revision={`${revision}:${reloadTick}`} onOpenFile={onOpenFile} toggle={(next) => {
              const opens = !expanded.includes(next);
              setExpanded((value) => toggleWorkspaceDirectory(value, next).slice(0, 128));
              setPath(opens ? next : "");
            }} />
        </div>
        {query.trim() && (
          <div className="ws-search-results">
            {!results ? <p className="ws-muted">{S.common.loading}</p> : results.failed ? <p className="ws-err">{S.files.loadFailed}</p> : (
              <>
                {!results.paths.length && <p className="ws-muted">{S.files.empty}</p>}
                {results.truncated && <p className="ws-muted">{S.files.searchLimited}</p>}
                {results.paths.map((match) => {
                  const open = () => { clearSearch(); onOpenFile(match); };
                  return <FileContextMenu key={match} path={match} source={{ target: scope, path: match }}
                                          canOpenHost={isAdmin} onOpenSidebar={open}>
                    <button type="button" className="ws-row" title={match} onClick={open}>
                      <span className="ws-tree-caret" aria-hidden="true" /><FileTypeIcon path={match} size={16} />
                      <span className="ws-row-name">{match}</span>
                    </button>
                  </FileContextMenu>;
                })}
              </>
            )}
          </div>
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
