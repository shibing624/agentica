/**
 * Workspace file browser: directory list <-> file preview (penguin's drill-down).
 * Text over 12000 characters is truncated; Markdown has rendered/source views.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import rehypeKatex from "rehype-katex";
import * as api from "../api";
import { useStrings } from "../i18n";
import { fmtFileSize, formatDateTime } from "../lib/format";
import { joinWorkspacePath } from "../lib/file-path";
import { askConfirm, showToast } from "../store";
import { IconCopy, IconFinder, IconFolder, IconTerminal } from "../icons";

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

let previewSeq = 0;

export function WorkspaceBrowser({
  root, openRequest, active, onPreviewOpen,
}: {
  root: string;
  openRequest?: { path: string } | null;
  active?: boolean;
  onPreviewOpen?: () => void;
}) {
  const S = useStrings();
  const [path, setPath] = useState("");
  const [data, setData] = useState<{ base: string; entries: Entry[] } | null>(null);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [uploading, setUploading] = useState(false);
  const [reloadTick, setReloadTick] = useState(0);
  const [showPath, setShowPath] = useState(false);
  const [richView, setRichView] = useState<"rendered" | "source">("rendered");
  const detailsRef = useRef<HTMLDivElement>(null);

  const [renderedRoot, setRenderedRoot] = useState(root);
  if (renderedRoot !== root) {
    setRenderedRoot(root);
    setPath("");
    setPreview(null);
    setData(null);
    previewSeq++;
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

  useEffect(() => {
    if (!showPath) return;
    const onDown = (e: MouseEvent) => {
      if (!detailsRef.current?.contains(e.target as Node)) setShowPath(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setShowPath(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [showPath]);

  const onPreviewOpenRef = useRef(onPreviewOpen);
  onPreviewOpenRef.current = onPreviewOpen;

  const previewPath = useCallback(async (filePath: string) => {
    onPreviewOpenRef.current?.();
    const name = filePath.includes("/") ? filePath.slice(filePath.lastIndexOf("/") + 1) : filePath;
    const ext = extOf(name);
    const nonce = ++previewSeq;
    const present = (p: Preview) => { if (nonce === previewSeq) setPreview(p); };
    setRichView("rendered");
    if (IMAGE_EXTS.has(ext)) { present({ path: filePath, name, kind: "image" }); return; }
    if (ext === "pdf") { present({ path: filePath, name, kind: "pdf" }); return; }
    const isHtml = HTML_EXTS.has(ext);
    const isMd = ext === "md";
    if (!isHtml && !TEXT_EXTS.has(ext)) {
      present({ path: filePath, name, kind: "unsupported" });
      return;
    }
    const { ok, data } = await api.fetchWorkspacePreview(root, filePath);
    if (nonce !== previewSeq) return;
    if (!ok || !data) { present({ path: filePath, name, kind: "unsupported" }); return; }
    present({
      path: filePath,
      name,
      kind: isHtml ? "html" : isMd ? "md" : "text",
      content: data.content || "",
      truncated: !!data.truncated,
    });
  }, [root]);

  const handledOpenRequest = useRef<{ path: string } | null>(null);
  useEffect(() => {
    if (!openRequest || handledOpenRequest.current === openRequest) return;
    handledOpenRequest.current = openRequest;
    const target = openRequest.path;
    setPath(dirOf(target));
    setReloadTick((t) => t + 1);
    void previewPath(target);
  }, [openRequest, previewPath]);

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

  if (preview) {
    const src = api.workspaceContentUrl(root, preview.path);
    const dl = api.workspaceContentUrl(root, preview.path, true);
    const sourceView = preview.kind === "text" || ((preview.kind === "md" || preview.kind === "html") && richView === "source");
    const copyVisible = () => {
      const text = preview.content || "";
      if (!text) return;
      void navigator.clipboard.writeText(text).then(() => showToast(S.common.copied));
    };
    return (
      <div className="ws-browser">
        <div className="ws-preview-bar">
          <button type="button" className="ws-back" onClick={() => { setPreview(null); setReloadTick((t) => t + 1); }}>
            {S.files.backToList}
          </button>
          <span className="ws-preview-name" title={preview.path}>{preview.name}</span>
          {(preview.kind === "html" || preview.kind === "md") && (
            <div className="ws-view-toggle">
              {(["rendered", "source"] as const).map((key) => (
                <button key={key} type="button" className={richView === key ? "on" : ""}
                        onClick={() => setRichView(key)}>
                  {key === "rendered" ? S.files.htmlRendered : S.files.htmlSource}
                </button>
              ))}
            </div>
          )}
          {sourceView && preview.content != null && (
            <button type="button" className="ws-tool-btn ws-copy-btn" title={S.files.copyVisible}
                    onClick={copyVisible}>
              <IconCopy />
            </button>
          )}
          <a className="ws-tool-btn" href={dl} download={preview.name}>{S.files.download}</a>
        </div>
        <div className="ws-preview-body">
          {preview.kind === "image" && <img src={src} alt={preview.name} />}
          {preview.kind === "pdf" && <iframe src={src} title={preview.name} />}
          {preview.kind === "html" && richView === "rendered" && preview.content !== undefined && (
            <iframe srcDoc={preview.content} title={preview.name} sandbox="allow-scripts" />
          )}
          {preview.kind === "md" && richView === "rendered" && (
            <>
              <div className="ws-md">
                <Markdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeRaw, rehypeKatex]}
                  components={{
                  img: ({ src: href, alt }) => (
                    <img
                      src={typeof href === "string" && !EXTERNAL_REF_RE.test(href)
                        ? api.workspaceContentUrl(root, resolveRelative(dirOf(preview.path), href))
                        : href}
                      alt={alt || ""}
                    />
                  ),
                  a: ({ href, children }) => {
                    if (typeof href !== "string" || href.startsWith("#") || EXTERNAL_REF_RE.test(href)) {
                      return <a href={href} target={EXTERNAL_REF_RE.test(href || "") ? "_blank" : undefined} rel="noreferrer">{children}</a>;
                    }
                    const target = resolveRelative(dirOf(preview.path), href);
                    return <a href={api.workspaceContentUrl(root, target)} onClick={(e) => { e.preventDefault(); void previewPath(target); }}>{children}</a>;
                  },
                }}>{preview.content || ""}</Markdown>
              </div>
              {preview.truncated && <p className="ws-trunc">{S.files.previewTruncated}</p>}
            </>
          )}
          {(preview.kind === "text" || ((preview.kind === "md" || preview.kind === "html") && richView === "source")) && (
            <>
              <pre className="ws-source">{preview.content}</pre>
              {preview.truncated && <p className="ws-trunc">{S.files.previewTruncated}</p>}
            </>
          )}
          {preview.kind === "unsupported" && <p className="ws-muted">{S.files.previewUnsupported}</p>}
        </div>
      </div>
    );
  }

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
        <div className="ws-details-wrap" ref={detailsRef}>
          <button type="button" className={"ws-tool-btn" + (showPath ? " on" : "")} onClick={() => setShowPath((v) => !v)}>
            {S.files.details}
          </button>
          {showPath && (
            <div className="ws-details-pop">
              <p className="ws-details-label">{S.files.workspacePath}</p>
              <p className="ws-details-path">{root}</p>
              <div className="ws-details-acts">
                <button type="button" title={S.chat.copyPath} onClick={() => { void navigator.clipboard.writeText(root); showToast(S.common.copied); }}><IconCopy /></button>
                <button type="button" title={S.chat.openFinder} onClick={() => void api.openPathApi(root, "finder")}><IconFinder /></button>
                <button type="button" title={S.chat.openTerminal} onClick={() => void api.openPathApi(root, "terminal")}><IconTerminal /></button>
              </div>
            </div>
          )}
        </div>
        <button type="button" className="ws-tool-btn" onClick={() => setReloadTick((t) => t + 1)}>{S.files.refresh}</button>
        <label className="ws-tool-btn ws-upload">
          <input type="file" multiple hidden disabled={uploading} onChange={onUpload} />
          {uploading ? S.files.uploading : S.files.upload}
        </label>
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
                            onClick={() => { if (entry.kind === "dir") setPath(target); else void previewPath(target); }}>
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
