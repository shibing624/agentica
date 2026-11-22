import * as ContextMenu from "@radix-ui/react-context-menu";
import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState, type ReactElement, type ReactNode } from "react";
import { createPortal } from "react-dom";
import * as api from "../api";
import { useStrings } from "../i18n";
import { IconClose, IconCopy, IconDownload, IconExternal, IconFinder, IconSidebarRight } from "../icons";
import { classifyPreviewTarget, isAbsOrHomePath, splitExternalTarget, toWorkspaceRelative } from "../lib/filePath";
import { baseName } from "../lib/panelTabs";
import { getState, showToast } from "../store";

const FilePreview = lazy(() => import("../workspace/WorkspaceBrowser").then((m) => ({ default: m.FilePreview })));

export type FileResource = { target: api.WorkspaceTarget; path: string };
type FileSource = FileResource | ((path: string) => Promise<FileResource | null>);

export function panelFileResource(path: string, scope: api.WorkspaceScope | null, canOpenHost: boolean): FileResource | null {
  if (!isAbsOrHomePath(path)) return scope ? { target: scope, path } : null;
  const relative = scope && toWorkspaceRelative(path, scope.root.replaceAll("\\", "/"));
  if (relative !== null && scope) return { target: scope, path: relative };
  if (!canOpenHost) return null;
  const { targetRoot, targetRel } = splitExternalTarget(path, "");
  return { target: targetRoot, path: targetRel };
}

function absolutePath({ target, path }: FileResource): string {
  const root = typeof target === "string" ? target : target.root;
  return path ? `${root.replace(/[\\/]+$/, "")}/${path}` : root;
}

export async function downloadFile(source: FileSource, path: string): Promise<boolean> {
  const file = typeof source === "function" ? await source(path) : source;
  if (!file) return false;
  const link = document.createElement("a");
  link.href = api.workspaceContentUrl(file.target, file.path, true);
  link.download = baseName(file.path);
  link.click();
  return true;
}

function FloatingPreview({ path, source, canOpenHost, onClose, trigger }: {
  path: string; source: FileSource; canOpenHost: boolean; onClose: () => void; trigger: HTMLElement | null;
}) {
  const S = useStrings();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [file, setFile] = useState(path);
  const [loaded, setLoaded] = useState<{ file: string; resource: FileResource | null } | null>(null);
  const present = typeof source === "function" ? loaded?.file === file ? loaded : null : {
    file,
    resource: file === path ? source : panelFileResource(file, typeof source.target === "object" ? source.target : null, canOpenHost),
  };
  useLayoutEffect(() => {
    const dialog = dialogRef.current!;
    dialog.showModal();
    return () => {
      dialog.close();
      trigger?.focus({ preventScroll: true });
    };
  }, [trigger]);
  useEffect(() => {
    if (typeof source !== "function") return;
    let active = true;
    void source(file).then((resource) => { if (active) setLoaded({ file, resource }); });
    return () => { active = false; };
  }, [file, source]);
  const resource = present?.resource;
  const scope = typeof resource?.target === "object" ? resource.target : undefined;
  return createPortal(
    <dialog ref={dialogRef} className="chat-file-float" aria-label={baseName(file)}
            onKeyDown={(event) => event.stopPropagation()}
            onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <div className="chat-file-float-head">
        <span>{baseName(file)}</span>
        <button type="button" className="ib" aria-label={S.common.close} onClick={onClose}><IconClose /></button>
      </div>
      {resource ? <Suspense fallback={<p className="ws-muted">{S.common.loading}</p>}>
        <FilePreview key={file} root={scope?.root ?? ""} scope={scope}
                     path={scope ? resource.path : absolutePath(resource)} allowNativeOpen={canOpenHost}
                     onOpenFile={(next) => setFile(scope && typeof source === "function" && !isAbsOrHomePath(next)
                       ? absolutePath({ target: scope, path: next }) : next)} />
      </Suspense> : <p className="ws-muted">{present ? S.chat.openFailed : S.common.loading}</p>}
    </dialog>, document.body,
  );
}

export function FileContextMenu({ path, source, canOpenHost, onOpenSidebar, directory = false, extraItems, children }: {
  path: string;
  source: FileSource;
  canOpenHost: boolean;
  onOpenSidebar?: () => void;
  directory?: boolean;
  extraItems?: ReactNode;
  children: ReactElement;
}) {
  const S = useStrings();
  const [floating, setFloating] = useState<{ path: string; source: FileSource } | null>(null);
  const trigger = useRef<HTMLElement>(null);
  const canPreview = !directory && classifyPreviewTarget(path) !== "unsupported";
  const platform = getState().serverPlatform;
  const fullPath = typeof source === "function" ? path : absolutePath(source);
  const hostPath = platform === "win32" ? fullPath.replaceAll("/", "\\") : fullPath;
  const revealLabel = platform === "darwin" ? S.files.revealInFinder
    : platform === "win32" ? S.files.revealInExplorer : S.files.revealInFolder;
  const openHost = async (app: "reveal" | "finder") => {
    if (typeof source !== "function" && typeof source.target === "object" && !directory) {
      const { ok, data } = await api.statWorkspaceFiles(source.target, [source.path]);
      if (!ok || !data?.existing.includes(source.path)) { showToast(S.chat.openFailed); return; }
    }
    if (!(await api.openPathApi(hostPath, app)).ok) showToast(S.chat.openFailed);
  };
  return <>
    <ContextMenu.Root>
      <ContextMenu.Trigger asChild ref={(node) => { trigger.current = node; }} aria-haspopup="menu"
        onKeyDown={(event) => {
          if (event.key !== "ContextMenu" && !(event.shiftKey && event.key === "F10")) return;
          event.preventDefault();
          const bounds = event.currentTarget.getBoundingClientRect();
          event.currentTarget.dispatchEvent(new MouseEvent("contextmenu", {
            bubbles: true, cancelable: true, clientX: bounds.left, clientY: bounds.bottom,
          }));
        }}>{children}</ContextMenu.Trigger>
      <ContextMenu.Portal>
        <ContextMenu.Content className="pt-menu chat-file-menu" aria-label={baseName(path)} onKeyDown={(event) => event.stopPropagation()}>
          {!directory && <>
            <ContextMenu.Item className="pt-menu-item" disabled={!canPreview || !onOpenSidebar} onSelect={onOpenSidebar}><IconSidebarRight />{S.files.openInSidebar}</ContextMenu.Item>
            <ContextMenu.Item className="pt-menu-item" disabled={!canPreview} onSelect={() => setFloating({ path, source })}><IconExternal />{S.files.openInFloat}</ContextMenu.Item>
          </>}
          {canOpenHost && <>
            <ContextMenu.Item className="pt-menu-item" onSelect={() => void openHost("reveal")}><IconFinder />{revealLabel}</ContextMenu.Item>
            {!directory && <ContextMenu.Item className="pt-menu-item" onSelect={() => void openHost("finder")}><IconExternal />{S.files.openDefault}</ContextMenu.Item>}
          </>}
          <ContextMenu.Item className="pt-menu-item" onSelect={async () => {
            try {
              await navigator.clipboard.writeText(hostPath);
              showToast(S.common.copied);
            } catch { showToast(directory ? S.files.copyFolderPathFailed : S.files.copyFilePathFailed); }
          }}><IconCopy />{directory ? S.files.copyFolderPath : S.files.copyFilePath}</ContextMenu.Item>
          {!directory && <ContextMenu.Item className="pt-menu-item" onSelect={async () => {
            if (!await downloadFile(source, path)) showToast(S.chat.openFailed);
          }}><IconDownload />{S.files.saveFileAs}</ContextMenu.Item>}
          {extraItems && <><ContextMenu.Separator className="file-menu-separator" />{extraItems}</>}
        </ContextMenu.Content>
      </ContextMenu.Portal>
    </ContextMenu.Root>
    {floating && <FloatingPreview {...floating} canOpenHost={canOpenHost} trigger={trigger.current} onClose={() => setFloating(null)} />}
  </>;
}