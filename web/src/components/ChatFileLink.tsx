import * as ContextMenu from "@radix-ui/react-context-menu";
import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import * as api from "../api";
import { getStrings, useStrings } from "../i18n";
import { FileTypeIcon, IconClose, IconCopy, IconDownload, IconExternal, IconFinder, IconSidebarRight } from "../icons";
import { classifyPreviewTarget, isAbsOrHomePath, splitExternalTarget, toWorkspaceRelative } from "../lib/filePath";
import { baseName } from "../lib/panelTabs";
import { showToast } from "../store";

const FilePreview = lazy(() => import("../workspace/WorkspaceBrowser").then((m) => ({ default: m.FilePreview })));

export type ChatFileContext = {
  sessionId: string;
  workspace: string;
  canOpenHost: boolean;
  platform: string;
};

type LinkTarget = { target: api.WorkspaceTarget; path: string };

async function resolveTarget(path: string, context: ChatFileContext): Promise<LinkTarget | null> {
  if (toWorkspaceRelative(path, context.workspace.replaceAll("\\", "/")) !== null) {
    const { ok, data } = await api.fetchWorkspaceScope(context.sessionId);
    if (ok && data) {
      const relative = toWorkspaceRelative(path, data.root.replaceAll("\\", "/"));
      return relative === null ? null : { target: data, path: relative };
    }
  }
  if (!context.canOpenHost) return null;
  const { targetRoot, targetRel } = splitExternalTarget(path, "");
  return { target: targetRoot, path: targetRel };
}

function FloatingPreview({ path, context, onClose, trigger }: {
  path: string; context: ChatFileContext; onClose: () => void; trigger: HTMLElement | null;
}) {
  const S = useStrings();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [file, setFile] = useState(path);
  const [loaded, setLoaded] = useState<{ file: string; target: LinkTarget | null } | null>(null);
  useLayoutEffect(() => {
    const dialog = dialogRef.current!;
    dialog.showModal();
    return () => {
      dialog.close();
      trigger?.focus({ preventScroll: true });
    };
  }, [trigger]);
  useEffect(() => {
    let active = true;
    void resolveTarget(file, context).then((target) => { if (active) setLoaded({ file, target }); });
    return () => { active = false; };
  }, [file, context]);
  const result = loaded?.file === file ? loaded : null;
  const target = result?.target?.target;
  const scope = typeof target === "object" ? target : undefined;
  return createPortal(
    <dialog ref={dialogRef} className="chat-file-float" aria-label={baseName(file)}
            onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <div className="chat-file-float-head">
        <span>{baseName(file)}</span>
        <button type="button" className="ib" aria-label={S.common.close} onClick={onClose}><IconClose /></button>
      </div>
      {target ? <Suspense fallback={<p className="ws-muted">{S.common.loading}</p>}>
        <FilePreview key={file} root={scope?.root ?? context.workspace} scope={scope} path={file}
                     allowNativeOpen={context.canOpenHost}
                     onOpenFile={(next) => setFile(isAbsOrHomePath(next) ? next : `${(scope?.root ?? context.workspace).replace(/[\\/]+$/, "")}/${next}`)} />
      </Suspense> : <p className="ws-muted">{result ? S.chat.openFailed : S.common.loading}</p>}
    </dialog>, document.body,
  );
}

export function ChatFileLink({ path, title, children, context, onOpenFile }: {
  path: string; title?: string; children?: ReactNode; context: ChatFileContext; onOpenFile?: (path: string) => void;
}) {
  const S = useStrings();
  const [floating, setFloating] = useState(false);
  const trigger = useRef<HTMLElement>(null);
  const triggerProps = {
    ref: (element: HTMLElement | null) => { trigger.current = element; },
    title,
    className: "chat-file-link",
    "aria-haspopup": "menu" as const,
    onKeyDown: (event: KeyboardEvent<HTMLElement>) => {
      if (event.key !== "ContextMenu" && !(event.shiftKey && event.key === "F10")) return;
      event.preventDefault();
      const bounds = event.currentTarget.getBoundingClientRect();
      // Send both keyboard shortcuts through Radix's pointer-positioned trigger.
      event.currentTarget.dispatchEvent(new MouseEvent("contextmenu", {
        bubbles: true, cancelable: true, clientX: bounds.left, clientY: bounds.bottom,
      }));
    },
  };
  const canPreview = classifyPreviewTarget(path) !== "unsupported";
  if (!context.canOpenHost && toWorkspaceRelative(path, context.workspace.replaceAll("\\", "/")) === null) return <>{children}</>;
  const download = async () => {
    const file = await resolveTarget(path, context);
    if (!file) { showToast(getStrings().chat.openFailed); return; }
    const link = document.createElement("a");
    link.href = api.workspaceContentUrl(file.target, file.path, true);
    link.download = baseName(path);
    link.click();
  };
  const openHost = async (app: "reveal" | "finder") => {
    const { ok } = await api.openPathApi(path, app);
    if (!ok) showToast(getStrings().chat.openFailed);
  };
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(path);
      showToast(getStrings().common.copied);
    } catch { showToast(getStrings().files.copyFilePathFailed); }
  };
  const revealLabel = context.platform === "darwin" ? S.files.revealInFinder
    : context.platform === "win32" ? S.files.revealInExplorer : S.files.revealInFolder;
  const { targetRoot, targetRel } = splitExternalTarget(path, "");
  const contents = <><FileTypeIcon path={path} size={15} />{children}</>;
  return <>
    <ContextMenu.Root>
      <ContextMenu.Trigger asChild>
        {context.canOpenHost ? <a {...triggerProps}
             href={api.workspaceContentUrl(targetRoot, targetRel, true)} download
             onClick={(event) => {
               if (!canPreview || !onOpenFile || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
               event.preventDefault();
               onOpenFile(path);
             }}>{contents}</a>
          : <button {...triggerProps} type="button"
                    onClick={() => canPreview && onOpenFile ? onOpenFile(path) : void download()}>{contents}</button>}
      </ContextMenu.Trigger>
      <ContextMenu.Portal>
        <ContextMenu.Content className="pt-menu chat-file-menu" aria-label={baseName(path)}>
          <ContextMenu.Item className="pt-menu-item" disabled={!canPreview || !onOpenFile} onSelect={() => onOpenFile?.(path)}><IconSidebarRight />{S.files.openInSidebar}</ContextMenu.Item>
          <ContextMenu.Item className="pt-menu-item" disabled={!canPreview} onSelect={() => setFloating(true)}><IconExternal />{S.files.openInFloat}</ContextMenu.Item>
          {context.canOpenHost && <>
            <ContextMenu.Item className="pt-menu-item" onSelect={() => void openHost("reveal")}><IconFinder />{revealLabel}</ContextMenu.Item>
            <ContextMenu.Item className="pt-menu-item" onSelect={() => void openHost("finder")}><IconExternal />{S.files.openDefault}</ContextMenu.Item>
          </>}
          <ContextMenu.Item className="pt-menu-item" onSelect={() => void copy()}><IconCopy />{S.files.copyFilePath}</ContextMenu.Item>
          <ContextMenu.Item className="pt-menu-item" onSelect={() => void download()}><IconDownload />{S.files.saveFileAs}</ContextMenu.Item>
        </ContextMenu.Content>
      </ContextMenu.Portal>
    </ContextMenu.Root>
    {floating && <FloatingPreview path={path} context={context} trigger={trigger.current} onClose={() => setFloating(false)} />}
  </>;
}