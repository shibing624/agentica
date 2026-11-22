import type { ReactNode } from "react";
import * as api from "../api";
import { getStrings } from "../i18n";
import { FileTypeIcon } from "../icons";
import { classifyPreviewTarget, splitExternalTarget, toWorkspaceRelative } from "../lib/filePath";
import { showToast } from "../store";
import { downloadFile, FileContextMenu, type FileResource } from "./FileContextMenu";

export type ChatFileContext = {
  sessionId: string;
  workspace: string;
  canOpenHost: boolean;
};

async function resolveTarget(path: string, context: ChatFileContext): Promise<FileResource | null> {
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

export function ChatFileLink({ path, title, children, context, onOpenFile }: {
  path: string; title?: string; children?: ReactNode; context: ChatFileContext; onOpenFile?: (path: string) => void;
}) {
  if (!context.canOpenHost && toWorkspaceRelative(path, context.workspace.replaceAll("\\", "/")) === null) return <>{children}</>;
  const source = (file: string) => resolveTarget(file, context);
  const canPreview = classifyPreviewTarget(path) !== "unsupported";
  const openSidebar = onOpenFile && (() => onOpenFile(path));
  const { targetRoot, targetRel } = splitExternalTarget(path, "");
  const contents = <><FileTypeIcon path={path} size={15} />{children}</>;
  return <FileContextMenu path={path} source={source} canOpenHost={context.canOpenHost} onOpenSidebar={openSidebar}>
    {context.canOpenHost ? <a className="chat-file-link" title={title}
        href={api.workspaceContentUrl(targetRoot, targetRel, true)} download
        onClick={(event) => {
          if (!canPreview || !openSidebar || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
          event.preventDefault();
          openSidebar();
        }}>{contents}</a>
      : <button type="button" className="chat-file-link" title={title} onClick={async () => {
          if (canPreview && openSidebar) openSidebar();
          else if (!await downloadFile(source, path)) showToast(getStrings().chat.openFailed);
        }}>{contents}</button>}
  </FileContextMenu>;
}
