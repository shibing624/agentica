import { memo, useMemo, type ComponentProps, type MouseEvent, type ReactNode } from "react";
import Markdown, { type ExtraProps } from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { MarkdownCode } from "./CodeFrame";
import { PreviewableImage } from "./ImageLightbox";
import { ChatFileLink, type ChatFileContext } from "./ChatFileLink";
import { workspaceContentUrl } from "../api";
import { classifyPreviewTarget, splitExternalTarget } from "../lib/filePath";
import { GFM_PLUGINS, localMarkdownPath, markdownUrlTransform } from "../lib/markdownPlugins";

/**
 * Frozen plugin / component maps so a per-frame stream bump does not remount
 * every code block (new `components={{}}` identity was the flash). Streaming
 * skips KaTeX + highlight; settle flips `streaming` once and highlights.
 */
const STREAM_REMARK = GFM_PLUGINS;
const SETTLED_REMARK = [...GFM_PLUGINS, remarkMath];
const SETTLED_REHYPE = [rehypeKatex];

function StreamCode({ className, children }: { className?: string; children?: ReactNode }) {
  const text = String(children ?? "").replace(/\n$/, "");
  if (className || text.includes("\n")) return <pre className="stream-pre">{text}</pre>;
  return <code>{children}</code>;
}

function MdTable({ children }: { children?: ReactNode }) {
  return <div className="md-table"><table>{children}</table></div>;
}

const STREAMING_COMPONENTS = {
  pre: ({ children }: { children?: ReactNode }) => <>{children}</>,
  code: StreamCode,
  table: MdTable,
};

type MarkdownAnchorProps = Pick<ComponentProps<"a">, "href" | "title" | "children">;

type MarkdownLinkProps = MarkdownAnchorProps & {
  localPath?: string;
  onOpenFile?: (path: string) => void;
  fileContext?: ChatFileContext;
};

export function MarkdownLink({ href, title, children, localPath, onOpenFile, fileContext }: MarkdownLinkProps) {
  if (localPath === undefined) return href ? <a href={href} title={title}>{children}</a> : <>{children}</>;
  if (fileContext) return <ChatFileLink path={localPath} context={fileContext} title={title} onOpenFile={onOpenFile}>{children}</ChatFileLink>;
  const { targetRoot, targetRel } = splitExternalTarget(localPath, "");
  const preview = onOpenFile && classifyPreviewTarget(localPath) !== "unsupported"
    ? (e: MouseEvent<HTMLAnchorElement>) => {
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      onOpenFile(localPath);
    }
    : undefined;
  return (
    <a href={workspaceContentUrl(targetRoot, targetRel, true)} title={title} download onClick={preview}>
      {children}
    </a>
  );
}

const SETTLED_COMPONENTS = {
  pre: ({ children }: { children?: ReactNode }) => <>{children}</>,
  code: MarkdownCode,
  table: MdTable,
  img: ({ src, alt }: { src?: string; alt?: string }) => <PreviewableImage src={src} alt={alt || ""} />,
};

export const ChatMarkdown = memo(function ChatMarkdown({
  text,
  streaming = false,
  onOpenFile,
  fileContext,
}: {
  text: string;
  streaming?: boolean;
  onOpenFile?: (path: string) => void;
  fileContext?: ChatFileContext;
}) {
  const components = useMemo(() => {
    const a = ({ node, ...props }: MarkdownAnchorProps & ExtraProps) => (
      <MarkdownLink {...props} localPath={localMarkdownPath(node)} onOpenFile={onOpenFile} fileContext={fileContext} />
    );
    return streaming
      ? { ...STREAMING_COMPONENTS, a }
      : { ...SETTLED_COMPONENTS, a };
  }, [streaming, onOpenFile, fileContext]);
  return (
    <Markdown
      remarkPlugins={streaming ? STREAM_REMARK : SETTLED_REMARK}
      rehypePlugins={streaming ? undefined : SETTLED_REHYPE}
      components={components}
      urlTransform={markdownUrlTransform}
    >
      {text}
    </Markdown>
  );
});
