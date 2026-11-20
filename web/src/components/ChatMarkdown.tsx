import { memo, useMemo, type ReactNode } from "react";
import Markdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { MarkdownCode } from "./CodeFrame";
import { PreviewableImage } from "./ImageLightbox";
import { GFM_PLUGINS, markdownUrlTransform, localFileLinkTarget, openLocalFile } from "../lib/markdownPlugins";

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

type MarkdownLinkProps = {
  href?: string;
  children?: ReactNode;
  onOpenFile?: (path: string) => void;
};

export function MarkdownLink({ href, children, onOpenFile }: MarkdownLinkProps) {
  if (typeof href !== "string" || href === "") return <a aria-disabled="true">{children}</a>;
  const target = localFileLinkTarget(href);
  if (!target) return <a href={href}>{children}</a>;
  return (
    <a
      href={href}
      data-open-file={target.path}
      onClick={(e) => { e.preventDefault(); openLocalFile(target, onOpenFile); }}
    >
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
}: {
  text: string;
  streaming?: boolean;
  onOpenFile?: (path: string) => void;
}) {
  const components = useMemo(() => {
    const a = ({ href, children }: { href?: string; children?: ReactNode }) => (
      <MarkdownLink href={href} onOpenFile={onOpenFile}>{children}</MarkdownLink>
    );
    return streaming
      ? { ...STREAMING_COMPONENTS, a }
      : { ...SETTLED_COMPONENTS, a };
  }, [streaming, onOpenFile]);
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
