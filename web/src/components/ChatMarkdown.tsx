import { memo, type ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { MarkdownCode } from "./CodeFrame";

/**
 * Frozen plugin / component maps so a per-frame stream bump does not remount
 * every code block (new `components={{}}` identity was the flash). Streaming
 * skips KaTeX + highlight; settle flips `streaming` once and highlights.
 */
const STREAM_REMARK = [remarkGfm];
const SETTLED_REMARK = [remarkGfm, remarkMath];
const SETTLED_REHYPE = [rehypeKatex];

function StreamCode({ className, children }: { className?: string; children?: ReactNode }) {
  const text = String(children ?? "").replace(/\n$/, "");
  if (className || text.includes("\n")) return <pre className="stream-pre">{text}</pre>;
  return <code>{children}</code>;
}

const STREAMING_COMPONENTS = {
  pre: ({ children }: { children?: ReactNode }) => <>{children}</>,
  code: StreamCode,
};

const SETTLED_COMPONENTS = {
  pre: ({ children }: { children?: ReactNode }) => <>{children}</>,
  code: MarkdownCode,
};

export const ChatMarkdown = memo(function ChatMarkdown({
  text,
  streaming = false,
}: {
  text: string;
  streaming?: boolean;
}) {
  return (
    <Markdown
      remarkPlugins={streaming ? STREAM_REMARK : SETTLED_REMARK}
      rehypePlugins={streaming ? undefined : SETTLED_REHYPE}
      components={streaming ? STREAMING_COMPONENTS : SETTLED_COMPONENTS}
    >
      {text}
    </Markdown>
  );
});
