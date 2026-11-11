import {
  useCallback, useEffect, useMemo, useRef, useState,
  type CSSProperties, type MouseEvent, type ReactNode,
} from "react";
import { HIGHLIGHT_LIMIT, highlightToHtml, splitHighlightedHtmlLines } from "../lib/highlight";
import { IconCopy, IconTick } from "../icons";
import { showToast } from "../store";
import { useStrings } from "../i18n";

type Bars = {
  v: boolean;
  h: boolean;
  top: number;
  height: number;
  left: number;
  width: number;
};

const MIN_THUMB = 24;

function CodeScroll({ html, wrap }: { html: string; wrap: boolean }) {
  const scroller = useRef<HTMLPreElement>(null);
  const [bars, setBars] = useState<Bars>({ v: false, h: false, top: 0, height: 0, left: 0, width: 0 });
  // One object per html: a fresh `{__html}` makes React rewrite the block's
  // innerHTML on every render, and each scroll-bar update is a render.
  const markup = useMemo(() => ({ __html: html }), [html]);

  const sync = useCallback(() => {
    const el = scroller.current;
    if (!el) return;
    const v = el.scrollHeight > el.clientHeight + 1;
    const h = !wrap && el.scrollWidth > el.clientWidth + 1;
    const vTrack = el.clientHeight;
    const hTrack = el.clientWidth;
    const vThumb = v ? Math.max(MIN_THUMB, (el.clientHeight / el.scrollHeight) * vTrack) : 0;
    const hThumb = h ? Math.max(MIN_THUMB, (el.clientWidth / el.scrollWidth) * hTrack) : 0;
    const vRange = el.scrollHeight - el.clientHeight;
    const hRange = el.scrollWidth - el.clientWidth;
    setBars({
      v,
      h,
      top: v && vRange > 0 ? (el.scrollTop / vRange) * (vTrack - vThumb) : 0,
      height: vThumb,
      left: h && hRange > 0 ? (el.scrollLeft / hRange) * (hTrack - hThumb) : 0,
      width: hThumb,
    });
  }, [wrap]);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    if (wrap && el.scrollLeft !== 0) el.scrollLeft = 0;
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    const inner = el.firstElementChild;
    if (inner) ro.observe(inner);
    el.addEventListener("scroll", sync, { passive: true });
    return () => {
      ro.disconnect();
      el.removeEventListener("scroll", sync);
    };
  }, [html, wrap, sync]);

  const jump = (axis: "x" | "y", e: MouseEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest(".sb-thumb")) return;
    const el = scroller.current;
    if (!el) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (axis === "y") {
      const thumb = bars.height;
      const max = rect.height - thumb;
      const ratio = max > 0 ? Math.max(0, Math.min(1, (e.clientY - rect.top - thumb / 2) / max)) : 0;
      el.scrollTop = ratio * (el.scrollHeight - el.clientHeight);
    } else {
      const thumb = bars.width;
      const max = rect.width - thumb;
      const ratio = max > 0 ? Math.max(0, Math.min(1, (e.clientX - rect.left - thumb / 2) / max)) : 0;
      el.scrollLeft = ratio * (el.scrollWidth - el.clientWidth);
    }
  };

  const drag = (axis: "x" | "y", e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const el = scroller.current;
    if (!el) return;
    const startPtr = axis === "y" ? e.clientY : e.clientX;
    const startScroll = axis === "y" ? el.scrollTop : el.scrollLeft;
    const track = axis === "y" ? el.clientHeight : el.clientWidth;
    const content = axis === "y" ? el.scrollHeight : el.scrollWidth;
    const thumb = axis === "y" ? bars.height : bars.width;
    const maxThumb = track - thumb;
    const range = content - track;
    const move = (ev: globalThis.MouseEvent) => {
      const delta = (axis === "y" ? ev.clientY : ev.clientX) - startPtr;
      const next = startScroll + (maxThumb > 0 ? (delta / maxThumb) * range : 0);
      if (axis === "y") el.scrollTop = next;
      else el.scrollLeft = next;
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  return (
    <div className={"code-scroll" + (bars.v ? " has-v" : "") + (bars.h ? " has-h" : "")}>
      <pre ref={scroller} className="code-frame-pre">
        <code className="hljs" dangerouslySetInnerHTML={markup} />
      </pre>
      {bars.v && (
        <div className="sb sb-y" onMouseDown={(e) => jump("y", e)}>
          <div className="sb-thumb" style={{ top: bars.top, height: bars.height }} onMouseDown={(e) => drag("y", e)} />
        </div>
      )}
      {bars.h && (
        <div className="sb sb-x" onMouseDown={(e) => jump("x", e)}>
          <div className="sb-thumb" style={{ left: bars.left, width: bars.width }} onMouseDown={(e) => drag("x", e)} />
        </div>
      )}
    </div>
  );
}

export function CodeFrame({
  code,
  language,
  highlight = true,
  lineNumbers = false,
  wrap = false,
  pane = false,
}: {
  code: string;
  language?: string;
  highlight?: boolean;
  lineNumbers?: boolean;
  wrap?: boolean;
  /** Flush edge-to-edge sidebar document preview style (dsh `CodeBody`). */
  pane?: boolean;
}) {
  const S = useStrings();
  const [copied, setCopied] = useState(false);
  const trimmed = lineNumbers && code.endsWith("\n") ? code.slice(0, -1) : code;
  const { html, lineCount } = useMemo(() => {
    const rawHtml = highlightToHtml(trimmed, language, highlight);
    if (!lineNumbers) return { html: rawHtml, lineCount: 0 };
    return splitHighlightedHtmlLines(rawHtml);
  }, [trimmed, language, highlight, lineNumbers]);

  const copy = () => {
    if (!code) return;
    void navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
      if (!pane) showToast(S.common.copied);
    });
  };

  const style = lineNumbers
    ? ({ "--code-line-no-w": `${Math.max(2, String(lineCount).length)}ch` } as CSSProperties)
    : undefined;

  return (
    <div
      className={
        "code-frame"
        + (lineNumbers ? " numbered" : "")
        + (wrap ? " wrap" : "")
        + (pane ? " pane" : "")
      }
      data-line-numbers={lineNumbers || undefined}
      data-code-wrap={wrap}
      style={style}
    >
      <div className="code-frame-bar">
        <span className="code-frame-lang">{language || "text"}</span>
        <button
          type="button"
          className="code-frame-copy"
          title={copied ? S.common.copied : S.common.copy}
          aria-label={copied ? S.common.copied : S.common.copy}
          onClick={copy}
        >
          {copied ? <IconTick /> : <IconCopy />}
        </button>
      </div>
      <CodeScroll html={html} wrap={wrap} />
    </div>
  );
}

export function MarkdownCode({ className, children }: { className?: string; children?: ReactNode }) {
  const text = String(children ?? "").replace(/\n$/, "");
  const lang = /language-(\w+)/.exec(className || "")?.[1];
  if (lang || text.includes("\n")) {
    return <CodeFrame code={text} language={lang} highlight={text.length <= HIGHLIGHT_LIMIT} />;
  }
  return <code className={className}>{children}</code>;
}
