import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { HIGHLIGHT_LIMIT, highlightToHtml } from "../lib/highlight";
import { IconCopy } from "../icons";
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

function CodeScroll({ html }: { html: string }) {
  const scroller = useRef<HTMLPreElement>(null);
  const [bars, setBars] = useState<Bars>({ v: false, h: false, top: 0, height: 0, left: 0, width: 0 });

  const sync = useCallback(() => {
    const el = scroller.current;
    if (!el) return;
    const v = el.scrollHeight > el.clientHeight + 1;
    const h = el.scrollWidth > el.clientWidth + 1;
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
  }, []);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
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
  }, [html, sync]);

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
        <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
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
}: {
  code: string;
  language?: string;
  highlight?: boolean;
}) {
  const S = useStrings();
  const html = useMemo(
    () => highlightToHtml(code, language, highlight),
    [code, language, highlight],
  );
  const copy = () => {
    if (!code) return;
    void navigator.clipboard.writeText(code).then(() => showToast(S.common.copied));
  };
  return (
    <div className="code-frame">
      <div className="code-frame-bar">
        <span className="code-frame-lang">{language || "text"}</span>
        <button type="button" className="code-frame-copy" title={S.common.copy} onClick={copy}>
          <IconCopy />
        </button>
      </div>
      <CodeScroll html={html} />
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
