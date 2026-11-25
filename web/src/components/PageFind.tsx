import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useStrings } from "../i18n";
import { IconChevronDown, IconClose, IconSearch } from "../icons";
import { pageMatches, revealPageMatch } from "../lib/pageFind";

const EMPTY = { matches: [] as Range[], index: -1 };
const MATCH = "page-find-match";
const ACTIVE = "page-find-active";

export function PageFind() {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [{ matches, index }, setResult] = useState(EMPTY);
  const input = useRef<HTMLInputElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const bar = useRef<HTMLDivElement>(null);
  const current = matches[index];

  const close = () => {
    setOpen(false);
    setResult(EMPTY);
    if (previousFocus.current?.isConnected) previousFocus.current.focus({ preventScroll: true });
  };

  useEffect(() => {
    // Leave the browser's built-in find shortcut alone on older engines.
    if (!CSS.highlights) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.isComposing) return;
      if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "f") {
        e.preventDefault();
        e.stopPropagation();
        if (open) {
          input.current?.focus({ preventScroll: true });
          input.current?.select();
        } else {
          previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
          setOpen(true);
        }
      } else if (open && e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        close();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open]);

  useLayoutEffect(() => {
    if (!open) return;
    input.current?.focus({ preventScroll: true });
    input.current?.select();
  }, [open]);

  useEffect(() => {
    if (!open || !query.trim()) {
      setResult(EMPTY);
      return;
    }
    const first = pageMatches(document.body, query);
    setResult({ matches: first, index: first.length ? 0 : -1 });
    if (first.length) revealPageMatch(first[0]);

    let frame = 0;
    // Streaming replies and expanded panels can change without a navigation.
    const observer = new MutationObserver((records) => {
      if (frame || records.every((record) => bar.current?.contains(record.target))) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        const next = pageMatches(document.body, query);
        setResult((prev) => {
          const active = prev.matches[prev.index];
          const sameRange = (range: Range, other: Range) =>
            range.startContainer === other.startContainer && range.startOffset === other.startOffset &&
            range.endContainer === other.endContainer && range.endOffset === other.endOffset;
          if (next.length === prev.matches.length && next.every((range, i) => sameRange(range, prev.matches[i]))) return prev;
          const same = active ? next.findIndex((range) => sameRange(range, active)) : -1;
          const index = same >= 0 ? same : Math.min(Math.max(prev.index, 0), next.length - 1);
          return { matches: next, index };
        });
      });
    });
    observer.observe(document.body, {
      childList: true, characterData: true, subtree: true, attributes: true,
      attributeFilter: ["class", "style", "hidden", "open"],
    });
    return () => { observer.disconnect(); cancelAnimationFrame(frame); };
  }, [open, query]);

  useLayoutEffect(() => {
    if (!open) return;
    const highlight = new Highlight();
    for (const match of matches) highlight.add(match);
    CSS.highlights.set(MATCH, highlight);
    if (current) {
      const active = new Highlight(current);
      active.priority = 1;
      CSS.highlights.set(ACTIVE, active);
    }
    return () => { CSS.highlights.delete(MATCH); CSS.highlights.delete(ACTIVE); };
  }, [open, matches, current]);

  const move = (direction: number) => {
    if (!matches.length) return;
    const next = (index + direction + matches.length) % matches.length;
    setResult({ matches, index: next });
    revealPageMatch(matches[next]);
  };

  if (!open) return null;
  return (
    <div ref={bar} className="page-find" data-page-find role="search" aria-label={S.pageFind.label}>
      <span className="page-find-icon" aria-hidden="true"><IconSearch /></span>
      <input ref={input} type="text" autoComplete="off" spellCheck={false} placeholder={S.pageFind.label}
             aria-label={S.pageFind.label} value={query} onChange={(e) => setQuery(e.target.value)}
             onKeyDown={(e) => {
               if (e.key !== "Enter" || e.nativeEvent.isComposing) return;
               e.preventDefault();
               move(e.shiftKey ? -1 : 1);
             }} />
      <span className="page-find-count" role="status">{query.trim() && S.pageFind.count(index + 1, matches.length)}</span>
      <button className="page-find-prev" type="button" title={S.pageFind.prev} aria-label={S.pageFind.prev} disabled={!matches.length} onClick={() => move(-1)}><IconChevronDown /></button>
      <button type="button" title={S.pageFind.next} aria-label={S.pageFind.next} disabled={!matches.length} onClick={() => move(1)}><IconChevronDown /></button>
      <button type="button" title={S.common.close} aria-label={S.common.close} onClick={close}><IconClose /></button>
    </div>
  );
}