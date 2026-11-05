import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { useStrings } from "../i18n";
import { IconDatabase } from "../icons";
import { fmtN } from "../lib/format";
import { placeAbove, turnUsageRows } from "../lib/turnUsage";
import type { ChatMsg } from "../store";

/** Laid out but invisible, so the first pass can measure the panel. */
const MEASURE: CSSProperties = { visibility: "hidden", left: 0, top: 0 };

/**
 * The reply footer's token pill. Click opens this turn's accounting (as in
 * deepseek-harness's TurnUsagePanel); a hover tooltip could not be read
 * carefully, copied from, or reached on touch.
 */
export function TurnUsagePanel({ m, onOpenChange }: { m: ChatMsg; onOpenChange?: (open: boolean) => void }) {
  const S = useStrings();
  const [open, setOpenState] = useState(false);
  const [pos, setPos] = useState<CSSProperties | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);

  const setOpen = (next: boolean) => {
    setOpenState(next);
    if (!next) setPos(null);
    onOpenChange?.(next);
  };

  const place = () => {
    const a = triggerRef.current?.getBoundingClientRect();
    const p = panelRef.current?.getBoundingClientRect();
    if (!a || !p) return;
    const { left, top } = placeAbove(a, p, { width: window.innerWidth, height: window.innerHeight });
    setPos({ left, top });
  };

  useLayoutEffect(() => {
    if (open) place();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (panelRef.current?.contains(t) || triggerRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setOpen(false); triggerRef.current?.focus(); }
    };
    // The chat scrolls under a fixed panel; follow the trigger instead of
    // leaving the panel hanging over another reply.
    const onMove = () => place();
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onMove);
    window.addEventListener("scroll", onMove, true);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onMove);
      window.removeEventListener("scroll", onMove, true);
    };
  }, [open]);

  useEffect(() => () => onOpenChange?.(false), []);

  const { total, rows } = turnUsageRows(m, S.chat);
  const compact = fmtN((m.tokIn || 0) + (m.tokOut || 0));

  return (
    <>
      <button type="button" ref={triggerRef} className={"msg-usage" + (open ? " open" : "")}
              aria-haspopup="dialog" aria-expanded={open} title={open ? undefined : S.chat.turnUsageOpen}
              onClick={() => setOpen(!open)}>
        <IconDatabase />
        <span>{S.chat.usageTokens(compact)}</span>
      </button>
      {open && createPortal(
        <div ref={panelRef} className="turn-usage" role="dialog" aria-label={S.chat.turnUsageTitle}
             style={pos ?? MEASURE}>
          <div className="turn-usage-head">
            <span className="turn-usage-title"><IconDatabase />{S.chat.turnUsageTitle}</span>
            <span className="turn-usage-total">{total} tok</span>
          </div>
          <div className="turn-usage-rule" aria-hidden />
          <dl className="turn-usage-rows">
            {rows.map((r) => (
              <div key={r.label} className="turn-usage-row">
                <dt>{r.label}</dt>
                <dd className={r.mono ? "mono" : undefined}>{r.value}</dd>
              </div>
            ))}
          </dl>
          <div className="turn-usage-note">{S.chat.usageTip}</div>
        </div>,
        document.body,
      )}
    </>
  );
}
