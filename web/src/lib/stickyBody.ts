/**
 * Bottom-follow for a capped inner scroll area (an expanded work group), with
 * the deepseek-harness `useProcessScroll` semantics: content growth sticks to
 * the bottom while the reader has not scrolled away, an upward wheel / touch
 * drag leaves the tail, and returning within 80px of the bottom resumes.
 * Events are not consumed here: the outer transcript's own follow rule sees the
 * same gesture, so reading inside a group also releases the conversation tail
 * (dsh reads intent from the scroller, not from the row it is over).
 */
import { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { RefObject, TouchEvent, WheelEvent } from "react";
import { createStreamFollow, stickToBottom, type StreamFollow } from "./streamFollow";

export type BodyEdges = { up: boolean; down: boolean };

export type StickyBody = {
  /** The capped scrolling element. */
  bodyRef: RefObject<HTMLDivElement | null>;
  /** The uncapped content inside it, observed for growth. */
  contentRef: RefObject<HTMLDivElement | null>;
  /** Whether more content is clipped above / below (drives the edge fades). */
  edges: BodyEdges;
  handlers: {
    onScroll: () => void;
    onWheel: (e: WheelEvent<HTMLDivElement>) => void;
    onTouchStart: (e: TouchEvent<HTMLDivElement>) => void;
    onTouchMove: (e: TouchEvent<HTMLDivElement>) => void;
    onTouchEnd: () => void;
  };
};

const AT_REST: BodyEdges = { up: false, down: false };

export function useStickyBody(open: boolean, followTail: boolean): StickyBody {
  const bodyRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const followRef = useRef<StreamFollow | null>(null);
  followRef.current ??= createStreamFollow();
  const follow = followRef.current;
  const positionedRef = useRef(false);
  const [edges, setEdges] = useState<BodyEdges>(AT_REST);

  const sync = useCallback(() => {
    const el = bodyRef.current;
    if (!el) return;
    const up = el.scrollTop > 1;
    const down = el.scrollTop < el.scrollHeight - el.clientHeight - 1;
    setEdges((prev) => (prev.up === up && prev.down === down ? prev : { up, down }));
  }, []);

  // First frame of an opened body: a running group lands on its tail, a settled
  // one starts at its first row. Later followTail flips (the turn finishing)
  // never move the reader.
  useLayoutEffect(() => {
    const el = bodyRef.current;
    if (!open) {
      positionedRef.current = false;
      return;
    }
    if (!el || positionedRef.current) return;
    positionedRef.current = true;
    if (followTail) {
      stickToBottom(el, follow);
    } else {
      follow.leave();
      el.scrollTop = 0;
    }
    sync();
  }, [open, followTail, follow, sync]);

  // Growth follows the tail while the reader is still there.
  useLayoutEffect(() => {
    const el = bodyRef.current;
    const content = contentRef.current;
    if (!open || !el || !content || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (follow.stick) stickToBottom(el, follow);
      sync();
    });
    ro.observe(content);
    ro.observe(el);
    return () => ro.disconnect();
  }, [open, follow, sync]);

  return {
    bodyRef,
    contentRef,
    edges,
    handlers: {
      onScroll: () => {
        const el = bodyRef.current;
        if (!el) return;
        follow.scrolled({
          scrollTop: el.scrollTop,
          scrollHeight: el.scrollHeight,
          clientHeight: el.clientHeight,
        });
        sync();
      },
      onWheel: (e) => follow.wheel(e.deltaY),
      onTouchStart: (e) => follow.touchStart(e.touches[0]?.clientY ?? 0),
      onTouchMove: (e) => follow.touchMove(e.touches[0]?.clientY ?? 0),
      onTouchEnd: () => follow.touchEnd(),
    },
  };
}
