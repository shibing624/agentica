/** Stick-to-bottom for the chat stream. Ported from the penguin-harness
 *  follow rule (issue #75 there): exit and resume are independent.
 *
 *  Exit: any upward intent, immediately — wheel-up / touch-drag-down even
 *  when the position cannot change (short viewport already at the top),
 *  and a scrollTop regression for scrollbar / keyboard. Not gated on an
 *  "80px from bottom" threshold, or a short area could never unstick.
 *
 *  Resume: only when the user brings the viewport back within 80px of
 *  the bottom. Content shrinking that clamps scrollTop while still
 *  touching the bottom (≤1px) is not an upward scroll.
 *
 *  Programmatic snaps go through stickToBottom, which reports the landed
 *  position synchronously so a delayed scroll event cannot be misread as
 *  the user parking at a historical position. */

export type ScrollMetrics = {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
};

export type StreamFollow = {
  readonly stick: boolean;
  wheel(deltaY: number): void;
  touchStart(clientY: number): void;
  touchMove(clientY: number): void;
  touchEnd(): void;
  scrolled(m: ScrollMetrics): void;
  resume(): void;
  /** Park at a historical position (chat-nav jump), so layout stick-to-bottom
   *  does not snap the viewport back to the live tail. */
  leave(): void;
};

export function createStreamFollow(): StreamFollow {
  let stick = true;
  let lastTop: number | null = null;
  let touchY: number | null = null;
  return {
    get stick() { return stick; },
    wheel(deltaY) {
      if (deltaY < 0) stick = false;
    },
    touchStart(clientY) { touchY = clientY; },
    touchMove(clientY) {
      if (touchY !== null && clientY > touchY) stick = false;
      touchY = clientY;
    },
    touchEnd() { touchY = null; },
    scrolled(m) {
      const dist = m.scrollHeight - m.scrollTop - m.clientHeight;
      const prev = lastTop;
      lastTop = m.scrollTop;
      if (prev === null) {
        stick = dist < 80;
        return;
      }
      if (m.scrollTop < prev && dist > 1) {
        stick = false;
        return;
      }
      if (dist < 80) stick = true;
    },
    resume() {
      stick = true;
      lastTop = null;
    },
    leave() { stick = false; },
  };
}

export type ScrollContainer = {
  scrollTop: number;
  readonly scrollHeight: number;
  readonly clientHeight: number;
};

export function stickToBottom(el: ScrollContainer, follow: StreamFollow): void {
  el.scrollTop = el.scrollHeight;
  follow.scrolled({
    scrollTop: el.scrollTop,
    scrollHeight: el.scrollHeight,
    clientHeight: el.clientHeight,
  });
}
