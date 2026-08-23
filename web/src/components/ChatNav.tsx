import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { useStrings } from "../i18n";
import {
  CHAT_NAV_MIN_TICKS,
  activeTickIdx,
  buildChatNavTicks,
  ensureRailTickVisible,
  scrollChatToMessage,
  type ChatNavTick,
} from "../lib/chatNav";
import type { ChatMsg } from "../store";

/**
 * Codex-style chat navigation minimap:
 * - Positioned in the left gutter of the chat area.
 * - Shows only subtle dashes by default.
 * - Hovering a dash opens a compact opaque popover to its right with the truncated query.
 * - Clicking jumps directly to that turn with a smooth scroll & subtle flash.
 * - Naturally hidden on narrower screens to avoid overlapping main conversation.
 */
export function ChatNav({
  msgs,
  areaRef,
  onLeaveFollow,
}: {
  msgs: ChatMsg[];
  areaRef: RefObject<HTMLDivElement | null>;
  onLeaveFollow: () => void;
}) {
  const S = useStrings();
  const ticks = buildChatNavTicks(msgs);
  const tickKey = ticks.map((t) => t.idx).join(",");
  const ticksRef = useRef<ChatNavTick[]>(ticks);
  ticksRef.current = ticks;
  const leaveRef = useRef(onLeaveFollow);
  leaveRef.current = onLeaveFollow;
  const listRef = useRef<HTMLElement>(null);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);

  const syncActive = () => {
    const area = areaRef.current;
    if (!area) return;
    setActiveIdx(activeTickIdx(ticksRef.current, area));
  };

  const jumpTo = (idx: number) => {
    const area = areaRef.current;
    if (!area) return;
    leaveRef.current();
    scrollChatToMessage(area, idx);
    syncActive();
  };

  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    syncActive();
    area.addEventListener("scroll", syncActive, { passive: true });
    return () => area.removeEventListener("scroll", syncActive);
  }, [areaRef, tickKey]);

  useEffect(() => {
    const nav = listRef.current;
    const area = areaRef.current;
    if (!nav || !area) return;
    const onWheel = (e: WheelEvent) => {
      if (nav.scrollHeight > nav.clientHeight + 1) {
        const atTop = nav.scrollTop <= 0 && e.deltaY < 0;
        const atBottom = nav.scrollTop + nav.clientHeight >= nav.scrollHeight - 1 && e.deltaY > 0;
        if (!atTop && !atBottom) return;
      }
      e.preventDefault();
      leaveRef.current();
      area.scrollTop += e.deltaY;
      syncActive();
    };
    nav.addEventListener("wheel", onWheel, { passive: false });
    return () => nav.removeEventListener("wheel", onWheel);
  }, [areaRef, tickKey]);

  useLayoutEffect(() => {
    const list = listRef.current;
    if (list == null || activeIdx == null) return;
    ensureRailTickVisible(list, activeIdx);
  }, [activeIdx]);

  if (ticks.length < CHAT_NAV_MIN_TICKS) return null;

  return (
    <nav ref={listRef} className="chat-nav" aria-label={S.chat.navAria}>
      <div className="chat-nav-list">
        {ticks.map((t) => (
          <button
            key={t.idx}
            type="button"
            className="chat-nav-tick"
            data-idx={t.idx}
            aria-current={t.idx === activeIdx ? "true" : undefined}
            onClick={(e) => {
              e.preventDefault();
              jumpTo(t.idx);
            }}
          >
            <span className="chat-nav-dash" />
            <span className="chat-nav-label"><span className="chat-nav-label-text">{t.query || S.chat.navEmpty}</span></span>
          </button>
        ))}
      </div>
    </nav>
  );
}
