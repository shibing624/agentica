import type { ChatMsg } from "../store";

/** Show the rail once there are at least two user turns to jump between. */
export const CHAT_NAV_MIN_TICKS = 2;

export type ChatNavTick = {
  idx: number;
  query: string;
};

export function previewText(text: string, n = 50): string {
  return text.trim().replace(/\s+/g, " ").slice(0, n);
}

/** One tick per real user query (steer chips are mid-run interrupts, not turns). */
export function buildChatNavTicks(msgs: ChatMsg[]): ChatNavTick[] {
  const ticks: ChatNavTick[] = [];
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    if (m.role !== "user" || m.steer) continue;
    ticks.push({
      idx: i,
      query: previewText(m.content || ""),
    });
  }
  return ticks;
}

export function offsetIn(el: HTMLElement, container: HTMLElement): number {
  return el.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
}

export function scrollChatToMessage(area: HTMLElement, idx: number): boolean {
  const el = area.querySelector("#msg-" + idx) ?? document.getElementById("msg-" + idx);
  if (!(el instanceof HTMLElement)) return false;
  try {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch {
    area.scrollTop = Math.max(0, offsetIn(el, area) - 12);
  }
  flashUserTurn(el);
  return true;
}

export function flashUserTurn(el: HTMLElement) {
  el.classList.remove("chat-nav-flash");
  void el.offsetWidth;
  el.classList.add("chat-nav-flash");
  const done = () => el.classList.remove("chat-nav-flash");
  el.addEventListener("animationend", done, { once: true });
}

export function activeTickIdx(ticks: ChatNavTick[], area: HTMLElement): number | null {
  if (!ticks.length) return null;
  let active = ticks[0].idx;
  for (const t of ticks) {
    const el = area.querySelector("#msg-" + t.idx) ?? document.getElementById("msg-" + t.idx);
    if (!(el instanceof HTMLElement)) continue;
    if (offsetIn(el, area) <= area.scrollTop + 64) active = t.idx;
    else break;
  }
  return active;
}

export function ensureRailTickVisible(list: HTMLElement, idx: number) {
  const n = list.querySelector(`[data-idx="${idx}"]`);
  if (!(n instanceof HTMLElement)) return;
  if (n.offsetTop < list.scrollTop) list.scrollTop = n.offsetTop;
  else if (n.offsetTop + n.offsetHeight > list.scrollTop + list.clientHeight) {
    list.scrollTop = n.offsetTop + n.offsetHeight - list.clientHeight + 1;
  }
}
