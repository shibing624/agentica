// ============ CHAT NAVIGATOR (Codex-style minimap for jumping between turns) ============
// A thin vertical tick stack on the left edge of the chat area (the right
// edge is reserved for the native scrollbar), one tick per user message.
// Ticks are packed tightly together (centered flex stack) rather than
// spread proportionally across the scroll height, so jumping between many
// turns stays a small mouse move. Hovering a tick previews the query + the
// first reply that follows it; clicking jumps there.
import { state } from './state.js';
import { esc } from './utils.js';

// Cached from the last render — reused by updateChatNavActive() (called on
// every scroll tick) so it never has to touch the DOM/session data.
let ticks = [];
let rafPending = false;

export function renderChatNav() {
  const nav = document.getElementById('chatNav');
  if (!nav) return;
  const s = state.curSess ? state.sessions[state.curSess] : null;
  const container = document.getElementById('messages');
  const msgs = s ? s.msgs : [];
  const userIdxs = [];
  for (let i = 0; i < msgs.length; i++) if (msgs[i].role === 'user') userIdxs.push(i);

  // Not worth showing a navigator for a one-turn conversation.
  if (!container || userIdxs.length < 2) {
    ticks = [];
    nav.classList.remove('visible');
    nav.innerHTML = '';
    return;
  }

  ticks = userIdxs.map((idx) => {
    const el = document.getElementById('msg-' + idx);
    const offsetTop = el ? el.offsetTop : 0;
    let reply = '';
    for (let j = idx + 1; j < msgs.length; j++) {
      if (msgs[j].role === 'assistant') { reply = (msgs[j].content || '').trim().replace(/\s+/g, ' ').slice(0, 120); break }
    }
    return { idx, offsetTop, query: (msgs[idx].content || '').trim().replace(/\s+/g, ' ').slice(0, 80), reply };
  });

  // Ticks are plain flex children (no per-tick top offset) — the stack is
  // centered as a group by the `.chat-nav` flex container, kept dense
  // regardless of how tall the underlying messages actually render.
  nav.innerHTML = ticks.map((t) => `
    <div class="chat-nav-tick" data-idx="${t.idx}" onclick="jumpToChatMsg(${t.idx})">
      <span class="chat-nav-dash"></span>
      <div class="chat-nav-tip">
        <div class="chat-nav-tip-q">${esc(t.query) || '(empty)'}</div>
        ${t.reply ? `<div class="chat-nav-tip-a">${esc(t.reply)}</div>` : ''}
      </div>
    </div>`).join('');
  nav.classList.add('visible');
  updateChatNavActive();
}

// Cheap, DOM-class-only update — called on every #chatArea scroll event, so
// it must not re-measure or rebuild HTML.
export function updateChatNavActive() {
  if (!ticks.length) return;
  const nav = document.getElementById('chatNav');
  const area = document.getElementById('chatArea');
  if (!nav || !area) return;
  const scrollTop = area.scrollTop;
  let activeIdx = ticks[0].idx;
  for (const t of ticks) {
    if (t.offsetTop <= scrollTop + 48) activeIdx = t.idx;
    else break;
  }
  nav.querySelectorAll('.chat-nav-tick').forEach((el) => {
    el.classList.toggle('active', Number(el.dataset.idx) === activeIdx);
  });
}

// Streaming content keeps growing the doc height, which shifts each
// message's offsetTop (used by updateChatNavActive's scrollspy) — throttle
// re-renders to at most once per frame instead of on every SSE chunk.
export function scheduleRenderChatNav() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; renderChatNav() });
}

export function jumpToChatMsg(idx) {
  const el = document.getElementById('msg-' + idx);
  const area = document.getElementById('chatArea');
  if (!el || !area) return;
  area.scrollTo({ top: el.offsetTop - 12, behavior: 'smooth' });
}
