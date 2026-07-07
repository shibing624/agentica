// ============ SHARED UTILS ============
import { state } from './state.js';

export function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML }
export function ago(ts) { const d = Date.now() - ts; const m = Math.floor(d / 60000); if (m < 1) return 'now'; if (m < 60) return m + 'm'; const h = Math.floor(m / 60); if (h < 24) return h + 'h'; return Math.floor(h / 24) + 'd' }
export function fmtTime(sec) { if (!sec) return '0s'; if (sec < 60) return sec.toFixed(1) + 's'; const m = Math.floor(sec / 60); const s = sec % 60; return m + 'm' + s.toFixed(0) + 's' }

// Compact duration for the "Worked for" step label, mirroring the CLI's
// format_duration_compact: escalates the unit tiers (s -> m/s -> h/m -> d/h)
// so multi-hour/day agent runs still read as a short duration, not a huge
// raw second count.
export function fmtDurationCompact(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  if (sec < 60) return sec + 's';
  let m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return `${m}m${String(s).padStart(2, '0')}s`;
  let h = Math.floor(m / 60); m %= 60;
  if (h < 24) return `${h}h${String(m).padStart(2, '0')}m`;
  const d = Math.floor(h / 24); h %= 24;
  return `${d}d${String(h).padStart(2, '0')}h`;
}
export function fmtN(n) { if (!n) return '0'; if (n >= 1000) return (n / 1000).toFixed(1) + 'K'; return String(n) }
export function fmtFileSize(sz) { return sz > 1024 ? (sz / 1024).toFixed(1) + 'K' : sz + 'B' }

export function uid() {
  if (crypto && crypto.randomUUID) return 'w:' + crypto.randomUUID().replace(/-/g, '').slice(0, 12);
  const a = new Uint8Array(8); crypto.getRandomValues(a);
  return 'w:' + Array.from(a, b => b.toString(16).padStart(2, '0')).join('').slice(0, 12);
}

// Shorten file path for display: keep filename + parent dir
export function shortenFilePath(p) {
  if (!p) return '';
  const parts = p.replace(/\\/g, '/').split('/').filter(Boolean);
  if (parts.length <= 2) return parts.join('/');
  return '…/' + parts.slice(-2).join('/');
}

export function shortenPath(p) {
  const home = p.startsWith('/Users/') ? p.split('/').slice(0, 3).join('/') : null;
  if (home && p.startsWith(home)) {
    const rel = p.slice(home.length);
    if (!rel || rel === '/') return '~';
    const parts = rel.split('/').filter(Boolean);
    if (parts.length <= 2) return '~/' + parts.join('/');
    return '~/…/' + parts.slice(-2).join('/');
  }
  return p;
}

export function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  document.getElementById('sidebar').classList.toggle('collapsed', state.sidebarCollapsed);
  document.getElementById('sidebarExpandBtn').classList.toggle('show', state.sidebarCollapsed);
}

// Collapsed sidebar hides the search <input> (display:none), so a plain
// label-click can't focus it. Expand the sidebar first (if needed), then
// focus — used on the search nav item's click handler.
export function focusSidebarSearch() {
  if (state.sidebarCollapsed) toggleSidebar();
  document.querySelector('.side-search-item input')?.focus();
}

// ---- Sidebar width resize (drag handle on the right edge) ----
const SIDEBAR_W_KEY = 'ag_sidebar_w';
const SIDEBAR_MIN_W = 200;
const SIDEBAR_MAX_W = 420;

export function initSidebarResize() {
  const saved = parseInt(localStorage.getItem(SIDEBAR_W_KEY), 10);
  if (saved && saved >= SIDEBAR_MIN_W && saved <= SIDEBAR_MAX_W) {
    document.documentElement.style.setProperty('--sidebar-w', saved + 'px');
  }
  const handle = document.getElementById('sidebarResizeHandle');
  const sidebar = document.getElementById('sidebar');
  if (!handle || !sidebar) return;

  let startX = 0, startW = 0;
  const onMove = (e) => {
    const w = Math.min(SIDEBAR_MAX_W, Math.max(SIDEBAR_MIN_W, startW + (e.clientX - startX)));
    document.documentElement.style.setProperty('--sidebar-w', w + 'px');
  };
  const onUp = () => {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    handle.classList.remove('dragging');
    sidebar.classList.remove('resizing');
    document.body.style.userSelect = '';
    const w = sidebar.getBoundingClientRect().width;
    localStorage.setItem(SIDEBAR_W_KEY, String(Math.round(w)));
  };
  handle.addEventListener('mousedown', (e) => {
    if (state.sidebarCollapsed) return;
    e.preventDefault();
    startX = e.clientX;
    startW = sidebar.getBoundingClientRect().width;
    handle.classList.add('dragging');
    sidebar.classList.add('resizing');
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

export function highlightCode(root) {
  if (typeof hljs === 'undefined') return;
  (root || document).querySelectorAll('pre code').forEach(el => {
    if (!el.dataset.highlighted) hljs.highlightElement(el);
  });
}
