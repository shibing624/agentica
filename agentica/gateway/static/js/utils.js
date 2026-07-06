// ============ SHARED UTILS ============
export function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML }
export function ago(ts) { const d = Date.now() - ts; const m = Math.floor(d / 60000); if (m < 1) return 'now'; if (m < 60) return m + 'm'; const h = Math.floor(m / 60); if (h < 24) return h + 'h'; return Math.floor(h / 24) + 'd' }
export function fmtTime(sec) { if (!sec) return '0s'; if (sec < 60) return sec.toFixed(1) + 's'; const m = Math.floor(sec / 60); const s = sec % 60; return m + 'm' + s.toFixed(0) + 's' }
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

export function toggleSidebar() { document.getElementById('sidebar').classList.toggle('collapsed') }

export function highlightCode(root) {
  if (typeof hljs === 'undefined') return;
  (root || document).querySelectorAll('pre code').forEach(el => {
    if (!el.dataset.highlighted) hljs.highlightElement(el);
  });
}
