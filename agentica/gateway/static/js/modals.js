// ============ TOAST / CONFIRM / DIR MODAL / ACCOUNT / PLUGINS ============
import { state } from './state.js';
import { fmtN, ago, shortenPath } from './utils.js';
import { fetchDirHistory, saveBaseDirApi, openPathApi } from './api.js';
import { loadProjectMeta, saveProjectMeta, ensureProjectForSession, activeProjectIdForDir } from './state.js';
import { startNewChatDraft, save, delSession } from './sessions.js';
import { renderSidebar } from './sidebar.js';

// ---- Toast ----
let toastTimer = null;
export function showToast(msg, duration = 2500) {
  state.toast.msg = msg;
  state.toast.show = true;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { state.toast.show = false }, duration);
}

// ---- Confirm modal ----
let _confirmCb = null;
export function showConfirm(msg, cb) {
  _confirmCb = cb;
  state.confirm.msg = msg;
  state.confirm.open = true;
}
export function confirmOk() {
  state.confirm.open = false;
  if (_confirmCb) { _confirmCb(); _confirmCb = null }
}
export function confirmCancel() {
  state.confirm.open = false;
  _confirmCb = null;
}

// ---- Working directory modal ----
export function updateDirDisplay() {
  const dir = currentDir();
  if (dir) {
    const p = dir.split('/').filter(Boolean);
    const short = p.length > 3 ? '…/' + p.slice(-2).join('/') : dir;
    document.getElementById('dirVal').textContent = short;
    document.getElementById('dirWrap').title = 'Working Directory: ' + dir + ' (click to edit)';
  }
}

export function updateDirDisplayForSession(s) {
  const dir = (s && s.dir) ? s.dir : state.serverDir;
  if (dir) {
    const p = dir.split('/').filter(Boolean);
    const short = p.length > 3 ? '…/' + p.slice(-2).join('/') : dir;
    document.getElementById('dirVal').textContent = short;
    document.getElementById('dirWrap').title = 'Working Directory: ' + dir + ' (click to edit)';
  }
}

export function currentDir() {
  return state.curSess && state.sessions[state.curSess] ? state.sessions[state.curSess].dir : state.serverDir;
}

export function dirHistoryFiltered() {
  const q = (state.dirModal.value || '').toLowerCase();
  return q ? state.dirModal.historyList.filter(p => p.toLowerCase().includes(q)) : state.dirModal.historyList;
}

export function openDirModal() {
  state.dirModal.open = true;
  state.dirModal.historyOpen = false;
  if (state.dirModal.forNewSession) {
    state.dirModal.value = state.pendingNewChatDir || state.serverDir || '';
  } else {
    state.dirModal.value = currentDir() || '';
  }
  loadDirHistory();
  setTimeout(() => document.getElementById('dirEditInput')?.focus(), 50);
}

export function closeDirModal() {
  state.dirModal.forNewSession = false;
  state.dirModal.historyOpen = false;
  state.dirModal.open = false;
}

async function loadDirHistory() {
  const { ok, data } = await fetchDirHistory();
  state.dirModal.historyList = ok && data ? (data.history || []) : [];
}

export function toggleDirHistory() {
  state.dirModal.historyOpen = !state.dirModal.historyOpen;
}

export function selectDirHistory(path) {
  state.dirModal.value = path;
  state.dirModal.historyOpen = false;
  document.getElementById('dirEditInput')?.focus();
}

export function openDirModalForNewSession() {
  state.dirModal.forNewSession = true;
  openDirModal();
}

export async function saveDir() {
  const val = (state.dirModal.value || '').trim();
  if (!val) return;
  // The server only validates that the dir already exists (it never creates
  // one on the user's behalf) — a clear, actionable error surfaces here.
  const { ok, data } = await saveBaseDirApi(val);
  if (!ok) {
    showToast(data?.detail || 'Directory does not exist. Create it first, then try again.', 3500);
    return;
  }
  if (data.status === 'ok') {
    state.serverDir = data.base_dir;
    loadDirHistory();

    // A new/different dir from the current project's is how a new project
    // gets created — ensureProjectForSession keys projects 1:1 by dir.
    if (state.dirModal.forNewSession) {
      state.dirModal.forNewSession = false;
      state.dirModal.open = false;
      // Only remember the chosen dir — the session itself isn't created
      // until the user actually sends a first message (startNewChatDraft).
      startNewChatDraft(data.base_dir);
      return;
    }

    if (state.curSess && state.sessions[state.curSess]) {
      state.sessions[state.curSess].dir = data.base_dir;
      const meta = loadProjectMeta();
      const projectId = activeProjectIdForDir(meta, data.base_dir);
      meta.sessionMeta[state.curSess] = { ...(meta.sessionMeta[state.curSess] || {}), projectId, archived: !!state.sessions[state.curSess].archived };
      ensureProjectForSession(state.curSess, state.sessions[state.curSess], meta);
      saveProjectMeta(meta);
      save();
      renderSidebar();
    }
  }
  state.dirModal.open = false;
}

function copyText(text) {
  if (!text) return;
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => showToast('Copied: ' + text));
  } else {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.cssText = 'position:fixed;left:-9999px';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); showToast('Copied: ' + text); }
    catch (e) { showToast('Copy failed'); }
    document.body.removeChild(ta);
  }
}

export function copyDir() {
  const val = (state.dirModal.value || '').trim();
  copyText(val || currentDir());
}

// Topbar "copy path" button — always copies the *actual* current session's
// working directory, ignoring any in-progress (unsaved) dir-modal edit.
export function copyCurrentDir() {
  copyText(currentDir());
}

export function openInFinder() {
  const dir = currentDir();
  if (!dir) return;
  openPathApi(dir, 'finder').catch(() => { window.open('file://' + dir, '_blank'); });
}

export function openInTerminal() {
  const dir = currentDir();
  if (!dir) return;
  openPathApi(dir, 'terminal').catch(() => { });
}

// ---- Account panel (small popover anchored above the account button) ----
export function openAccountPanel() {
  state.accountPanelOpen = true;
}

export function closeAccountPanel() {
  state.accountPanelOpen = false;
}

// ---- Archived sessions (rendered inside the Settings modal's "Archived"
// tab — see settings-panel.js) ----
export function deleteArchivedSession(id) {
  delSession(id);
}

// Account panel usage is user-wide, not tied to whichever session happens to
// be open — it sums every local session's counters (the per-session tally in
// the input box's ctx-tip is the one scoped to state.curSess).
export function currentUsage() {
  const sessions = Object.values(state.sessions);
  let tokIn = 0, tokOut = 0, requests = 0, totalTime = 0;
  for (const s of sessions) {
    tokIn += s.tokIn || 0;
    tokOut += s.tokOut || 0;
    requests += s.requests || 0;
    totalTime += s.totalTime || 0;
  }
  return {
    sessionCount: sessions.length,
    tokIn, tokOut, tokTotal: tokIn + tokOut,
    requests, totalTime,
  };
}

export function archivedSessions() {
  return Object.keys(state.sessions)
    .filter(id => state.sessions[id] && state.sessions[id].archived)
    .sort((a, b) => (state.sessions[b].ts || 0) - (state.sessions[a].ts || 0))
    .map(id => {
      const s = state.sessions[id];
      return {
        id, title: s.title || 'Chat',
        msgCount: s.msgs ? s.msgs.filter(m => m.role === 'user').length : 0,
        dir: s.dir ? shortenPath(s.dir) : '',
        agoStr: ago(s.ts || Date.now()),
      };
    });
}

export { fmtN };
