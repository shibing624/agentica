// ============ TOAST / CONFIRM / DIR MODAL / ACCOUNT / PLUGINS ============
import { state } from './state.js';
import { fmtN, ago, shortenPath } from './utils.js';
import { fetchDirHistory, saveBaseDirApi, openPathApi, getToken, setToken } from './api.js';
import { loadProjectMeta, saveProjectMeta, ensureProjectForSession, projectIdForDir } from './state.js';
import { createSessionWithDir, save } from './sessions.js';
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
    state.dirModal.value = state.serverDir || '';
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
  const { ok, data } = await saveBaseDirApi(val);
  if (!ok) {
    showToast(data?.detail || '设置工作目录失败，请检查路径是否正确');
    return;
  }
  if (data.status === 'ok') {
    state.serverDir = data.base_dir;
    loadDirHistory();
    if (data.created) showToast('已自动创建文件夹: ' + data.base_dir);

    if (state.dirModal.forNewSession) {
      state.dirModal.forNewSession = false;
      state.dirModal.open = false;
      createSessionWithDir(data.base_dir);
      return;
    }

    if (state.curSess && state.sessions[state.curSess]) {
      state.sessions[state.curSess].dir = data.base_dir;
      const meta = loadProjectMeta();
      const projectId = projectIdForDir(data.base_dir);
      meta.sessionMeta[state.curSess] = { ...(meta.sessionMeta[state.curSess] || {}), projectId, archived: !!state.sessions[state.curSess].archived };
      ensureProjectForSession(state.curSess, state.sessions[state.curSess], meta);
      saveProjectMeta(meta);
      save();
      renderSidebar();
    }
  }
  state.dirModal.open = false;
}

export function copyDir() {
  const val = (state.dirModal.value || '').trim();
  const text = val || currentDir();
  if (!text) return;
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => showToast('已复制: ' + text));
  } else {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.cssText = 'position:fixed;left:-9999px';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); showToast('已复制: ' + text); }
    catch (e) { showToast('复制失败'); }
    document.body.removeChild(ta);
  }
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

// ---- Account panel ----
export function openAccountPanel() {
  state.accountPanelOpen = true;
  state.gatewayTokenInput = getToken();
}

export function closeAccountPanel() {
  state.accountPanelOpen = false;
}

export function saveGatewayToken() {
  setToken((state.gatewayTokenInput || '').trim());
  showToast('Access token saved');
}

export function currentUsage() {
  const s = state.curSess && state.sessions[state.curSess] ? state.sessions[state.curSess] : null;
  const ctxWin = state.serverContextWindow || 128000;
  const input = s ? (s.lastInputTokens || s.tokIn || 0) : 0;
  const pct = ctxWin && input ? Math.min(Math.round((input / ctxWin) * 100), 100) : 0;
  const entries = s && s.usageEntries ? s.usageEntries : [];
  return {
    s, ctxWin, input, pct,
    tokIn: s ? s.tokIn || 0 : 0, tokOut: s ? s.tokOut || 0 : 0,
    requests: s ? s.requests || 0 : 0, totalTime: s ? s.totalTime || 0 : 0,
    entries,
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

// ---- Plugins panel ----
export function openPluginsPanel() {
  state.pluginsPanelOpen = true;
}

export function closePluginsPanel() {
  state.pluginsPanelOpen = false;
}
