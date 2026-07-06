// ============ SESSION CRUD (create/switch/rename/archive/fork) ============
import { state, loadProjectMeta, saveProjectMeta, ensureProjectForSession, createForkSessionDraft, persistSessionMeta } from './state.js';
import { nextTick } from './vendor/petite-vue.js';
import { uid } from './utils.js';
import { fetchSessions, deleteSessionApi, archiveSessionApi, unarchiveSessionApi, renameSessionApi } from './api.js';
import { showToast, showConfirm, openDirModalForNewSession, updateDirDisplayForSession } from './modals.js';
import { renderSidebar } from './sidebar.js';
import { renderChat } from './chat.js';

export async function loadSessions() {
  // Server SessionLog is the single source of truth for session list/name.
  // Local localStorage keeps msgs/dir/tokens for rendering (a later phase
  // will move message history to the server too). Merge: server wins for name/existence.
  let local = {};
  try { local = JSON.parse(localStorage.getItem('ag_s') || '{}') } catch { local = {} }
  let serverList = [];
  const { ok, data } = await fetchSessions();
  if (ok && data) serverList = data.sessions || [];
  const meta = loadProjectMeta();
  const merged = {};
  for (const sv of serverList) {
    const id = sv.session_id;
    const lc = local[id] || {};
    merged[id] = ensureProjectForSession(id, {
      title: sv.name || lc.title || 'Chat',
      msgs: lc.msgs || [],
      ts: sv.last_timestamp ? (new Date(sv.last_timestamp).getTime()) : (lc.ts || Date.now()),
      tokIn: lc.tokIn || 0, tokOut: lc.tokOut || 0, tokTotal: lc.tokTotal || 0, requests: lc.requests || 0, totalTime: lc.totalTime || 0, usageEntries: lc.usageEntries || [], lastInputTokens: lc.lastInputTokens || 0,
      dir: lc.dir || state.serverDir || '',
      projectId: lc.projectId,
      archived: !!(sv.archived || lc.archived),
      user_count: sv.user_count || 0,
    }, meta);
  }
  // keep local-only sessions (new sessions not yet persisted to SessionLog)
  for (const id in local) { if (!merged[id]) merged[id] = ensureProjectForSession(id, local[id], meta); }
  state.sessions = merged;
  saveProjectMeta(meta);
  save();
  renderSidebar();
  const last = localStorage.getItem('ag_a');
  if (last && state.sessions[last]) switchTo(last);
}

export function save() { localStorage.setItem('ag_s', JSON.stringify(state.sessions)) }

export function newSession() {
  if (state.streaming) return;
  openDirModalForNewSession();
}

export function createSessionWithDir(dir) {
  const id = uid();
  const meta = loadProjectMeta();
  state.sessions[id] = ensureProjectForSession(id, { title: 'New Chat', msgs: [], ts: Date.now(), tokIn: 0, tokOut: 0, tokTotal: 0, requests: 0, totalTime: 0, usageEntries: [], lastInputTokens: 0, dir: dir || state.serverDir || '' }, meta);
  saveProjectMeta(meta);
  save(); switchTo(id); renderSidebar();
  document.getElementById('inputTa').focus();
}

export function switchTo(id) {
  state.curSess = id;
  localStorage.setItem('ag_a', id);
  const s = state.sessions[id];
  updateDirDisplayForSession(s);
  renderSidebar(); renderChat();
}

export function delSession(id) {
  const title = state.sessions[id]?.title || '该会话';
  showConfirm(`删除会话「${title}」？`, () => {
    const ids = Object.keys(state.sessions).sort((a, b) => state.sessions[b].ts - state.sessions[a].ts);
    const idx = ids.indexOf(id);
    const nextId = ids[idx + 1] || ids[idx - 1] || null;
    delete state.sessions[id]; save();
    deleteSessionApi(id).then(({ ok }) => { if (!ok) throw 0 }).catch(() => showToast('服务端删除同步失败', 2000));
    if (state.curSess === id && nextId && state.sessions[nextId]) switchTo(nextId);
    else if (state.curSess === id) { state.curSess = null; localStorage.removeItem('ag_a'); renderChat() }
    renderSidebar();
  });
}

export function archiveSession(id) {
  if (!state.sessions[id]) return;
  persistSessionMeta(id, { archived: true });
  save();
  archiveSessionApi(id)
    .then(({ ok }) => { if (!ok) throw 0 })
    .catch(() => showToast('归档已本地保存，服务端同步失败', 2200));
  if (state.curSess === id) {
    const ids = Object.keys(state.sessions).filter(sid => sid !== id && !state.sessions[sid].archived).sort((a, b) => state.sessions[b].ts - state.sessions[a].ts);
    if (ids.length) switchTo(ids[0]);
    else { state.curSess = null; localStorage.removeItem('ag_a'); renderChat(); }
  }
  renderSidebar();
  showToast('已归档', 1200);
}

export function unarchiveSession(id) {
  if (!state.sessions[id]) return;
  persistSessionMeta(id, { archived: false });
  unarchiveSessionApi(id)
    .then(({ ok }) => { if (!ok) throw 0 })
    .catch(() => showToast('恢复已本地保存，服务端同步失败', 2200));
  save(); renderSidebar();
  showToast('已恢复', 1200);
}

export function forkSession(sessionId, fromMsgIdx) {
  const source = state.sessions[sessionId]; if (!source) return null;
  const meta = loadProjectMeta();
  ensureProjectForSession(sessionId, source, meta);
  const id = uid();
  const draft = createForkSessionDraft(source, fromMsgIdx);
  draft.parentSessionId = sessionId;
  state.sessions[id] = ensureProjectForSession(id, draft, meta);
  meta.sessionMeta[id] = { ...(meta.sessionMeta[id] || {}), parentSessionId: sessionId, forkedFromMsgIdx: fromMsgIdx, archived: false };
  saveProjectMeta(meta);
  save(); switchTo(id); renderSidebar();
  showToast('已创建分叉会话', 1200);
  return id;
}

export async function renameSession(id) {
  const s = state.sessions[id]; if (!s) return;
  state.renamingSessionId = id;
  nextTick(() => {
    const inp = document.getElementById(`rename_${id}`);
    if (inp) { inp.focus(); inp.select(); }
  });
}

export function cancelSessionRename() {
  state.renamingSessionId = null;
}

export async function commitSessionRename(id, value) {
  if (state.renamingSessionId !== id) return;
  const s = state.sessions[id]; if (!s) return;
  const t = (value || '').trim();
  state.renamingSessionId = null;
  if (!t) { showToast('会话名不能为空', 1500); return; }
  if (t === s.title) return;
  s.title = t; save();
  const { ok } = await renameSessionApi(id, t);
  if (!ok) { showToast('重命名同步到服务端失败', 2000); return; }
  showToast('已重命名', 1200);
}

export function renameKey(ev, id) {
  if (ev.key === 'Enter') commitSessionRename(id, ev.target.value);
  else if (ev.key === 'Escape') cancelSessionRename();
}
