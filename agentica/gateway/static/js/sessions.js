// ============ SESSION CRUD (create/switch/rename/archive/fork) ============
import { state, loadProjectMeta, saveProjectMeta, ensureProjectForSession, createForkSessionDraft, persistSessionMeta } from './state.js';
import { nextTick } from './vendor/petite-vue.js';
import { uid } from './utils.js';
import { fetchSessions, deleteSessionApi, archiveSessionApi, unarchiveSessionApi, renameSessionApi } from './api.js';
import { showToast, showConfirm, openDirModalForNewSession, updateDirDisplayForSession } from './modals.js';
import { renderSidebar } from './sidebar.js';
import { renderChat, drainQueue } from './chat.js';

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

// "+ New Chat" (sidebar-level) — reuse the currently viewed session's (or
// pending draft's) project directory by default, so it doesn't interrupt the
// user with a dir prompt every time. Only when there's no current project
// (no session yet, or the active session is Unfiled) do we fall back to
// asking for a directory. Nothing is persisted here — see startNewChatDraft.
export function newSession() {
  const cur = state.curSess ? state.sessions[state.curSess] : null;
  const dir = (cur && cur.dir) || state.pendingNewChatDir || '';
  if (dir) { startNewChatDraft(dir); return; }
  openDirModalForNewSession();
}

// Switch to the "composing a new chat" view without creating/persisting a
// session yet. The session is only actually created (and shown in the
// sidebar) once the user sends the first message — see sendMessage() in
// chat.js — so opening "+ New Chat" and not typing anything leaves no trace.
export function startNewChatDraft(dir) {
  state.curSess = null;
  localStorage.removeItem('ag_a');
  state.pendingNewChatDir = dir || '';
  state.inputText = '';
  state.pendingFiles = [];
  renderSidebar(); renderChat();
  const ta = document.getElementById('inputTa');
  if (ta) { ta.style.height = 'auto'; ta.focus() }
}

// Best-effort default working directory for a brand-new session started
// from the empty/welcome screen: reuse whatever directory was used most
// recently, falling back to the server's base directory.
export function defaultDirForNewSession() {
  const ids = Object.keys(state.sessions).sort((a, b) => (state.sessions[b].ts || 0) - (state.sessions[a].ts || 0));
  for (const id of ids) {
    const d = state.sessions[id].dir;
    if (d) return d;
  }
  return state.serverDir || '';
}

export function createSessionWithDir(dir) {
  const id = uid();
  const meta = loadProjectMeta();
  state.sessions[id] = ensureProjectForSession(id, { title: 'New Chat', msgs: [], ts: Date.now(), tokIn: 0, tokOut: 0, tokTotal: 0, requests: 0, totalTime: 0, usageEntries: [], lastInputTokens: 0, dir: dir || state.serverDir || '' }, meta);
  saveProjectMeta(meta);
  save(); switchTo(id); renderSidebar();
  document.getElementById('inputTa').focus();
}

// "New chat" from inside an existing project (the per-project "+" button) —
// reuse that project's working dir directly instead of asking again, since
// every session in the project already shares it. Only the project-less
// "Unfiled" bucket (no dir) still needs to prompt.
export function createSessionInProject(projectId) {
  const meta = loadProjectMeta();
  const project = meta.projects[projectId];
  const dir = project && project.dir ? project.dir : '';
  if (!dir) { openDirModalForNewSession(); return; }
  startNewChatDraft(dir);
}

export function switchTo(id) {
  state.curSess = id;
  localStorage.setItem('ag_a', id);
  const s = state.sessions[id];
  if (s) s.unread = false;
  updateDirDisplayForSession(s);
  renderSidebar(); renderChat();
  // If a message was queued for this session while the user was elsewhere
  // and the agent is free, send it now instead of waiting for another event.
  drainQueue();
}

export function delSession(id) {
  const title = state.sessions[id]?.title || 'this session';
  showConfirm(`Delete session "${title}"?`, () => {
    const ids = Object.keys(state.sessions).sort((a, b) => state.sessions[b].ts - state.sessions[a].ts);
    const idx = ids.indexOf(id);
    const nextId = ids[idx + 1] || ids[idx - 1] || null;
    delete state.sessions[id]; save();
    deleteSessionApi(id).then(({ ok }) => { if (!ok) throw 0 }).catch(() => showToast('Failed to sync deletion to server', 2000));
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
    .catch(() => showToast('Archived locally, failed to sync to server', 2200));
  if (state.curSess === id) {
    const ids = Object.keys(state.sessions).filter(sid => sid !== id && !state.sessions[sid].archived).sort((a, b) => state.sessions[b].ts - state.sessions[a].ts);
    if (ids.length) switchTo(ids[0]);
    else { state.curSess = null; localStorage.removeItem('ag_a'); renderChat(); }
  }
  renderSidebar();
  showToast('Archived', 1200);
}

export function unarchiveSession(id) {
  if (!state.sessions[id]) return;
  persistSessionMeta(id, { archived: false });
  unarchiveSessionApi(id)
    .then(({ ok }) => { if (!ok) throw 0 })
    .catch(() => showToast('Restored locally, failed to sync to server', 2200));
  save(); renderSidebar();
  showToast('Restored', 1200);
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
  showToast('Fork created', 1200);
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
  if (!t) { showToast('Session name cannot be empty', 1500); return; }
  if (t === s.title) return;
  s.title = t; save();
  const { ok } = await renameSessionApi(id, t);
  if (!ok) { showToast('Failed to sync rename to server', 2000); return; }
  showToast('Renamed', 1200);
}

export function renameKey(ev, id) {
  if (ev.key === 'Enter') commitSessionRename(id, ev.target.value);
  else if (ev.key === 'Escape') cancelSessionRename();
}

// ---- Topbar "..." chat menu (rename/fork/archive the current session) ----
export function toggleChatMenu() {
  state.chatMenuOpen = !state.chatMenuOpen;
}

export function closeChatMenu() {
  state.chatMenuOpen = false;
}

export async function renameCurrentSession() {
  if (!state.curSess) return;
  state.chatMenuOpen = false;
  state.renamingSessionId = state.curSess;
  nextTick(() => {
    const inp = document.getElementById('renameTbInput');
    if (inp) { inp.focus(); inp.select(); }
  });
}

export function forkCurrentSession() {
  state.chatMenuOpen = false;
  if (!state.curSess || !state.sessions[state.curSess]) return;
  forkSession(state.curSess, (state.sessions[state.curSess].msgs || []).length - 1);
}

export function archiveCurrentSession() {
  state.chatMenuOpen = false;
  if (!state.curSess) return;
  archiveSession(state.curSess);
}

// Download the current session as a Markdown file (client-side only — no
// server round-trip needed since messages already live in state.sessions).
export function exportCurrentSessionMarkdown() {
  state.chatMenuOpen = false;
  const id = state.curSess;
  const s = id && state.sessions[id];
  if (!s) return;
  const msgs = (s.msgs || []).filter(m => m.role === 'user' || m.role === 'assistant');
  if (!msgs.length) { showToast('No messages to export', 1500); return; }

  const lines = [`# ${s.title || 'Chat'}`, '', `_Exported: ${new Date().toISOString()}_`, '', '---'];
  for (const m of msgs) {
    lines.push('', `## ${m.role === 'user' ? 'User' : 'Assistant'}`, '');
    lines.push((m.content || '').trim());
    const toolCalls = (m.steps || []).filter(st => st.type === 'tool').length;
    if (toolCalls) lines.push('', `_(${toolCalls} tool call${toolCalls > 1 ? 's' : ''})_`);
  }
  const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const safeName = (s.title || 'chat').trim().replace(/[^\w\-]+/g, '_').slice(0, 60) || 'chat';
  const a = document.createElement('a');
  a.href = url;
  a.download = `${safeName}_${new Date().toISOString().slice(0, 10)}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast('Exported to Markdown', 1200);
}
