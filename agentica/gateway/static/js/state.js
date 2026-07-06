import { reactive } from './vendor/petite-vue.js';

// ============ SHARED MUTABLE STATE ============
// Plain mutable object wrapped in petite-vue's reactive() so the petite-vue
// templates (input box, modals, sidebar) re-render automatically when any
// field is mutated. Every module keeps reading/writing `state.foo = ...`
// exactly as before — reactive() is a transparent Proxy over the same object,
// so this is the single source of truth for both business logic and UI.
export const state = reactive({
  curSess: null,
  sessions: {},
  streaming: false,
  abortCtrl: null,
  pendingFiles: [],
  serverModel: '-',
  serverDir: '',
  serverProvider: '',
  serverModelName: '',
  serverVersion: '',
  serverConfigPath: '',
  serverReasoningEffort: '',
  modelsData: null,
  userScrolledUp: false,
  _scrollLock: false, // 用户主动上翻后锁定，不再因内容增长而意外滚动
  serverContextWindow: 128000,
  serverProfile: '',
  profilesData: { active: '', profiles: [] },
  providersList: [],
  renamingSessionId: null,
  renamingProjectId: null,
  sidebarSearch: '',
  // Bumped by saveProjectMeta() — project metadata lives in localStorage, not
  // in this reactive object, so the sidebar's petite-vue template reads this
  // counter to know when to recompute its project tree from localStorage.
  projectMetaVersion: 0,

  // ---- input box (petite-vue driven) ----
  inputText: '',
  modelDDOpen: false,
  quickMenuOpen: false,
  approvalMenuOpen: false,
  ctxTipOpen: false,
  switchingLabel: null,
  selectedApprovalMode: 'ask',
  goalInput: '',
  skillInput: '',
  toolInput: '',

  // ---- toast / confirm ----
  toast: { show: false, msg: '' },
  confirm: { open: false, msg: '' },

  // ---- dir modal ----
  dirModal: { open: false, forNewSession: false, value: '', historyOpen: false, historyList: [] },

  // ---- account / plugins panels ----
  accountPanelOpen: false,
  pluginsPanelOpen: false,

  // ---- GATEWAY_TOKEN auth (see api.js) ----
  authRequired: false,
  gatewayTokenInput: '',

  // ---- cron modal ----
  cronModal: {
    open: false, jobs: [], formOpen: false, editingId: null,
    name: '', prompt: '', schedule: '', timeout: '', retries: '',
    openRuns: {}, runsData: {}, runsLoading: {},
  },

  // ---- settings/profiles modal ----
  settingsModal: {
    open: false, formOpen: false,
    form: {
      name: '', model_provider: '', model_name: '', base_url: '', api_key: '',
      reasoning_effort: '', max_tokens: '', context_window: '', temperature: '', top_p: '',
      aux_provider: '', aux_model: '', aux_base_url: '', aux_api_key: '',
      envRows: [],
    },
  },
});

// ============ PROJECT META (local sidebar tree: projects + fork/archive links) ============
const PROJECT_META_KEY = 'ag_project_meta_v1';
export const UNFILED_PROJECT_ID = 'unfiled';

function emptyProjectMeta() {
  return {
    projects: {
      [UNFILED_PROJECT_ID]: {
        id: UNFILED_PROJECT_ID, name: 'Unfiled', dir: '', collapsed: false, removed: false, ts: 0,
      },
    },
    sessionMeta: {},
  };
}

export function loadProjectMeta() {
  let meta;
  try { meta = JSON.parse(localStorage.getItem(PROJECT_META_KEY) || '{}') } catch { meta = {} }
  const base = emptyProjectMeta();
  meta = {
    projects: { ...base.projects, ...(meta.projects || {}) },
    sessionMeta: meta.sessionMeta || {},
  };
  if (!meta.projects[UNFILED_PROJECT_ID]) meta.projects[UNFILED_PROJECT_ID] = base.projects[UNFILED_PROJECT_ID];
  return meta;
}

export function saveProjectMeta(meta) {
  localStorage.setItem(PROJECT_META_KEY, JSON.stringify(meta));
  state.projectMetaVersion++;
}

export function projectIdForDir(dir) {
  const d = (dir || '').trim();
  if (!d) return UNFILED_PROJECT_ID;
  return 'dir:' + encodeURIComponent(d.replace(/\/+$/, ''));
}

export function projectNameForDir(dir) {
  const d = (dir || '').replace(/\/+$/, '');
  if (!d) return 'Unfiled';
  const parts = d.split('/').filter(Boolean);
  return parts[parts.length - 1] || d;
}

export function ensureProjectForSession(sessionId, session, meta) {
  const sessionMeta = meta.sessionMeta[sessionId] || {};
  const projectId = sessionMeta.projectId || session.projectId || projectIdForDir(session.dir || '');
  if (!meta.projects[projectId]) {
    meta.projects[projectId] = {
      id: projectId,
      name: projectNameForDir(session.dir || ''),
      dir: session.dir || '',
      collapsed: false,
      removed: false,
      ts: session.ts || Date.now(),
    };
  }
  const project = meta.projects[projectId];
  if (session.dir && !project.dir) project.dir = session.dir;
  project.ts = Math.max(project.ts || 0, session.ts || 0);
  meta.sessionMeta[sessionId] = {
    projectId,
    archived: !!(sessionMeta.archived || session.archived),
    parentSessionId: sessionMeta.parentSessionId || session.parentSessionId || null,
    forkedFromMsgIdx: sessionMeta.forkedFromMsgIdx ?? session.forkedFromMsgIdx ?? null,
    editedFromMsgIdx: sessionMeta.editedFromMsgIdx ?? session.editedFromMsgIdx ?? null,
  };
  session.projectId = projectId;
  session.archived = !!meta.sessionMeta[sessionId].archived;
  session.parentSessionId = meta.sessionMeta[sessionId].parentSessionId;
  session.forkedFromMsgIdx = meta.sessionMeta[sessionId].forkedFromMsgIdx;
  session.editedFromMsgIdx = meta.sessionMeta[sessionId].editedFromMsgIdx;
  return session;
}

export function buildProjectTree(sessionMap, meta) {
  const byProject = {};
  const archivedSessions = [];
  for (const id of Object.keys(sessionMap)) {
    const s = sessionMap[id];
    ensureProjectForSession(id, s, meta);
    const sm = meta.sessionMeta[id];
    const item = { id, session: s, meta: sm };
    if (sm.archived) { archivedSessions.push(item); continue; }
    const p = meta.projects[sm.projectId] || meta.projects[UNFILED_PROJECT_ID];
    if (p.removed) continue;
    if (!byProject[p.id]) byProject[p.id] = { ...p, sessions: [] };
    byProject[p.id].sessions.push(item);
  }
  const activeProjects = Object.values(byProject)
    .map(p => ({ ...p, sessions: p.sessions.sort((a, b) => b.session.ts - a.session.ts) }))
    .sort((a, b) => (b.sessions[0]?.session.ts || b.ts || 0) - (a.sessions[0]?.session.ts || a.ts || 0));
  archivedSessions.sort((a, b) => b.session.ts - a.session.ts);
  return { activeProjects, archivedSessions };
}

export function createForkSessionDraft(source, fromMsgIdx) {
  const end = Math.max(0, Math.min(fromMsgIdx + 1, (source.msgs || []).length));
  const titleBase = (source.title || 'Chat').replace(/\s+\(fork\)$/, '');
  return {
    title: titleBase + ' (fork)',
    msgs: (source.msgs || []).slice(0, end).map(m => ({ ...m })),
    ts: Date.now(),
    tokIn: 0, tokOut: 0, tokTotal: 0, requests: 0, totalTime: 0, usageEntries: [], lastInputTokens: 0,
    dir: source.dir || state.serverDir || '',
    projectId: source.projectId || projectIdForDir(source.dir || state.serverDir || ''),
    archived: false,
    parentSessionId: null,
    forkedFromMsgIdx: fromMsgIdx,
    editedFromMsgIdx: null,
  };
}

export function persistSessionMeta(sessionId, updates) {
  const meta = loadProjectMeta();
  const s = state.sessions[sessionId];
  if (s) ensureProjectForSession(sessionId, s, meta);
  meta.sessionMeta[sessionId] = { ...(meta.sessionMeta[sessionId] || {}), ...updates };
  if (s) {
    if (Object.prototype.hasOwnProperty.call(updates, 'archived')) s.archived = !!updates.archived;
    if (Object.prototype.hasOwnProperty.call(updates, 'projectId')) s.projectId = updates.projectId;
    if (Object.prototype.hasOwnProperty.call(updates, 'parentSessionId')) s.parentSessionId = updates.parentSessionId;
    if (Object.prototype.hasOwnProperty.call(updates, 'forkedFromMsgIdx')) s.forkedFromMsgIdx = updates.forkedFromMsgIdx;
    if (Object.prototype.hasOwnProperty.call(updates, 'editedFromMsgIdx')) s.editedFromMsgIdx = updates.editedFromMsgIdx;
  }
  saveProjectMeta(meta);
}
