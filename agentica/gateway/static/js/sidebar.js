// ============ SIDEBAR (project tree + session list) ============
import { state, loadProjectMeta, saveProjectMeta, buildProjectTree } from './state.js';
import { nextTick } from './vendor/petite-vue.js';
import { ago, shortenPath } from './utils.js';
import { showToast } from './modals.js';

// Reactive getter consumed by the petite-vue sidebar template (see main.js's
// `sidebarTree` root getter). Visibility of each row is computed independently
// from the search box, matching the original per-row filtering behavior.
export function sidebarTree() {
  void state.projectMetaVersion; // establish reactive dependency — see state.js
  const ids = Object.keys(state.sessions);
  if (!ids.length) return { hasSessions: false, projects: [] };
  // Read-only: unlike the old imperative renderSidebar(), this getter must not
  // write back to localStorage (via saveProjectMeta) — it can run many times
  // per interaction as a reactive computation, and saveProjectMeta() bumps
  // `state.projectMetaVersion`, which would re-trigger this same getter from
  // other effects and thrash. Auto-derived project defaults are idempotent,
  // so skipping persistence here is safe; explicit user actions (collapse,
  // rename, remove, archive) already call saveProjectMeta() themselves.
  const meta = loadProjectMeta();
  const tree = buildProjectTree(state.sessions, meta);
  const q = (state.sidebarSearch || '').toLowerCase();
  const projects = tree.activeProjects.map(project => {
    const searchStr = (project.name + ' ' + (project.dir || '')).toLowerCase();
    return {
      id: project.id,
      name: project.name,
      dir: project.dir,
      dirShort: project.dir ? shortenPath(project.dir) : 'No directory',
      collapsed: project.collapsed,
      visible: !q || searchStr.includes(q),
      sessions: project.sessions.map(item => {
        const s = item.session;
        const sessionSearch = (s.title + ' ' + (s.dir || '')).toLowerCase();
        return {
          id: item.id,
          title: s.title,
          n: s.msgs && s.msgs.length ? s.msgs.filter(m => m.role === 'user').length : (s.user_count || 0),
          agoStr: ago(s.ts),
          dirShort: s.dir ? shortenPath(s.dir) : '',
          dir: s.dir || '',
          isFork: !!s.parentSessionId,
          unread: !!s.unread,
          visible: !q || sessionSearch.includes(q),
        };
      }),
    };
  });
  return { hasSessions: true, projects };
}

export function toggleProjectCollapsed(projectId) {
  const meta = loadProjectMeta();
  const project = meta.projects[projectId]; if (!project) return;
  project.collapsed = !project.collapsed;
  saveProjectMeta(meta);
}

export function renameProject(projectId) {
  state.renamingProjectId = projectId;
  nextTick(() => {
    const inp = document.getElementById(`project_rename_${projectId}`);
    if (inp) { inp.focus(); inp.select(); }
  });
}

export function commitProjectRename(projectId, value) {
  const meta = loadProjectMeta();
  const project = meta.projects[projectId]; if (!project) return;
  const name = (value || '').trim();
  state.renamingProjectId = null;
  if (name) project.name = name;
  saveProjectMeta(meta);
}

export function projectRenameKey(ev, projectId) {
  if (ev.key === 'Enter') commitProjectRename(projectId, ev.target.value);
  else if (ev.key === 'Escape') { state.renamingProjectId = null; }
}

export function removeProject(projectId) {
  const meta = loadProjectMeta();
  const project = meta.projects[projectId]; if (!project) return;
  project.removed = true;
  saveProjectMeta(meta);
  showToast('Project group hidden', 1200);
}

// The sidebar template re-renders reactively off `state.sessions` and
// `state.projectMetaVersion` (see `sidebarTree()` above) — this is kept as a
// no-op so call sites in sessions.js/chat.js/modals.js that historically
// triggered a manual DOM rebuild after mutating session/project data don't
// need to be touched.
export function renderSidebar() { }
