// ============ MODEL DROPDOWN / STATUS / TOKEN USAGE ============
import { state } from './state.js';
import { fetchStatus, fetchModels, fetchProfiles, fetchProviders, switchProfileApi, fetchDirHistory } from './api.js';
import { showToast } from './modals.js';
import { loadPluginsData } from './plugins-panel.js';

export async function loadStatus() {
  const { ok, data: d } = await fetchStatus();
  if (!ok || !d) return;
  state.serverModel = d.model || '-';
  state.serverDir = d.base_dir || d.workspace || '';
  state.serverProvider = d.model_provider || '';
  state.serverModelName = d.model_name || '';
  state.serverVersion = d.version || '';
  state.serverContextWindow = d.context_window || 128000;
  state.serverProfile = d.active_profile || '';
  state.serverConfigPath = d.config_path || '';
  state.serverReasoningEffort = d.tuning?.reasoning_effort || '';
}

export async function loadModels() {
  const { ok, data } = await fetchModels();
  if (ok) state.modelsData = data;
}

export function toggleModelDD() {
  state.modelDDOpen = !state.modelDDOpen;
  if (state.modelDDOpen) {
    state.quickMenuOpen = false;
    state.approvalMenuOpen = false;
    state.ctxTipOpen = false;
  }
}

export async function loadProfiles() {
  const { ok, data } = await fetchProfiles();
  if (ok) state.profilesData = data;
}

export async function loadProviders() {
  const { ok, data } = await fetchProviders();
  if (ok) state.providersList = data.providers || [];
}

export async function switchProfile(name) {
  if (!name || name === state.profilesData.active) { state.modelDDOpen = false; return }
  state.switchingLabel = 'Switching…';
  state.modelDDOpen = false;
  const { ok, data } = await switchProfileApi(name);
  state.switchingLabel = null;
  if (!ok) {
    showToast(data?.detail || 'Failed to switch model', 2500);
    return;
  }
  await loadStatus();
  await loadProfiles();
  showToast('Switched to ' + name, 1500);
}

export function tokenUsage() {
  const s = state.curSess && state.sessions[state.curSess] ? state.sessions[state.curSess] : null;
  const ctxWin = state.serverContextWindow || 128000;
  const lastIn = s ? s.lastInputTokens || 0 : 0;
  const tokIn = s ? s.tokIn || 0 : 0;
  const tokOut = s ? s.tokOut || 0 : 0;
  const ctxIn = lastIn > 0 ? lastIn : tokIn;
  const pct = ctxWin > 0 && ctxIn > 0 ? Math.min(Math.round((ctxIn / ctxWin) * 100), 100) : 0;
  return {
    lastIn, tokIn, tokOut, ctxWin, pct,
    requests: s ? s.requests || 0 : 0,
    totalTime: s ? s.totalTime || 0 : 0,
  };
}

export function toggleCtxTip() {
  state.ctxTipOpen = !state.ctxTipOpen;
  if (state.ctxTipOpen) {
    state.quickMenuOpen = false;
    state.approvalMenuOpen = false;
    state.modelDDOpen = false;
  }
}

export function toggleQuickMenu() {
  state.quickMenuOpen = !state.quickMenuOpen;
  if (state.quickMenuOpen) {
    state.approvalMenuOpen = false;
    state.modelDDOpen = false;
    state.ctxTipOpen = false;
    state.quickMenuSearch = '';
    loadPluginsData();
    fetchDirHistory().then(({ ok, data }) => { if (ok && data) state.dirModal.historyList = data.history || []; });
  }
}

// ---- "+" quick command palette: insert @path / /skill / /tool references
// directly into the input box, mirroring the CLI's slash-command syntax
// instead of composing hidden fields into the outgoing message. ----

export function slugifySkillName(name) {
  return '/' + name.toLowerCase().replace(/[\s_]+/g, '-').replace(/[^a-z0-9-]/g, '').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

export function quickSkills() {
  const q = (state.quickMenuSearch || '').toLowerCase().trim();
  const skills = state.pluginsData.skills || [];
  if (!q) return skills;
  return skills.filter(s => s.name.toLowerCase().includes(q) || (s.description || '').toLowerCase().includes(q));
}

function insertIntoInput(text) {
  const ta = document.getElementById('inputTa');
  const prefix = state.inputText && !state.inputText.endsWith(' ') && !state.inputText.endsWith('\n') ? state.inputText + ' ' : state.inputText;
  state.inputText = prefix + text;
  state.quickMenuOpen = false;
  setTimeout(() => { ta?.focus(); ta.selectionStart = ta.selectionEnd = ta.value.length; }, 0);
}

export function insertSkillRef(skill) {
  const trigger = skill.trigger || slugifySkillName(skill.name);
  insertIntoInput(trigger + ' ');
}

export function insertGoalPrefix() {
  state.inputText = '/goal ' + state.inputText;
  state.quickMenuOpen = false;
  const ta = document.getElementById('inputTa');
  setTimeout(() => { ta?.focus(); ta.selectionStart = ta.selectionEnd = ta.value.length; }, 0);
}

export function toggleApprovalMenu() {
  state.approvalMenuOpen = !state.approvalMenuOpen;
  if (state.approvalMenuOpen) {
    state.quickMenuOpen = false;
    state.modelDDOpen = false;
    state.ctxTipOpen = false;
  }
}

export function setApprovalMode(mode) {
  state.selectedApprovalMode = mode;
  state.approvalMenuOpen = false;
}

export function triggerFilePicker() {
  document.getElementById('fileInput')?.click();
  state.quickMenuOpen = false;
}

export function closeInputMenus() {
  state.quickMenuOpen = false;
  state.approvalMenuOpen = false;
  state.modelDDOpen = false;
  state.ctxTipOpen = false;
  state.slashOpen = false;
}

// ============ SLASH COMMAND PALETTE ============
// Mirrors the CLI's in-chat directive surface: typing "/" in the input opens a
// filtered palette of skills (by trigger) + the fixed /goal directive. Skills
// and goal are IN-CHAT directives (they flow into the conversation), so they
// belong in the chat box — management ops (/cron, /skills CRUD, /mcp) stay in
// their dedicated panels and are intentionally excluded here.
export function slashItems() {
  const text = (state.inputText || '').trim();
  if (!text.startsWith('/')) return [];
  const q = text.slice(1).toLowerCase().trim();
  const skills = state.pluginsData.skills || [];
  const items = [];
  for (const s of skills) {
    const label = s.trigger || slugifySkillName(s.name);
    items.push({ type: 'skill', label, name: s.name, description: s.description || '', skill: s });
  }
  items.push({ type: 'directive', label: '/goal', name: 'goal', description: 'Set a multi-turn standing objective' });
  if (!q) return items;
  return items.filter(it => it.label.toLowerCase().includes(q) || it.name.toLowerCase().includes(q));
}

export function updateSlash() {
  const text = (state.inputText || '').trim();
  if (text.startsWith('/') && !state.streaming) {
    if (state.slashCommitted) {
      // A selection was just committed; keep the palette closed until the
      // leading "/" is gone so continued typing doesn't reopen it.
      if (!text.startsWith('/')) state.slashCommitted = false;
      else return;
    }
    state.slashOpen = true;
    state.slashIndex = 0;
  } else {
    state.slashOpen = false;
    state.slashCommitted = false;
  }
}

export function selectSlash(item) {
  if (!item) return;
  state.slashOpen = false;
  state.slashCommitted = true;
  const ta = document.getElementById('inputTa');
  if (item.type === 'skill') {
    const trigger = item.skill.trigger || slugifySkillName(item.skill.name);
    state.inputText = trigger + ' ';
  } else if (item.name === 'goal') {
    state.inputText = '/goal ';
  }
  setTimeout(() => { ta?.focus(); ta.selectionStart = ta.selectionEnd = ta.value.length; }, 0);
}
