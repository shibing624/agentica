// ============ MODEL DROPDOWN / STATUS / TOKEN USAGE ============
import { state } from './state.js';
import { fetchHealth, fetchStatus, fetchModels, fetchProfiles, fetchProviders, switchProfileApi } from './api.js';
import { showToast } from './modals.js';

export async function loadHealth() {
  const { ok, data } = await fetchHealth();
  if (ok && data) state.authRequired = !!data.auth_required;
}

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
  if (state.serverVersion) document.getElementById('verLabel').textContent = 'v' + state.serverVersion;
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
  state.switchingLabel = '切换中…';
  state.modelDDOpen = false;
  const { ok, data } = await switchProfileApi(name);
  state.switchingLabel = null;
  if (!ok) {
    showToast(data?.detail || '切换模型失败', 2500);
    return;
  }
  await loadStatus();
  await loadProfiles();
  showToast('已切换到 ' + name, 1500);
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
    entries: s && s.usageEntries ? s.usageEntries : [],
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
  }
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

export function triggerFolderPicker() {
  const input = document.getElementById('folderInput');
  if (input) input.click();
  state.quickMenuOpen = false;
}

export function closeInputMenus() {
  state.quickMenuOpen = false;
  state.approvalMenuOpen = false;
  state.modelDDOpen = false;
  state.ctxTipOpen = false;
}
