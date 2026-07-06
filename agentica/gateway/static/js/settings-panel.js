// ============ SETTINGS / PROFILES ============
// Web UI only supports adding new profiles + deleting them. Editing an
// existing profile's model/tuning must be done by hand in config.yaml (the
// file remains the single source of truth for profile definitions).
import { state } from './state.js';
import { showToast, showConfirm } from './modals.js';
import { loadProfiles } from './model-panel.js';
import { fetchProfiles, createProfileApi, deleteProfileApi } from './api.js';

export async function openSettingsModal() {
  state.settingsModal.open = true;
  cancelProfileForm();
  await loadSettingsProfiles();
}

export function closeSettingsModal() {
  state.settingsModal.open = false;
}

async function loadSettingsProfiles() {
  const { ok, data } = await fetchProfiles();
  if (!ok) return;
  state.profilesData = data;
}

export function showProfileForm() {
  const m = state.settingsModal;
  m.formOpen = true;
  const f = m.form;
  f.name = '';
  f.model_provider = '';
  f.model_name = '';
  f.base_url = '';
  f.api_key = '';
  f.reasoning_effort = '';
  f.max_tokens = '';
  f.context_window = '';
  f.temperature = '';
  f.top_p = '';
  f.aux_provider = '';
  f.aux_model = '';
  f.aux_base_url = '';
  f.aux_api_key = '';
  f.envRows = [];
  setTimeout(() => document.getElementById('pfName')?.focus(), 0);
}

export function cancelProfileForm() {
  state.settingsModal.formOpen = false;
}

export function addEnvRow() {
  state.settingsModal.form.envRows.push({ key: '', value: '' });
}

export function removeEnvRow(i) {
  state.settingsModal.form.envRows.splice(i, 1);
}

function collectEnv() {
  const env = {};
  for (const row of state.settingsModal.form.envRows) {
    const k = (row.key || '').trim();
    if (k) env[k] = row.value;
  }
  return env;
}

export async function saveProfileForm() {
  const f = state.settingsModal.form;
  const name = f.name.trim();
  const provider = f.model_provider.trim();
  const model = f.model_name.trim();
  if (!name || !provider || !model) {
    showToast('name, provider, model_name are required', 2500);
    return;
  }
  const auxProvider = f.aux_provider.trim();
  const auxModel = f.aux_model.trim();
  const auxiliary = (auxProvider || auxModel) ? {
    model_provider: auxProvider,
    model_name: auxModel,
    base_url: f.aux_base_url.trim(),
    api_key: f.aux_api_key || undefined,
  } : undefined;
  const env = collectEnv();
  const body = {
    name,
    model_provider: provider,
    model_name: model,
    base_url: f.base_url.trim(),
    api_key: f.api_key || undefined,
    reasoning_effort: f.reasoning_effort.trim() || undefined,
    max_tokens: parseInt(f.max_tokens) || undefined,
    context_window: parseInt(f.context_window) || undefined,
    temperature: parseFloat(f.temperature) || undefined,
    top_p: parseFloat(f.top_p) || undefined,
    auxiliary_model: auxiliary,
    env: Object.keys(env).length ? env : undefined,
  };
  const { ok, data: d } = await createProfileApi(body);
  if (!ok) { showToast(d?.detail || 'Save failed', 2500); return; }
  cancelProfileForm();
  await loadSettingsProfiles();
  loadProfiles();
  showToast('Profile created', 1500);
}

export async function deleteProfile(name) {
  showConfirm('Delete profile "' + name + '"?', async () => {
    const { ok, data: d } = await deleteProfileApi(name);
    if (!ok) { showToast(d?.detail || 'Delete failed', 2500); return; }
    await loadSettingsProfiles();
    loadProfiles();
    showToast('Profile deleted', 1500);
  });
}
