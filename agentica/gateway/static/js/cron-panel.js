// ============ CRON JOBS (P1.2) ============
import { state } from './state.js';
import { showToast, showConfirm } from './modals.js';
import {
  fetchCronJobs, createCronJobApi, updateCronJobApi, deleteCronJobApi,
  pauseCronJobApi, resumeCronJobApi, triggerCronJobApi, fetchCronRuns, polishPromptApi,
} from './api.js';

// Cron jobs are rendered inside the Settings modal's "Scheduled Jobs" tab
// (see settings-panel.js's openSettingsModal/switchSettingsTab, which own
// loading this data when that tab becomes active).
export async function loadCronJobs() {
  const { ok, data } = await fetchCronJobs();
  if (!ok) return;
  state.cronModal.jobs = data.jobs || [];
}

export function fmtCronTime(ms) {
  if (!ms) return '-';
  return new Date(ms).toLocaleString();
}

export function showCronForm(job) {
  const m = state.cronModal;
  m.editingId = job ? job.id : null;
  m.formOpen = true;
  m.name = job ? job.name : '';
  m.prompt = job ? job.prompt : '';
  m.schedule = job ? job.schedule : '';
  m.timeout = job && job.timeout_seconds ? job.timeout_seconds : '';
  m.retries = job && job.max_retries ? job.max_retries : '';
  m.validateRun = true;
  setTimeout(() => document.getElementById('cronName')?.focus(), 0);
}

export function cancelCronForm() {
  state.cronModal.editingId = null;
  state.cronModal.formOpen = false;
}

export function editCronJob(id) {
  const j = state.cronModal.jobs.find(x => x.id === id);
  if (j) showCronForm(j);
}

export async function saveCronForm() {
  const m = state.cronModal;
  const prompt = m.prompt.trim();
  const schedule = m.schedule.trim();
  if (!prompt || !schedule) {
    showToast('prompt and schedule are required', 2500);
    return;
  }
  const body = {
    prompt,
    schedule,
    name: m.name.trim(),
    timeout_seconds: parseInt(m.timeout) || 0,
    max_retries: parseInt(m.retries) || 0,
  };
  if (!m.editingId) body.validate_run = m.validateRun;
  const { ok, data: d } = m.editingId
    ? await updateCronJobApi(m.editingId, body)
    : await createCronJobApi(body);
  if (!ok) { showToast(d?.detail || 'Save failed', 2500); return; }
  const wasEditing = !!m.editingId;
  cancelCronForm();
  await loadCronJobs();
  if (!wasEditing && body.validate_run) showToast('Job created, will validate with one run on the next schedule cycle', 2500);
  else showToast(wasEditing ? 'Job updated' : 'Job created', 1500);
}

export async function polishCronPrompt() {
  const m = state.cronModal;
  const draft = m.prompt.trim();
  if (!draft) return;
  m.polishing = true;
  const { ok, data } = await polishPromptApi(draft);
  m.polishing = false;
  if (!ok) { showToast(data?.detail || 'AI polish failed', 2500); return; }
  m.prompt = data.prompt;
  showToast('Prompt polished', 1200);
}

export async function deleteCronJob(id) {
  showConfirm('Delete this cron job?', async () => {
    const { ok } = await deleteCronJobApi(id);
    if (ok) await loadCronJobs();
  });
}

export async function pauseCronJob(id) {
  const { ok } = await pauseCronJobApi(id);
  if (ok) await loadCronJobs();
}

export async function resumeCronJob(id) {
  const { ok } = await resumeCronJobApi(id);
  if (ok) await loadCronJobs();
}

export async function triggerCronJob(id) {
  if (state.cronModal.triggering[id]) return;
  state.cronModal.triggering[id] = true;
  showToast('Running now…', 4000);
  const { ok, data: d } = await triggerCronJobApi(id);
  state.cronModal.triggering[id] = false;
  if (!ok) { showToast(d?.detail || 'Run failed', 2500); return; }
  const run = d.run || {};
  if (run.status === 'ok') showToast('✅ Run succeeded', 2000);
  else showToast('❌ Run failed: ' + (run.error || 'Unknown error'), 3500);
  await loadCronJobs();
  // Surface the fresh result immediately in the run history, instead of
  // making the user manually reopen it to see what just happened.
  state.cronModal.openRuns[id] = false;
  await toggleCronRuns(id);
}

export async function toggleCronRuns(id) {
  const m = state.cronModal;
  if (!m.openRuns[id]) {
    m.openRuns[id] = true;
    m.runsLoading[id] = true;
    const { ok, data } = await fetchCronRuns(id);
    m.runsLoading[id] = false;
    m.runsData[id] = ok && data ? (data.runs || []) : null;
  } else {
    m.openRuns[id] = false;
  }
}
