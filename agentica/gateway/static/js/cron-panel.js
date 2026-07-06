// ============ CRON JOBS (P1.2) ============
import { state } from './state.js';
import { showToast, showConfirm } from './modals.js';
import {
  fetchCronJobs, createCronJobApi, updateCronJobApi, deleteCronJobApi,
  pauseCronJobApi, resumeCronJobApi, triggerCronJobApi, fetchCronRuns,
} from './api.js';

export async function openCronModal() {
  state.cronModal.open = true;
  cancelCronForm();
  await loadCronJobs();
}

export function closeCronModal() {
  state.cronModal.open = false;
}

async function loadCronJobs() {
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
  const { ok, data: d } = m.editingId
    ? await updateCronJobApi(m.editingId, body)
    : await createCronJobApi(body);
  if (!ok) { showToast(d?.detail || 'Save failed', 2500); return; }
  const wasEditing = !!m.editingId;
  cancelCronForm();
  await loadCronJobs();
  showToast(wasEditing ? 'Job updated' : 'Job created', 1500);
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
  showToast('Triggering job...', 1500);
  const { ok, data: d } = await triggerCronJobApi(id);
  if (!ok) { showToast(d?.detail || 'Trigger failed', 2500); return; }
  showToast('Job will run on next tick', 2000);
  await loadCronJobs();
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
