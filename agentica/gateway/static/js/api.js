// ============ API (raw HTTP layer, no state/rendering) ============
// Every function returns {ok, status, data} where `data` is the parsed JSON
// body (or null if the response has no body / isn't JSON). Callers decide
// how to interpret ok/status/data — this module never throws on non-2xx.
export const API = '';

async function request(url, options = {}) {
  try {
    const headers = { ...(options.headers || {}) };
    const r = await fetch(`${API}${url}`, { ...options, headers });
    let data = null;
    try { data = await r.json(); } catch { /* no/invalid JSON body */ }
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: null, error: e };
  }
}

function postJson(url, body) {
  return request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}
function putJson(url, body) {
  return request(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}

// ---- status / models / profiles / providers ----
export const fetchHealth = () => request('/api/health');
export const fetchStatus = () => request('/api/status');
export const fetchModels = () => request('/api/models');
export const fetchProfiles = () => request('/api/profiles');
export const fetchProviders = () => request('/api/providers');
export const switchProfileApi = (name) => postJson('/api/profile/switch', { name });
export const switchModelApi = (model_provider, model_name) => postJson('/api/model', { model_provider, model_name });
export const setThinkingApi = (enabled) => postJson('/api/config/thinking', { enabled });

// ---- working directory ----
export const fetchDirHistory = () => request('/api/config/dir_history');
export const saveBaseDirApi = (base_dir) => postJson('/api/config/base_dir', { base_dir });
export const openPathApi = (path, app) => postJson('/api/open', { path, app });
export const fetchFsBrowse = (path) => request(`/api/fs/browse${path ? '?path=' + encodeURIComponent(path) : ''}`);

// ---- sessions ----
export const fetchSessions = () => request('/api/sessions');
export const deleteSessionApi = (id) => request(`/api/sessions/${id}`, { method: 'DELETE' });
export const archiveSessionApi = (id) => request(`/api/sessions/${id}/archive`, { method: 'POST' });
export const unarchiveSessionApi = (id) => request(`/api/sessions/${id}/unarchive`, { method: 'POST' });
export const renameSessionApi = (id, name) => postJson(`/api/sessions/${id}/rename`, { name });

// ---- file upload ----
export async function uploadFileApi(file, targetDir) {
  const fd = new FormData();
  fd.append('file', file);
  if (targetDir) fd.append('target_dir', targetDir);
  return request('/api/upload', { method: 'POST', body: fd });
}

// ---- cron jobs ----
export const fetchCronJobs = () => request('/api/scheduler/jobs');
export const createCronJobApi = (body) => postJson('/api/scheduler/jobs', body);
export const updateCronJobApi = (id, body) => putJson(`/api/scheduler/jobs/${id}`, body);
export const deleteCronJobApi = (id) => request(`/api/scheduler/jobs/${id}`, { method: 'DELETE' });
export const pauseCronJobApi = (id) => request(`/api/scheduler/jobs/${id}/pause`, { method: 'POST' });
export const resumeCronJobApi = (id) => request(`/api/scheduler/jobs/${id}/resume`, { method: 'POST' });
export const triggerCronJobApi = (id) => request(`/api/scheduler/jobs/${id}/trigger`, { method: 'POST' });
export const fetchCronRuns = (id) => request(`/api/scheduler/jobs/${id}/runs`);
export const polishPromptApi = (draft) => postJson('/api/scheduler/polish_prompt', { draft });

// ---- plugins: tools (read-only) + mcp servers (CRUD) + skills (CRUD) ----
export const fetchTools = () => request('/api/tools');
export const fetchMcpServers = () => request('/api/mcp/servers');
export const createMcpServerApi = (body) => postJson('/api/mcp/servers', body);
export const deleteMcpServerApi = (name) => request(`/api/mcp/servers/${encodeURIComponent(name)}`, { method: 'DELETE' });
export const fetchSkills = () => request('/api/skills');
export const fetchSkillDetail = (name) => request(`/api/skills/${encodeURIComponent(name)}`);
export const createSkillApi = (body) => postJson('/api/skills', body);
export const updateSkillApi = (name, body) => putJson(`/api/skills/${encodeURIComponent(name)}`, body);
export const deleteSkillApi = (name) => request(`/api/skills/${encodeURIComponent(name)}`, { method: 'DELETE' });

// ---- standing goal ----
export const runGoalApi = (objective, session_id) => postJson('/api/goal', { objective, session_id });

// ---- profiles (settings; web UI only creates + deletes, edits go in config.yaml) ----
export const createProfileApi = (body) => postJson('/api/profile', body);
export const deleteProfileApi = (name) => request(`/api/profile/${encodeURIComponent(name)}`, { method: 'DELETE' });

// ---- chat streaming (kept raw — caller needs the Response body reader) ----
export function streamChat(payload, signal) {
  return fetch(`${API}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
}
