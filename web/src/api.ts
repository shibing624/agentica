export type ApiResult<T = any> = { ok: boolean; status: number; data: T | null; error?: unknown };

async function request<T = any>(url: string, options: RequestInit = {}): Promise<ApiResult<T>> {
  try {
    const r = await fetch(url, options);
    let data: T | null = null;
    try { data = await r.json(); } catch { /* no body */ }
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: null, error: e };
  }
}

function postJson<T = any>(url: string, body: unknown) {
  return request<T>(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}
function putJson<T = any>(url: string, body: unknown) {
  return request<T>(url, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

export const fetchStatus = () => request("/api/status");
export const fetchProfiles = () => request("/api/profiles");
export const fetchProviders = () => request("/api/providers");
export const fetchProfileDetail = (name: string) => request(`/api/profile/${encodeURIComponent(name)}`);
export const switchProfileApi = (name: string) => postJson("/api/profile/switch", { name });
export const setThinkingApi = (enabled: boolean) => postJson("/api/config/thinking", { enabled });
export const fetchThinking = () => request("/api/config/thinking");
export const fetchDirHistory = () => request("/api/config/dir_history");
export const saveBaseDirApi = (base_dir: string) => postJson("/api/config/base_dir", { base_dir });
export const openPathApi = (path: string, app: string) => postJson("/api/open", { path, app });
export const fetchFsBrowse = (path: string) => request(`/api/fs/browse${path ? "?path=" + encodeURIComponent(path) : ""}`);
export const fetchSessions = () => request("/api/sessions");
export const deleteSessionApi = (id: string) => request(`/api/sessions/${id}`, { method: "DELETE" });
export const archiveSessionApi = (id: string) => request(`/api/sessions/${id}/archive`, { method: "POST" });
export const unarchiveSessionApi = (id: string) => request(`/api/sessions/${id}/unarchive`, { method: "POST" });
export const renameSessionApi = (id: string, name: string) => postJson(`/api/sessions/${id}/rename`, { name });

export async function uploadFileApi(file: File, targetDir?: string) {
  const fd = new FormData();
  fd.append("file", file);
  if (targetDir) fd.append("target_dir", targetDir);
  return request("/api/upload", { method: "POST", body: fd });
}

export const fetchCronJobs = () => request("/api/scheduler/jobs");
export const createCronJobApi = (body: unknown) => postJson("/api/scheduler/jobs", body);
export const updateCronJobApi = (id: string, body: unknown) => putJson(`/api/scheduler/jobs/${id}`, body);
export const deleteCronJobApi = (id: string) => request(`/api/scheduler/jobs/${id}`, { method: "DELETE" });
export const pauseCronJobApi = (id: string) => request(`/api/scheduler/jobs/${id}/pause`, { method: "POST" });
export const resumeCronJobApi = (id: string) => request(`/api/scheduler/jobs/${id}/resume`, { method: "POST" });
export const triggerCronJobApi = (id: string) => request(`/api/scheduler/jobs/${id}/trigger`, { method: "POST" });
export const fetchCronRuns = (id: string) => request(`/api/scheduler/jobs/${id}/runs`);
export const polishPromptApi = (draft: string) => postJson("/api/scheduler/polish_prompt", { draft });
export const fetchTools = () => request("/api/tools");
export const fetchMcpServers = () => request("/api/mcp/servers");
export const createMcpServerApi = (body: unknown) => postJson("/api/mcp/servers", body);
export const deleteMcpServerApi = (name: string) => request(`/api/mcp/servers/${encodeURIComponent(name)}`, { method: "DELETE" });
export const fetchSkills = () => request("/api/skills");
export const fetchSkillDetail = (name: string) => request(`/api/skills/${encodeURIComponent(name)}`);
export const createSkillApi = (body: unknown) => postJson("/api/skills", body);
export const updateSkillApi = (name: string, body: unknown) => putJson(`/api/skills/${encodeURIComponent(name)}`, body);
export const deleteSkillApi = (name: string) => request(`/api/skills/${encodeURIComponent(name)}`, { method: "DELETE" });
export const runGoalApi = (objective: string, session_id: string) => postJson("/api/goal", { objective, session_id });
export const createProfileApi = (body: unknown) => postJson("/api/profile", body);
export const updateProfileApi = (name: string, body: unknown) => putJson(`/api/profile/${encodeURIComponent(name)}`, body);
export const deleteProfileApi = (name: string) => request(`/api/profile/${encodeURIComponent(name)}`, { method: "DELETE" });
export const fetchTraceAnalysis = (sessionId: string) => request(`/api/sessions/${sessionId}/trace/analysis`);
export const fetchTraceEvents = (sessionId: string, offset = 0, limit = 200) =>
  request(`/api/sessions/${sessionId}/trace/events?offset=${offset}&limit=${limit}`);

export function streamChat(payload: unknown, signal?: AbortSignal) {
  return fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}
