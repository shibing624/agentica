export type ApiResult<T = any> = { ok: boolean; status: number; data: T | null; error?: unknown };

/** The header that lets a multipart write through the CSRF check (see
 *  `gateway/auth.py::_csrf_ok`). A cross-site form can forge the body type but
 *  not a custom header. */
export const CLIENT_HEADER = "X-Agentica-Client";

/** Where to send a browser whose session has gone. Set by the app so `api.ts`
 *  does not have to know about the router. */
let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(fn: () => void) { onUnauthorized = fn; }

async function request<T = any>(url: string, options: RequestInit = {}): Promise<ApiResult<T>> {
  try {
    const r = await fetch(url, options);
    // One place, because a 401 can come back from any of ~40 calls: the
    // session expired or the gateway restarted without one, and every caller's
    // correct reaction is the same.
    if (r.status === 401 && !url.startsWith("/api/auth/")) onUnauthorized?.();
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
export const fetchChannels = () => request("/api/channels");
export const fetchPrefs = () => request("/api/prefs");
export const savePrefsApi = (body: {
  theme?: string; lang?: string; approval_mode?: string; last_session_id?: string | null;
  auto_extract_memory?: boolean;
}) => putJson("/api/prefs", body);
export const fetchProfiles = () => request("/api/profiles");
export const fetchProviders = () => request("/api/providers");
export const fetchProfileDetail = (name: string) => request(`/api/profile/${encodeURIComponent(name)}`);
export const switchProfileApi = (name: string) => postJson("/api/profile/switch", { name });
export const fetchDirHistory = () => request("/api/config/dir_history");
export const deleteDirHistoryApi = (path?: string) =>
  request(`/api/config/dir_history${path ? "?path=" + encodeURIComponent(path) : ""}`, { method: "DELETE" });
export const saveBaseDirApi = (base_dir: string) => postJson("/api/config/base_dir", { base_dir });
export const openPathApi = (path: string, app: string) => postJson("/api/open", { path, app });
export const openUrlApi = (url: string) => postJson("/api/open", { url, app: "finder" });
export const startWechatQrApi = () => postJson<{
  status: string; qrcode?: string; png?: string; expires_in?: number; detail?: string;
}>("/api/channels/wechat/qr", {});
export const pollWechatQrApi = (id: string) =>
  request<{ status: string }>(`/api/channels/wechat/qr?id=${encodeURIComponent(id)}`);
export const fetchFsBrowse = (path: string) => request(`/api/fs/browse${path ? "?path=" + encodeURIComponent(path) : ""}`);
export const fetchSessions = () => request("/api/sessions");
export const deleteSessionApi = (id: string) => request(`/api/sessions/${id}`, { method: "DELETE" });
export const archiveSessionApi = (id: string) => request(`/api/sessions/${id}/archive`, { method: "POST" });
export const unarchiveSessionApi = (id: string) => request(`/api/sessions/${id}/unarchive`, { method: "POST" });
export const renameSessionApi = (id: string, name: string) => postJson(`/api/sessions/${id}/rename`, { name });
export const markSessionReadApi = (id: string) => request(`/api/sessions/${id}/read`, { method: "POST" });

export async function uploadFileApi(file: File, targetDir?: string) {
  const fd = new FormData();
  fd.append("file", file);
  if (targetDir) fd.append("target_dir", targetDir);
  return request("/api/upload", {
    method: "POST",
    body: fd,
    headers: { [CLIENT_HEADER]: "web" },
  });
}

export const fetchAuthStatus = () => request("/api/auth/status");
export const loginApi = (username: string, password: string) =>
  postJson("/api/auth/login", { username, password });
export const logoutApi = () => postJson("/api/auth/logout", {});
export const fetchUsers = () => request("/api/auth/users");
export const createUserApi = (username: string, password: string) =>
  postJson("/api/auth/users", { username, password });
export const changeUserPasswordApi = (userId: string, password: string) =>
  postJson(`/api/auth/users/${encodeURIComponent(userId)}/password`, { password });
export const deleteUserApi = (userId: string) =>
  request(`/api/auth/users/${encodeURIComponent(userId)}`, { method: "DELETE" });
export const setPasswordApi = (password: string, old_password?: string) =>
  postJson("/api/auth/password", old_password ? { password, old_password } : { password });

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
export function streamGoal(payload: unknown, signal?: AbortSignal) {
  return fetch("/api/goal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}
export const compactSessionApi = (session_id: string, instructions = "") =>
  postJson(`/api/sessions/${encodeURIComponent(session_id)}/compact`, { instructions });
export const makeTempDirApi = () => postJson<{ path: string }>("/api/fs/temp", {});
export const createProfileApi = (body: unknown) => postJson("/api/profile", body);
export const updateProfileApi = (name: string, body: unknown) => putJson(`/api/profile/${encodeURIComponent(name)}`, body);
export const deleteProfileApi = (name: string) => request(`/api/profile/${encodeURIComponent(name)}`, { method: "DELETE" });
export const fetchTraceAnalysis = (sessionId: string) => request(`/api/sessions/${sessionId}/trace/analysis`);
export const fetchSessionUsage = (sessionId: string) => request(`/api/sessions/${sessionId}/usage`);
export const fetchTraceEvents = (sessionId: string, offset = 0, limit = 200) =>
  request(`/api/sessions/${sessionId}/trace/events?offset=${offset}&limit=${limit}`);

export function workspaceContentUrl(root: string, path: string, download = false) {
  const q = new URLSearchParams({ root, path });
  if (download) q.set("download", "1");
  return `/api/workspace/content?${q.toString()}`;
}

export const fetchWorkspaceFiles = (root: string, path: string) =>
  request(`/api/workspace/files?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`);

export const fetchWorkspacePreview = (root: string, path: string) =>
  request(`/api/workspace/content?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}&preview=1`);

export const statWorkspaceFiles = (root: string, paths: string[]) =>
  postJson<{ existing: string[] }>("/api/workspace/stat", { root, paths });

export async function uploadWorkspaceFile(file: File, root: string, dir: string) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("root", root);
  fd.append("path", dir);
  return request("/api/workspace/upload", {
    method: "POST",
    body: fd,
    headers: { [CLIENT_HEADER]: "web" },
  });
}

export function streamChat(payload: unknown, signal?: AbortSignal) {
  return fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export const createChatRunApi = (body: unknown) =>
  postJson<{ run_id: string; session_id: string; status: string; kind: string; seq: number }>(
    "/api/chat/runs", body,
  );

export function runEvents(runId: string, after = 0, signal?: AbortSignal) {
  return fetch(
    `/api/chat/runs/${encodeURIComponent(runId)}/events?after=${after}`,
    { signal },
  );
}

export const fetchActiveRun = (sessionId: string) =>
  request<{ run: { run_id: string; session_id: string; status: string; kind: string; seq: number } | null }>(
    `/api/chat/runs/active?session_id=${encodeURIComponent(sessionId)}`,
  );

export const cancelRunApi = (runId: string) =>
  postJson<{ status: string; cancelled: boolean; run_id?: string }>(
    `/api/chat/runs/${encodeURIComponent(runId)}/cancel`, {},
  );

export function attachChatStream(sessionId: string, signal?: AbortSignal, after = 0) {
  return fetch(
    `/api/chat/stream/${encodeURIComponent(sessionId)}?after=${after}`,
    { signal },
  );
}

export const postSessionApproval = (
  sessionId: string,
  toolCallId: string,
  decision: "allow" | "allow_prefix" | "deny",
) =>
  postJson<{ ok?: boolean; tool_call_id?: string; decision?: string }>(
    `/api/sessions/${encodeURIComponent(sessionId)}/approvals/${encodeURIComponent(toolCallId)}`,
    { decision },
  );

export const cancelChatApi = (session_id: string) =>
  postJson<{ status?: string; cancelled: boolean }>("/api/chat/cancel", { session_id, message: "" });

export const steerChatApi = (session_id: string, message: string) =>
  postJson<{ accepted: boolean }>("/api/chat/steer", { session_id, message });

export const takeSteerApi = (session_id: string) =>
  postJson<{ messages: string[] }>("/api/chat/steer/take", { session_id, message: "" });

export type MemoryDoc = {
  content: string;
  path: string;
  empty_template: boolean;
  auto_extract: boolean;
  user_id: string;
};

export const fetchMemory = () => request<MemoryDoc>("/api/memory");
export const saveMemoryApi = (content: string) => putJson<MemoryDoc>("/api/memory", { content });
