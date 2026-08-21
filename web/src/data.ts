import * as api from "./api";
import { applyLang } from "./i18n";
import { applyTheme, getState, setState, writeLastSessionId } from "./store";

/** Server-owned state loaders, shared by the chat page and every panel so a
 *  mutation can refresh its own list without reaching into a component. */

export async function loadStatus() {
  const { ok, data } = await api.fetchStatus();
  if (!ok || !data) return;
  setState({
    serverModel: data.model || "-",
    serverModelName: data.model_name || "",
    serverProvider: data.model_provider || "",
    serverDir: data.base_dir || "",
    serverVersion: data.version || "",
    serverConfigPath: data.config_path || "",
    serverProfile: data.active_profile || "",
    // Tuning is nested; reading a flat `reasoning_effort` left it blank until
    // the first profile switch.
    serverReasoningEffort: data.tuning?.reasoning_effort || "",
    serverContextWindow: data.context_window || getState().serverContextWindow,
    serverCompactTokenLimit: data.compact_token_limit || 0,
    serverSupportsImages: !!data.supports_images,
    serverMediaModel: data.media_model || "",
  });
}

export async function loadAuthStatus() {
  const { ok, data } = await api.fetchAuthStatus();
  if (!ok || !data) return;
  setState({
    passwordSet: !!data.password_set,
    passwordIsInitial: !!data.password_is_initial,
    accountId: data.user_id || data.default_account_id || "default",
    accountRole: data.role || "user",
    defaultAccountId: data.default_account_id || "default",
    sessionVia: data.via || "",
    // The server owns the minimum: a number hardcoded here would go stale the
    // day it moves and the only symptom is a form that rejects what the API
    // accepts (or worse, the reverse).
    minPasswordLength: data.min_password_length || 6,
  });
}

export async function loadPrefs() {
  const { ok, data } = await api.fetchPrefs();
  if (!ok || !data) return;
  if (data.theme) {
    localStorage.setItem("ag_theme", data.theme);
    applyTheme(data.theme);
    setState({ theme: data.theme });
  }
  if (data.lang === "zh" || data.lang === "en") {
    localStorage.setItem("ag_lang", data.lang);
    applyLang(data.lang);
    setState({ lang: data.lang });
  }
  if (data.approval_mode) {
    localStorage.setItem("ag_approval", data.approval_mode);
    setState({ selectedApprovalMode: data.approval_mode });
  }
  if (data.last_session_id) writeLastSessionId(data.last_session_id, false);
}

/** The account table. Admin-only server-side; a 403 just leaves it empty. */
export async function loadUsers() {
  const { ok, data } = await api.fetchUsers();
  setState({ users: ok && data ? data.users || [] : [] });
}

export async function loadProfiles() {
  const { ok, data } = await api.fetchProfiles();
  if (ok && data) setState({ profilesData: data });
}

export async function loadProviders() {
  const { ok, data } = await api.fetchProviders();
  if (ok && data?.providers) setState({ providers: data.providers });
}

export async function loadPlugins() {
  const [tools, skills, mcp] = await Promise.all([
    api.fetchTools(), api.fetchSkills(), api.fetchMcpServers(),
  ]);
  setState({
    pluginsData: {
      tools: tools.data?.tools || [],
      skills: skills.data?.skills || [],
      mcpServers: mcp.data?.servers || [],
    },
  });
}

export async function loadCronJobs() {
  const { ok, data } = await api.fetchCronJobs();
  if (ok && data) setState({ cronJobs: data.jobs || [] });
}

export async function loadDirHistory() {
  const { ok, data } = await api.fetchDirHistory();
  if (ok && data?.history) setState({ dirHistory: data.history });
}

export async function browseDir(path: string) {
  const { ok, data } = await api.fetchFsBrowse(path);
  if (!ok || !data) return false;
  setState({ dirBrowse: { open: true, path: data.path, parent: data.parent, dirs: data.dirs || [] } });
  return true;
}
