import * as api from "./api";
import { getState, setState } from "./store";

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
    serverThinking: data.model_thinking || "",
    // Tuning is nested; reading a flat `reasoning_effort` left it blank until
    // the first profile switch.
    serverReasoningEffort: data.tuning?.reasoning_effort || "",
    serverContextWindow: data.context_window || getState().serverContextWindow,
  });
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
