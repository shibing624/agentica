// ============ PLUGINS PANEL: built-in tools (read-only) + MCP servers (CRUD) + skills (CRUD) ============
import { state } from './state.js';
import {
  fetchTools, fetchMcpServers, createMcpServerApi, deleteMcpServerApi,
  fetchSkills, fetchSkillDetail, createSkillApi, updateSkillApi, deleteSkillApi,
} from './api.js';
import { showToast, showConfirm } from './modals.js';

export async function openPluginsPanel() {
  state.pluginsPanelOpen = true;
  cancelSkillForm();
  cancelMcpForm();
  await loadPluginsData();
}

export function closePluginsPanel() {
  state.pluginsPanelOpen = false;
}

// Shared cache: also used by the input box's "+" quick command palette so
// the skill/tool list only needs to be fetched once per page session.
let _loaded = false;
export async function loadPluginsData(force = false) {
  if (_loaded && !force) return;
  _loaded = true;
  const [tools, mcpServers, skills] = await Promise.all([fetchTools(), fetchMcpServers(), fetchSkills()]);
  if (tools.ok) state.pluginsData.tools = tools.data.tools || [];
  if (mcpServers.ok) state.pluginsData.mcpServers = mcpServers.data.servers || [];
  if (skills.ok) state.pluginsData.skills = skills.data.skills || [];
}

export function filteredTools() {
  const q = (state.pluginsSearch || '').toLowerCase().trim();
  if (!q) return state.pluginsData.tools;
  return state.pluginsData.tools.filter(t =>
    t.name.toLowerCase().includes(q) || (t.description || '').toLowerCase().includes(q));
}

export function filteredSkills() {
  const q = (state.pluginsSearch || '').toLowerCase().trim();
  if (!q) return state.pluginsData.skills;
  return state.pluginsData.skills.filter(s =>
    s.name.toLowerCase().includes(q) || (s.description || '').toLowerCase().includes(q));
}

export function showSkillForm(skill) {
  const f = state.skillForm;
  f.open = true;
  f.editingName = skill ? skill.name : null;
  f.name = skill ? skill.name : '';
  f.description = skill ? skill.description : '';
  f.trigger = skill ? (skill.trigger || '') : '';
  f.content = skill ? (skill.content || '') : '';
}

export function cancelSkillForm() {
  state.skillForm.open = false;
  state.skillForm.editingName = null;
}

export async function editSkill(name) {
  // List responses omit `content` for size — fetch the single skill for the full body.
  const { ok, data } = await fetchSkillDetail(name);
  if (!ok || !data) return;
  showSkillForm(data);
}

export async function saveSkillForm() {
  const f = state.skillForm;
  const name = f.name.trim();
  const description = f.description.trim();
  if (!name || !description) {
    showToast('Name and description are required', 2200);
    return;
  }
  const body = { description, content: f.content, trigger: f.trigger.trim() || null };
  const { ok, data } = f.editingName
    ? await updateSkillApi(f.editingName, body)
    : await createSkillApi({ name, ...body });
  if (!ok) { showToast(data?.detail || 'Save failed', 2500); return; }
  cancelSkillForm();
  await loadPluginsData(true);
  showToast(f.editingName ? 'Skill updated' : 'Skill created', 1500);
}

export function deleteSkill(name) {
  showConfirm(`Delete skill "${name}"?`, async () => {
    const { ok, data } = await deleteSkillApi(name);
    if (!ok) { showToast(data?.detail || 'Delete failed', 2500); return; }
    await loadPluginsData(true);
    showToast('Skill deleted', 1500);
  });
}

// ---- MCP servers: adds a tool server that the currently running web agent
// picks up on its next turn (see routes/plugins.py — writes mcp_config.json
// + invalidates the agent cache so DeepAgent's auto MCP loader re-runs) ----

export function showMcpForm() {
  const f = state.mcpForm;
  f.open = true;
  f.name = ''; f.kind = 'stdio'; f.command = ''; f.args = ''; f.url = ''; f.envRows = [];
}

export function cancelMcpForm() {
  state.mcpForm.open = false;
}

export function addMcpEnvRow() {
  state.mcpForm.envRows.push({ key: '', value: '' });
}

export function removeMcpEnvRow(i) {
  state.mcpForm.envRows.splice(i, 1);
}

export async function saveMcpForm() {
  const f = state.mcpForm;
  const name = f.name.trim();
  if (!name) { showToast('Server name is required', 2200); return; }
  const env = {};
  for (const row of f.envRows) {
    const k = (row.key || '').trim();
    if (k) env[k] = row.value;
  }
  const body = { name, env: Object.keys(env).length ? env : undefined };
  if (f.kind === 'stdio') {
    const command = f.command.trim();
    if (!command) { showToast('Command is required', 2200); return; }
    body.command = command;
    body.args = f.args.trim() ? f.args.trim().split(/\s+/) : [];
  } else {
    const url = f.url.trim();
    if (!url) { showToast('URL is required', 2200); return; }
    body.url = url;
  }
  const { ok, data } = await createMcpServerApi(body);
  if (!ok) { showToast(data?.detail || 'Save failed', 2500); return; }
  cancelMcpForm();
  await loadPluginsData(true);
  showToast('MCP server added — takes effect on the agent\'s next turn', 2500);
}

export function deleteMcpServer(name) {
  showConfirm(`Delete MCP server "${name}"?`, async () => {
    const { ok, data } = await deleteMcpServerApi(name);
    if (!ok) { showToast(data?.detail || 'Delete failed', 2500); return; }
    await loadPluginsData(true);
    showToast('MCP server deleted', 1500);
  });
}
