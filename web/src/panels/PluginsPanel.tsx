import * as api from "../api";
import { loadPlugins } from "../data";
import { getStrings, useStrings, type Strings } from "../i18n";
import { IconClose } from "../icons";
import {
  askConfirm, emptyMcpForm, emptySkillForm, getState, setState, showToast,
  useAppState, type McpForm, type SkillForm,
} from "../store";

const TABS: Array<[string, (S: Strings) => string]> = [
  ["skills", (S) => S.plugins.tabSkills],
  ["tools", (S) => S.plugins.tabTools],
  ["mcp", (S) => S.plugins.tabMcp],
];

function patchSkill(p: Partial<SkillForm>) {
  const f = getState().skillForm;
  if (f) setState({ skillForm: { ...f, ...p } });
}

function patchMcp(p: Partial<McpForm>) {
  const f = getState().mcpForm;
  if (f) setState({ mcpForm: { ...f, ...p } });
}

async function editSkill(name: string) {
  const { ok, data } = await api.fetchSkillDetail(name);
  if (!ok || !data) { showToast(getStrings().plugins.readSkillFailed, 2500); return; }
  setState({
    skillForm: {
      name: (data as any).name || name,
      editing: true,
      description: (data as any).description || "",
      trigger: (data as any).trigger || "",
      content: (data as any).content || "",
    },
  });
}

async function saveSkill() {
  const f = getState().skillForm;
  if (!f) return;
  const S = getStrings();
  if (!f.name.trim() || !f.description.trim()) {
    showToast(S.plugins.skillRequired, 2500);
    return;
  }
  const body = {
    name: f.name.trim(),
    description: f.description.trim(),
    trigger: f.trigger.trim() || undefined,
    content: f.content,
  };
  const res = f.editing
    ? await api.updateSkillApi(f.name.trim(), { description: body.description, trigger: body.trigger, content: body.content })
    : await api.createSkillApi(body);
  if (!res.ok) { showToast((res.data as any)?.detail || S.common.saveFailed, 3000); return; }
  setState({ skillForm: null });
  await loadPlugins();
  showToast(f.editing ? S.plugins.skillUpdated : S.plugins.skillCreated);
}

function removeSkill(name: string) {
  const S = getStrings();
  askConfirm({
    title: S.plugins.removeSkill,
    msg: S.plugins.removeSkillMsg(name),
    onOk: async () => {
      const { ok, data } = await api.deleteSkillApi(name);
      if (!ok) { showToast((data as any)?.detail || S.common.deleteFailed, 3000); return; }
      await loadPlugins();
      showToast(S.plugins.skillDeleted);
    },
  });
}

async function saveMcp() {
  const f = getState().mcpForm;
  if (!f) return;
  const S = getStrings();
  const name = f.name.trim();
  if (!name) { showToast(S.plugins.mcpNameRequired, 2500); return; }
  if (f.kind === "stdio" && !f.command.trim()) { showToast(S.plugins.mcpNeedsCommand, 2500); return; }
  if (f.kind === "sse" && !f.url.trim()) { showToast(S.plugins.mcpNeedsUrl, 2500); return; }
  const env: Record<string, string> = {};
  for (const row of f.envRows) if (row.key.trim()) env[row.key.trim()] = row.value;
  const body: Record<string, unknown> = { name };
  if (f.kind === "stdio") {
    body.command = f.command.trim();
    body.args = f.args.trim() ? f.args.trim().split(/\s+/) : [];
  } else {
    body.url = f.url.trim();
  }
  if (Object.keys(env).length) body.env = env;
  const { ok, data } = await api.createMcpServerApi(body);
  if (!ok) { showToast((data as any)?.detail || S.plugins.addFailed, 3000); return; }
  setState({ mcpForm: null });
  await loadPlugins();
  showToast(S.plugins.mcpAdded);
}

function removeMcp(name: string) {
  const S = getStrings();
  askConfirm({
    title: S.plugins.removeMcp,
    msg: S.plugins.removeMcpMsg(name),
    okLabel: S.common.remove,
    onOk: async () => {
      const { ok, data } = await api.deleteMcpServerApi(name);
      if (!ok) { showToast((data as any)?.detail || S.plugins.removeFailed, 3000); return; }
      await loadPlugins();
      showToast(S.plugins.removed);
    },
  });
}

export function PluginsPanel() {
  const s = useAppState();
  const S = useStrings();
  const q = s.pluginsSearch.trim().toLowerCase();
  const match = (t: any) =>
    !q || (t.name || "").toLowerCase().includes(q) || (t.description || "").toLowerCase().includes(q);
  const skills = (s.pluginsData.skills || []).filter(match);
  const tools = (s.pluginsData.tools || []).filter(match);
  const servers = (s.pluginsData.mcpServers || []).filter(match);

  return (
    <div className="plugins-overlay open" onClick={() => setState({ pluginsPanelOpen: false })}>
      <div className="plugins-modal" onClick={(e) => e.stopPropagation()}>
        <div className="plugins-head">
          <h3>{S.plugins.title}</h3>
          <div className="settings-head-actions">
            <button className="dp-btn" onClick={() => void loadPlugins()}>{S.common.refresh}</button>
            {s.pluginsTab === "skills" && (
              <button className="dp-btn primary" onClick={() => setState({ skillForm: emptySkillForm() })}>{S.plugins.newSkill}</button>
            )}
            {s.pluginsTab === "mcp" && (
              <button className="dp-btn primary" onClick={() => setState({ mcpForm: emptyMcpForm() })}>{S.plugins.addServer}</button>
            )}
            <button className="ib" onClick={() => setState({ pluginsPanelOpen: false })}><IconClose /></button>
          </div>
        </div>
        <div className="plugins-tabs">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              className={"plugins-tab" + (s.pluginsTab === id ? " active" : "")}
              onClick={() => setState({ pluginsTab: id })}
            >
              {label(S)}
              <span className="plugin-badge">
                {id === "skills" ? skills.length : id === "tools" ? tools.length : servers.length}
              </span>
            </button>
          ))}
          <input
            className="plugins-search"
            placeholder={S.plugins.searchPlaceholder}
            value={s.pluginsSearch}
            onChange={(e) => setState({ pluginsSearch: e.target.value })}
          />
        </div>
        <div className="plugins-body">
          {s.pluginsTab === "skills" && (
            <>
              {s.skillForm && <SkillFormView />}
              <div className="plugin-list">
                {!skills.length && <div className="settings-empty">{S.plugins.noSkills}</div>}
                {skills.map((t: any) => (
                  <div key={t.name} className="plugin-row">
                    <div className="plugin-row-main">
                      <div className="plugin-row-title">
                        {t.name}
                        {t.trigger && <code className="plugin-trigger">{t.trigger}</code>}
                        <span className="plugin-loc">{t.location}</span>
                      </div>
                      <div className="plugin-row-desc">{t.description}</div>
                    </div>
                    <div className="plugin-row-actions">
                      {t.editable ? (
                        <>
                          <button className="cron-act" onClick={() => void editSkill(t.name)}>{S.common.edit}</button>
                          <button className="cron-act danger" onClick={() => removeSkill(t.name)}>{S.common.delete}</button>
                        </>
                      ) : <span className="plugin-readonly">{S.plugins.builtin}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {s.pluginsTab === "tools" && (
            <div className="plugin-list">
              {!tools.length && <div className="settings-empty">{S.plugins.noTools}</div>}
              {tools.map((t: any) => (
                <div key={t.name} className="plugin-row">
                  <div className="plugin-row-main">
                    <div className="plugin-row-title">
                      {t.name}
                      {t.tool_group && <span className="plugin-loc">{t.tool_group}</span>}
                      {t.is_read_only && <span className="plugin-loc ro">{S.plugins.readOnly}</span>}
                    </div>
                    <div className="plugin-row-desc">{t.description}</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {s.pluginsTab === "mcp" && (
            <>
              {s.mcpForm && <McpFormView />}
              <div className="plugin-list">
                {!servers.length && <div className="settings-empty">{S.plugins.noServers}</div>}
                {servers.map((t: any) => (
                  <div key={t.name} className="plugin-row">
                    <div className="plugin-row-main">
                      <div className="plugin-row-title">{t.name}<span className="plugin-loc">{t.type}</span></div>
                      <div className="plugin-row-desc">
                        {t.command ? `${t.command} ${(t.args || []).join(" ")}` : t.url}
                        {(t.env_keys || []).length ? ` · env: ${(t.env_keys || []).join(", ")}` : ""}
                      </div>
                    </div>
                    <div className="plugin-row-actions">
                      <button className="cron-act danger" onClick={() => removeMcp(t.name)}>{S.common.remove}</button>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function SkillFormView() {
  const f = useAppState().skillForm!;
  const S = useStrings();
  return (
    <div className="settings-form skill-form">
      <h4>{f.editing ? S.plugins.editSkill(f.name) : S.plugins.newSkillTitle}</h4>
      <input className="pf-input" placeholder={S.plugins.skillNamePlaceholder} value={f.name} disabled={f.editing}
             onChange={(e) => patchSkill({ name: e.target.value })} />
      <input className="pf-input" placeholder={S.plugins.skillDescPlaceholder} value={f.description}
             onChange={(e) => patchSkill({ description: e.target.value })} />
      <input className="pf-input" placeholder={S.plugins.skillTriggerPlaceholder} value={f.trigger}
             onChange={(e) => patchSkill({ trigger: e.target.value })} />
      <textarea className="pf-input pf-textarea" rows={10} placeholder={S.plugins.skillBodyPlaceholder} value={f.content}
                onChange={(e) => patchSkill({ content: e.target.value })} />
      <div className="pf-actions">
        <button className="dp-btn" onClick={() => setState({ skillForm: null })}>{S.common.cancel}</button>
        <button className="dp-btn primary" onClick={() => void saveSkill()}>{f.editing ? S.common.save : S.common.create}</button>
      </div>
    </div>
  );
}

function McpFormView() {
  const f = useAppState().mcpForm!;
  const S = useStrings();
  return (
    <div className="settings-form mcp-form">
      <h4>{S.plugins.addMcp}</h4>
      <input className="pf-input" placeholder={S.plugins.mcpName} value={f.name} onChange={(e) => patchMcp({ name: e.target.value })} />
      <div className="pf-toggle">
        {(["stdio", "sse"] as const).map((k) => (
          <button key={k} className={"pf-toggle-btn" + (f.kind === k ? " active" : "")} onClick={() => patchMcp({ kind: k })}>
            {k === "stdio" ? S.plugins.mcpStdio : S.plugins.mcpSse}
          </button>
        ))}
      </div>
      {f.kind === "stdio" ? (
        <>
          <input className="pf-input" placeholder={S.plugins.mcpCommandPlaceholder} value={f.command} onChange={(e) => patchMcp({ command: e.target.value })} />
          <input className="pf-input" placeholder={S.plugins.mcpArgsPlaceholder} value={f.args} onChange={(e) => patchMcp({ args: e.target.value })} />
        </>
      ) : (
        <input className="pf-input" placeholder="url" value={f.url} onChange={(e) => patchMcp({ url: e.target.value })} />
      )}
      <div className="pf-section">{S.plugins.mcpEnv}</div>
      {f.envRows.map((row, i) => (
        <div className="pf-env-row" key={i}>
          <input className="pf-input pf-env-k" placeholder="KEY" value={row.key} onChange={(e) => {
            const envRows = f.envRows.slice();
            envRows[i] = { ...row, key: e.target.value };
            patchMcp({ envRows });
          }} />
          <input className="pf-input pf-env-v" placeholder="value" value={row.value} onChange={(e) => {
            const envRows = f.envRows.slice();
            envRows[i] = { ...row, value: e.target.value };
            patchMcp({ envRows });
          }} />
          <button className="pf-env-del" onClick={() => patchMcp({ envRows: f.envRows.filter((_, j) => j !== i) })}><IconClose /></button>
        </div>
      ))}
      <button className="pf-add-env" onClick={() => patchMcp({ envRows: [...f.envRows, { key: "", value: "" }] })}>{S.common.addVar}</button>
      <div className="pf-actions">
        <button className="dp-btn" onClick={() => setState({ mcpForm: null })}>{S.common.cancel}</button>
        <button className="dp-btn primary" onClick={() => void saveMcp()}>{S.common.add}</button>
      </div>
    </div>
  );
}
