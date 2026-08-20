import * as api from "../api";
import { loadPlugins } from "../data";
import { IconClose } from "../icons";
import {
  askConfirm, emptyMcpForm, emptySkillForm, getState, setState, showToast,
  useAppState, type McpForm, type SkillForm,
} from "../store";

const TABS: Array<[string, string]> = [["skills", "技能"], ["tools", "工具"], ["mcp", "MCP"]];

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
  if (!ok || !data) { showToast("读取技能失败", 2500); return; }
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
  if (!f.name.trim() || !f.description.trim()) {
    showToast("名称和描述是必填项", 2500);
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
  if (!res.ok) { showToast((res.data as any)?.detail || "保存失败", 3000); return; }
  setState({ skillForm: null });
  await loadPlugins();
  showToast(f.editing ? "技能已更新，下一轮生效" : "技能已创建，下一轮生效");
}

function removeSkill(name: string) {
  askConfirm({
    title: "删除技能",
    msg: `技能 “${name}” 的目录将被删除。`,
    onOk: async () => {
      const { ok, data } = await api.deleteSkillApi(name);
      if (!ok) { showToast((data as any)?.detail || "删除失败", 3000); return; }
      await loadPlugins();
      showToast("技能已删除");
    },
  });
}

async function saveMcp() {
  const f = getState().mcpForm;
  if (!f) return;
  const name = f.name.trim();
  if (!name) { showToast("名称是必填项", 2500); return; }
  if (f.kind === "stdio" && !f.command.trim()) { showToast("stdio 需要 command", 2500); return; }
  if (f.kind === "sse" && !f.url.trim()) { showToast("sse/http 需要 url", 2500); return; }
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
  if (!ok) { showToast((data as any)?.detail || "添加失败", 3000); return; }
  setState({ mcpForm: null });
  await loadPlugins();
  showToast("MCP server 已添加，agent 下一轮生效");
}

function removeMcp(name: string) {
  askConfirm({
    title: "移除 MCP server",
    msg: `“${name}” 将从 mcp_config.json 中移除。`,
    okLabel: "移除",
    onOk: async () => {
      const { ok, data } = await api.deleteMcpServerApi(name);
      if (!ok) { showToast((data as any)?.detail || "移除失败", 3000); return; }
      await loadPlugins();
      showToast("已移除");
    },
  });
}

export function PluginsPanel() {
  const s = useAppState();
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
          <h3>插件</h3>
          <div className="settings-head-actions">
            <button className="dp-btn" onClick={() => void loadPlugins()}>刷新</button>
            {s.pluginsTab === "skills" && (
              <button className="dp-btn primary" onClick={() => setState({ skillForm: emptySkillForm() })}>+ 新建技能</button>
            )}
            {s.pluginsTab === "mcp" && (
              <button className="dp-btn primary" onClick={() => setState({ mcpForm: emptyMcpForm() })}>+ 添加 server</button>
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
              {label}
              <span className="plugin-badge">
                {id === "skills" ? skills.length : id === "tools" ? tools.length : servers.length}
              </span>
            </button>
          ))}
          <input
            className="plugins-search"
            placeholder="搜索名称或描述"
            value={s.pluginsSearch}
            onChange={(e) => setState({ pluginsSearch: e.target.value })}
          />
        </div>
        <div className="plugins-body">
          {s.pluginsTab === "skills" && (
            <>
              {s.skillForm && <SkillFormView />}
              <div className="plugin-list">
                {!skills.length && <div className="settings-empty">没有匹配的技能</div>}
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
                          <button className="cron-act" onClick={() => void editSkill(t.name)}>编辑</button>
                          <button className="cron-act danger" onClick={() => removeSkill(t.name)}>删除</button>
                        </>
                      ) : <span className="plugin-readonly">内置</span>}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {s.pluginsTab === "tools" && (
            <div className="plugin-list">
              {!tools.length && <div className="settings-empty">没有匹配的工具</div>}
              {tools.map((t: any) => (
                <div key={t.name} className="plugin-row">
                  <div className="plugin-row-main">
                    <div className="plugin-row-title">
                      {t.name}
                      {t.tool_group && <span className="plugin-loc">{t.tool_group}</span>}
                      {t.is_read_only && <span className="plugin-loc ro">只读</span>}
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
                {!servers.length && <div className="settings-empty">没有配置 MCP server</div>}
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
                      <button className="cron-act danger" onClick={() => removeMcp(t.name)}>移除</button>
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
  return (
    <div className="settings-form skill-form">
      <h4>{f.editing ? `编辑技能：${f.name}` : "新建技能"}</h4>
      <input className="pf-input" placeholder="技能名（英文，作目录名）" value={f.name} disabled={f.editing}
             onChange={(e) => patchSkill({ name: e.target.value })} />
      <input className="pf-input" placeholder="描述：什么时候该用这个技能" value={f.description}
             onChange={(e) => patchSkill({ description: e.target.value })} />
      <input className="pf-input" placeholder="触发词（可选，例如 /review）" value={f.trigger}
             onChange={(e) => patchSkill({ trigger: e.target.value })} />
      <textarea className="pf-input pf-textarea" rows={10} placeholder="SKILL.md 正文" value={f.content}
                onChange={(e) => patchSkill({ content: e.target.value })} />
      <div className="pf-actions">
        <button className="dp-btn" onClick={() => setState({ skillForm: null })}>取消</button>
        <button className="dp-btn primary" onClick={() => void saveSkill()}>{f.editing ? "保存" : "创建"}</button>
      </div>
    </div>
  );
}

function McpFormView() {
  const f = useAppState().mcpForm!;
  return (
    <div className="settings-form mcp-form">
      <h4>添加 MCP server</h4>
      <input className="pf-input" placeholder="名称" value={f.name} onChange={(e) => patchMcp({ name: e.target.value })} />
      <div className="pf-toggle">
        {(["stdio", "sse"] as const).map((k) => (
          <button key={k} className={"pf-toggle-btn" + (f.kind === k ? " active" : "")} onClick={() => patchMcp({ kind: k })}>
            {k === "stdio" ? "stdio（本地命令）" : "sse / http（远程）"}
          </button>
        ))}
      </div>
      {f.kind === "stdio" ? (
        <>
          <input className="pf-input" placeholder="command，例如 npx" value={f.command} onChange={(e) => patchMcp({ command: e.target.value })} />
          <input className="pf-input" placeholder="args，空格分隔" value={f.args} onChange={(e) => patchMcp({ args: e.target.value })} />
        </>
      ) : (
        <input className="pf-input" placeholder="url" value={f.url} onChange={(e) => patchMcp({ url: e.target.value })} />
      )}
      <div className="pf-section">环境变量（可选）</div>
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
      <button className="pf-add-env" onClick={() => patchMcp({ envRows: [...f.envRows, { key: "", value: "" }] })}>+ 添加变量</button>
      <div className="pf-actions">
        <button className="dp-btn" onClick={() => setState({ mcpForm: null })}>取消</button>
        <button className="dp-btn primary" onClick={() => void saveMcp()}>添加</button>
      </div>
    </div>
  );
}
