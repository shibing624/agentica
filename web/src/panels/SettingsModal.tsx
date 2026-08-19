import * as api from "../api";
import { browseDir, loadCronJobs, loadDirHistory, loadProfiles, loadProviders, loadStatus } from "../data";
import { IconClose, IconFolder } from "../icons";
import { agoStr, shortenPath } from "../lib/format";
import { unarchiveSession, deleteSession } from "../sessions";
import {
  askConfirm, emptyProfileForm, getState, saveSessions, setState, setTheme,
  showToast, useAppState, type ProfileForm,
} from "../store";
import { CronPanel } from "./CronPanel";

const TABS: Array<[string, string]> = [
  ["settings", "常规"],
  ["profiles", "Profile"],
  ["cron", "定时任务"],
  ["archived", "归档"],
];

function patch(p: Partial<ProfileForm>) {
  const f = getState().profileForm;
  if (f) setState({ profileForm: { ...f, ...p } });
}

/** Editing loads the full profile so tuning/aux fields aren't silently wiped by
 *  a save built from an empty form. api_key comes back masked, so it stays
 *  blank here — empty means "keep existing" on the PUT. */
async function editProfile(name: string) {
  const { ok, data } = await api.fetchProfileDetail(name);
  if (!ok || !data) { showToast("读取 profile 失败", 2500); return; }
  const d = data as any;
  const aux = d.auxiliary_model || {};
  setState({
    profileForm: {
      name: d.name || name,
      editing: true,
      model_provider: d.model_provider || "",
      model_name: d.model_name || "",
      base_url: d.base_url || "",
      api_key: "",
      reasoning_effort: d.reasoning_effort || "",
      max_tokens: d.max_tokens ? String(d.max_tokens) : "",
      context_window: d.context_window ? String(d.context_window) : "",
      temperature: d.temperature != null ? String(d.temperature) : "",
      top_p: d.top_p != null ? String(d.top_p) : "",
      aux_provider: aux.model_provider || "",
      aux_model: aux.model_name || "",
      aux_base_url: aux.base_url || "",
      aux_api_key: "",
      envRows: Object.entries(d.env || {}).map(([key, value]) => ({ key, value: String(value) })),
    },
  });
}

async function saveProfile() {
  const f = getState().profileForm;
  if (!f) return;
  const name = f.name.trim();
  if (!name || !f.model_provider.trim() || !f.model_name.trim()) {
    showToast("name / provider / model_name 是必填项", 2500);
    return;
  }
  const aux = (f.aux_provider.trim() || f.aux_model.trim()) ? {
    model_provider: f.aux_provider.trim(),
    model_name: f.aux_model.trim(),
    base_url: f.aux_base_url.trim(),
    api_key: f.aux_api_key || undefined,
  } : undefined;
  const env: Record<string, string> = {};
  for (const row of f.envRows) if (row.key.trim()) env[row.key.trim()] = row.value;
  const body = {
    name,
    model_provider: f.model_provider.trim(),
    model_name: f.model_name.trim(),
    base_url: f.base_url.trim(),
    api_key: f.api_key || undefined,
    reasoning_effort: f.reasoning_effort.trim() || undefined,
    max_tokens: parseInt(f.max_tokens, 10) || undefined,
    context_window: parseInt(f.context_window, 10) || undefined,
    temperature: f.temperature.trim() ? parseFloat(f.temperature) : undefined,
    top_p: f.top_p.trim() ? parseFloat(f.top_p) : undefined,
    auxiliary_model: aux,
    env: Object.keys(env).length ? env : undefined,
  };
  const res = f.editing ? await api.updateProfileApi(name, body) : await api.createProfileApi(body);
  if (!res.ok) { showToast((res.data as any)?.detail || "保存失败", 3000); return; }
  setState({ profileForm: null });
  await loadProfiles();
  if (f.editing && name === (getState().serverProfile || getState().profilesData.active)) {
    // The live agent still holds the old settings; re-switching reloads them.
    await api.switchProfileApi(name);
    await loadStatus();
  }
  showToast(f.editing ? "profile 已更新" : "profile 已创建");
}

export async function switchProfile(name: string) {
  const st = getState();
  if (!name || name === (st.serverProfile || st.profilesData.active)) {
    setState({ modelDDOpen: false });
    return;
  }
  const { ok, data } = await api.switchProfileApi(name);
  if (!ok) { showToast((data as any)?.detail || "切换失败", 3000); return; }
  setState({ modelDDOpen: false });
  await loadStatus();
  await loadProfiles();
  showToast("已切换到 " + name);
}

function removeProfile(name: string) {
  askConfirm({
    title: "删除 profile",
    msg: `“${name}” 将从 config.yaml 中移除。`,
    onOk: async () => {
      const { ok, data } = await api.deleteProfileApi(name);
      if (!ok) { showToast((data as any)?.detail || "删除失败", 3000); return; }
      await loadProfiles();
      showToast("profile 已删除");
    },
  });
}

async function toggleThinking(enabled: boolean) {
  const { ok, data } = await api.setThinkingApi(enabled);
  if (!ok) { showToast((data as any)?.detail || "设置失败", 3000); return; }
  setState({ serverThinking: (data as any)?.thinking || "" });
  showToast(enabled ? "已开启 thinking" : "已关闭 thinking");
}

async function applyBaseDir(dir: string) {
  const raw = dir.trim();
  if (!raw) return;
  const { ok, data } = await api.saveBaseDirApi(raw);
  if (!ok) { showToast((data as any)?.detail || "目录设置失败", 3500); return; }
  await loadStatus();
  await loadDirHistory();
  const st = getState();
  if (st.curSess && st.sessions[st.curSess]) {
    st.sessions[st.curSess].dir = st.serverDir || raw;
    saveSessions();
  }
  showToast("工作目录已更新");
}

export function SettingsModal() {
  const s = useAppState();
  const archived = Object.entries(s.sessions).filter(([, sess]) => sess.archived);
  const close = () => setState({ settingsModal: { open: false }, profileForm: null, cronForm: null });
  return (
    <div className="settings-overlay open" onClick={close}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-head">
          <h3>
            {TABS.find((t) => t[0] === s.settingsTab)?.[1] || "设置"}
            <span className="settings-ver">{s.serverVersion ? "Agentica v" + s.serverVersion : ""}</span>
          </h3>
          <div className="settings-head-actions">
            {s.settingsTab === "profiles" && (
              <button className="settings-new-btn" onClick={() => setState({ profileForm: emptyProfileForm() })}>+ 新建</button>
            )}
            <button className="ib" onClick={close}><IconClose /></button>
          </div>
        </div>
        <div className="plugins-tabs">
          {TABS.map(([id, label]) => (
            <button key={id} className={"plugins-tab" + (s.settingsTab === id ? " active" : "")}
                    onClick={() => setState({ settingsTab: id })}>{label}</button>
          ))}
        </div>
        <div className="settings-body">
          {s.settingsTab === "settings" && <GeneralTab />}
          {s.settingsTab === "profiles" && <ProfilesTab />}
          {s.settingsTab === "cron" && <CronPanel />}
          {s.settingsTab === "archived" && (
            <div className="settings-list">
              {!archived.length && <div className="settings-empty">没有归档会话</div>}
              {archived.map(([id, sess]) => (
                <div key={id} className="settings-item">
                  <div className="settings-item-head">
                    <span className="settings-name">{sess.title}</span>
                    <div className="settings-item-actions">
                      <button className="cron-act" onClick={() => unarchiveSession(id)}>恢复</button>
                      <button className="cron-act danger" onClick={() => deleteSession(id)}>删除</button>
                    </div>
                  </div>
                  <div className="settings-item-meta">{shortenPath(sess.dir)} · {agoStr(sess.ts)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function GeneralTab() {
  const s = useAppState();
  const thinkingOn = !!s.serverThinking;
  return (
    <div className="settings-list">
      <div className="settings-block">
        <div className="settings-block-title">主题</div>
        <div className="pf-toggle">
          {([["auto", "跟随系统"], ["light", "浅色"], ["dark", "深色"]] as const).map(([v, label]) => (
            <button key={v} className={"pf-toggle-btn" + (s.theme === v ? " active" : "")} onClick={() => setTheme(v)}>{label}</button>
          ))}
        </div>
      </div>

      <div className="settings-block">
        <div className="settings-block-title">思考过程（thinking）</div>
        <label className="pf-check">
          <input type="checkbox" checked={thinkingOn} onChange={(e) => void toggleThinking(e.target.checked)} />
          让模型输出推理过程（仅支持 thinking 的模型有效，切换后下一轮生效）
        </label>
      </div>

      <div className="settings-block">
        <div className="settings-block-title">默认工作目录</div>
        <div className="dm-row">
          <input value={s.dirModal.value || s.serverDir} onChange={(e) => setState({ dirModal: { ...s.dirModal, value: e.target.value } })} />
          <button className="cron-act" onClick={() => void browseDir(s.dirModal.value || s.serverDir)}>浏览</button>
          <button className="dp-btn primary" onClick={() => void applyBaseDir(s.dirModal.value || s.serverDir)}>应用</button>
        </div>
        {!!s.dirHistory.length && (
          <div className="dir-history">
            {s.dirHistory.map((d) => (
              <button className="dir-hist-item" key={d} onClick={() => setState({ dirModal: { ...s.dirModal, value: d } })} title={d}>
                <IconFolder /> {shortenPath(d)}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="settings-block">
        <div className="settings-block-title">当前配置</div>
        <div className="settings-item-meta">
          Profile: {s.serverProfile || "default"} · {s.serverProvider}/{s.serverModelName || s.serverModel}
          {s.serverReasoningEffort ? ` · effort=${s.serverReasoningEffort}` : ""}
          {s.serverContextWindow ? ` · 上下文 ${s.serverContextWindow}` : ""}
        </div>
        <div className="config-path-row">
          <code className="config-path-val">{s.serverConfigPath || "~/.agentica/config.yaml"}</code>
          <button className="cron-act" onClick={() => {
            const p = s.serverConfigPath;
            if (p) { void navigator.clipboard.writeText(p); showToast("已复制"); }
          }}>复制</button>
        </div>
      </div>
    </div>
  );
}

function ProfilesTab() {
  const s = useAppState();
  const active = s.serverProfile || s.profilesData.active;
  return (
    <>
      {s.profileForm && <ProfileForm />}
      <div className="settings-list">
        {!(s.profilesData.profiles || []).length && <div className="settings-empty">还没有 profile，点“+ 新建”创建一个。</div>}
        {(s.profilesData.profiles || []).map((p: any) => (
          <div key={p.name} className={"settings-item" + (p.name === active ? " active" : "")}>
            <div className="settings-item-head">
              <span className="settings-name">
                {p.name}{p.name === active && <span className="settings-active"> ● 使用中</span>}
              </span>
              <div className="settings-item-actions">
                {p.name !== active && <button className="cron-act" onClick={() => void switchProfile(p.name)}>切换</button>}
                <button className="cron-act" onClick={() => void editProfile(p.name)}>编辑</button>
                <button className="cron-act danger" onClick={() => removeProfile(p.name)}>删除</button>
              </div>
            </div>
            <div className="settings-item-meta">
              {p.model_provider || "?"}/{p.model_name || "?"}{p.base_url ? " · " + p.base_url : ""}
              {p.has_api_key ? ` · key ${p.api_key_masked}` : " · 无 key"}
            </div>
            {p.tuning?.length ? <div className="settings-item-tuning">{p.tuning.join(" · ")}</div> : null}
            {p.auxiliary ? <div className="settings-item-aux">aux: {p.auxiliary.model_provider || "?"}/{p.auxiliary.model_name || "?"}</div> : null}
          </div>
        ))}
      </div>
    </>
  );
}

function ProfileForm() {
  const s = useAppState();
  const f = s.profileForm!;
  return (
    <div className="settings-form">
      <h4>{f.editing ? `编辑 profile：${f.name}` : "新建 profile"}</h4>
      <input className="pf-input" placeholder="profile 名（例如 default）" value={f.name} disabled={f.editing}
             onChange={(e) => patch({ name: e.target.value })} />
      <div className="pf-section">主模型</div>
      <input className="pf-input" list="provider-list" placeholder="provider（ark / deepseek / openai …）"
             value={f.model_provider} onChange={(e) => patch({ model_provider: e.target.value })} />
      <datalist id="provider-list">{s.providers.map((p) => <option key={p} value={p} />)}</datalist>
      <input className="pf-input" placeholder="model_name" value={f.model_name} onChange={(e) => patch({ model_name: e.target.value })} />
      <input className="pf-input" placeholder="base_url（可选）" value={f.base_url} onChange={(e) => patch({ base_url: e.target.value })} />
      <input className="pf-input" placeholder={f.editing ? "api_key（留空表示保持不变）" : "api_key"}
             value={f.api_key} onChange={(e) => patch({ api_key: e.target.value })} />
      <div className="pf-section">调参（可选）</div>
      <div className="pf-row">
        <input className="pf-input" placeholder="reasoning_effort（low/medium/high）" value={f.reasoning_effort} onChange={(e) => patch({ reasoning_effort: e.target.value })} />
        <input className="pf-input" type="number" placeholder="max_tokens" value={f.max_tokens} onChange={(e) => patch({ max_tokens: e.target.value })} />
      </div>
      <div className="pf-row">
        <input className="pf-input" type="number" placeholder="context_window" value={f.context_window} onChange={(e) => patch({ context_window: e.target.value })} />
        <input className="pf-input" type="number" step="0.1" placeholder="temperature" value={f.temperature} onChange={(e) => patch({ temperature: e.target.value })} />
      </div>
      <input className="pf-input" type="number" step="0.05" placeholder="top_p" value={f.top_p} onChange={(e) => patch({ top_p: e.target.value })} />
      <div className="pf-section">辅助模型（可选，用于子代理等廉价调用）</div>
      <input className="pf-input" placeholder="aux provider" value={f.aux_provider} onChange={(e) => patch({ aux_provider: e.target.value })} />
      <input className="pf-input" placeholder="aux model_name" value={f.aux_model} onChange={(e) => patch({ aux_model: e.target.value })} />
      <input className="pf-input" placeholder="aux base_url（可选）" value={f.aux_base_url} onChange={(e) => patch({ aux_base_url: e.target.value })} />
      <input className="pf-input" placeholder="aux api_key（留空保持不变）" value={f.aux_api_key} onChange={(e) => patch({ aux_api_key: e.target.value })} />
      <div className="pf-section">env 块（可选）</div>
      <div className="pf-env">
        {f.envRows.map((row, i) => (
          <div className="pf-env-row" key={i}>
            <input className="pf-input pf-env-k" placeholder="KEY" value={row.key} onChange={(e) => {
              const envRows = f.envRows.slice();
              envRows[i] = { ...row, key: e.target.value };
              patch({ envRows });
            }} />
            <input className="pf-input pf-env-v" placeholder="value" value={row.value} onChange={(e) => {
              const envRows = f.envRows.slice();
              envRows[i] = { ...row, value: e.target.value };
              patch({ envRows });
            }} />
            <button className="pf-env-del" onClick={() => patch({ envRows: f.envRows.filter((_, j) => j !== i) })}><IconClose /></button>
          </div>
        ))}
      </div>
      <button className="pf-add-env" onClick={() => patch({ envRows: [...f.envRows, { key: "", value: "" }] })}>+ 添加变量</button>
      <div className="pf-actions">
        <button className="dp-btn" onClick={() => setState({ profileForm: null })}>取消</button>
        <button className="dp-btn primary" onClick={() => void saveProfile()}>{f.editing ? "保存" : "创建"}</button>
      </div>
    </div>
  );
}

/** Loaded lazily so the settings modal always opens with current server state
 *  rather than whatever the chat page fetched at boot. */
export async function primeSettings(tab: string) {
  setState({ settingsModal: { open: true }, settingsTab: tab });
  await Promise.all([loadStatus(), loadProfiles(), loadProviders(), loadDirHistory(), loadCronJobs()]);
}
