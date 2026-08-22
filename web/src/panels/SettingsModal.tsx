import { useEffect, useState } from "react";
import * as api from "../api";
import {
  loadAuthStatus, loadCronJobs, loadDirHistory, loadProfiles, loadProviders,
  loadStatus,
} from "../data";
import { openPasswordDialog } from "../components/ChangePasswordDialog";
import { ChatMarkdown } from "../components/ChatMarkdown";
import { DirPicker } from "../components/DirPicker";
import { getStrings, LANGS, setLang, useStrings, type Strings } from "../i18n";
import { IconClose, IconExpand } from "../icons";
import { agoStr, lastQueryTs, shortenPath } from "../lib/format";
import { unarchiveSession, deleteSession } from "../sessions";
import {
  askConfirm, emptyProfileForm, getState, saveSessions, setState, setTheme,
  showToast, useAppState, type ProfileForm,
} from "../store";
import { CronPanel } from "./CronPanel";
import { AssistantPanel } from "../pages/AssistantPage";

/** Tab ids are stable (they are stored in `settingsTab`); only the label moves.
 *
 *  Account management is a page (`/users`), not a tab: this modal configures
 *  the machine (theme, models, working directory) plus this account's cron
 *  jobs and archived chats. Web URL and IM connectors are the 个人助理 tab
 *  (same panel as `/assistant`).
 */
const TABS: Array<[string, (S: Strings) => string]> = [
  ["settings", (S) => S.settings.tabGeneral],
  ["assistant", (S) => S.nav.assistant],
  ["memory", (S) => S.nav.memory],
  ["profiles", (S) => S.nav.profile],
  ["cron", (S) => S.settings.tabCron],
  ["archived", (S) => S.settings.tabArchived],
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
  if (!ok || !data) { showToast(getStrings().settings.readFailed, 2500); return; }
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
      compact_token_limit: d.compact_token_limit ? String(d.compact_token_limit) : "",
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
  const S = getStrings();
  const name = f.name.trim();
  if (!name || !f.model_provider.trim() || !f.model_name.trim()) {
    showToast(S.settings.required, 2500);
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
    compact_token_limit: parseInt(f.compact_token_limit, 10) || undefined,
    temperature: f.temperature.trim() ? parseFloat(f.temperature) : undefined,
    top_p: f.top_p.trim() ? parseFloat(f.top_p) : undefined,
    auxiliary_model: aux,
    env: Object.keys(env).length ? env : undefined,
  };
  const res = f.editing ? await api.updateProfileApi(name, body) : await api.createProfileApi(body);
  if (!res.ok) { showToast((res.data as any)?.detail || S.common.saveFailed, 3000); return; }
  setState({ profileForm: null });
  await loadProfiles();
  if (f.editing && name === (getState().serverProfile || getState().profilesData.active)) {
    // The live agent still holds the old settings; re-switching reloads them.
    await api.switchProfileApi(name);
    await loadStatus();
  }
  showToast(f.editing ? S.settings.profileUpdated : S.settings.profileCreated);
}

export async function switchProfile(name: string) {
  const st = getState();
  if (!name || name === (st.serverProfile || st.profilesData.active)) {
    setState({ modelDDOpen: false });
    return;
  }
  const S = getStrings();
  const { ok, data } = await api.switchProfileApi(name);
  if (!ok) { showToast((data as any)?.detail || S.settings.switchFailed, 3000); return; }
  setState({ modelDDOpen: false });
  await loadStatus();
  await loadProfiles();
  showToast(S.settings.switchedTo(name));
}

function removeProfile(name: string) {
  const S = getStrings();
  askConfirm({
    title: S.settings.removeProfile,
    msg: S.settings.removeProfileMsg(name),
    onOk: async () => {
      const { ok, data } = await api.deleteProfileApi(name);
      if (!ok) { showToast((data as any)?.detail || S.common.deleteFailed, 3000); return; }
      await loadProfiles();
      showToast(S.settings.profileDeleted);
    },
  });
}

async function applyBaseDir(dir: string) {
  const raw = dir.trim();
  if (!raw) return;
  const S = getStrings();
  const previous = getState().serverDir;
  const { ok } = await api.saveBaseDirApi(raw);
  if (!ok) {
    showToast(S.dir.invalid, 3500);
    setState({ dirModal: { ...getState().dirModal, value: previous } });
    return;
  }
  await loadStatus();
  await loadDirHistory();
  const st = getState();
  if (st.curSess && st.sessions[st.curSess]) {
    st.sessions[st.curSess].dir = st.serverDir || raw;
    saveSessions();
  }
  showToast(S.dir.updated);
}

/** Change the web password.
 *
 *  Worth having here as well as in `agentica-gateway --set-password`: that one
 *  is the only way in on a headless box, this one is the only way for somebody
 *  who is not at that terminal — which includes every desktop-shell user, since
 *  the shell signs itself in and shows no terminal at all. */
function AccessBlock() {
  const s = useAppState();
  const S = useStrings();
  return (
    <div className="settings-block">
      <div className="settings-block-title">{S.settings.access}</div>
      <div className="settings-item-meta">
        {S.settings.accessAccount(s.accountId)}{" "}
        {s.passwordIsInitial
          ? S.settings.accessInitial
          : s.passwordSet ? S.settings.accessSet : S.settings.accessNone}
      </div>
      <div className="pf-row">
        <button className="dp-btn primary" onClick={() => openPasswordDialog()}>
          {s.passwordSet ? S.settings.changePassword : S.settings.setPassword}
        </button>
        {s.passwordSet && (
          <button className="dp-btn" onClick={() => void logout()}>{S.settings.logout}</button>
        )}
      </div>
    </div>
  );
}

async function logout() {
  await api.logoutApi();
  window.location.href = "/login";
}

export function SettingsModal() {
  const s = useAppState();
  const S = useStrings();
  const archived = Object.entries(s.sessions).filter(([, sess]) => sess.archived);
  const close = () => setState({ settingsModal: { open: false }, profileForm: null, cronForm: null });
  return (
    <div className="settings-overlay open" onClick={close}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-head">
          <h3>
            {TABS.find((t) => t[0] === s.settingsTab)?.[1](S) || S.settings.title}
            <span className="settings-ver">{s.serverVersion ? "Agentica v" + s.serverVersion : ""}</span>
          </h3>
          <div className="settings-head-actions">
            {s.settingsTab === "profiles" && (
              <button className="settings-new-btn" onClick={() => setState({ profileForm: emptyProfileForm() })}>{S.common.newItem}</button>
            )}
            <button className="ib" onClick={close}><IconClose /></button>
          </div>
        </div>
        <div className="plugins-tabs">
          {TABS.map(([id, label]) => (
            <button key={id} className={"plugins-tab" + (s.settingsTab === id ? " active" : "")}
                    onClick={() => setState({ settingsTab: id })}>{label(S)}</button>
          ))}
        </div>
        <div className="settings-body">
          {s.settingsTab === "settings" && <GeneralTab />}
          {s.settingsTab === "assistant" && <div className="asst-body"><AssistantPanel /></div>}
          {s.settingsTab === "memory" && <MemoryTab />}
          {s.settingsTab === "profiles" && <ProfilesTab />}
          {s.settingsTab === "cron" && <CronPanel />}
          {s.settingsTab === "archived" && (
            <div className="settings-list">
              {!archived.length && <div className="settings-empty">{S.settings.noArchived}</div>}
              {archived.map(([id, sess]) => (
                <div key={id} className="settings-item">
                  <div className="settings-item-head">
                    <span className="settings-name">{sess.title}</span>
                    <div className="settings-item-actions">
                      <button className="cron-act" onClick={() => unarchiveSession(id)}>{S.common.restore}</button>
                      <button className="cron-act danger" onClick={() => deleteSession(id)}>{S.common.delete}</button>
                    </div>
                  </div>
                  <div className="settings-item-meta">{shortenPath(sess.dir)} · {agoStr(lastQueryTs(sess), S.chat)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MemoryTab() {
  const S = useStrings();
  const M = S.settings.memory;
  const [content, setContent] = useState("");
  const [draft, setDraft] = useState("");
  const [path, setPath] = useState("");
  const [emptyTemplate, setEmptyTemplate] = useState(true);
  const [autoExtract, setAutoExtract] = useState(true);
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);

  const apply = (data: api.MemoryDoc) => {
    setContent(data.content || "");
    setPath(data.path || "");
    setEmptyTemplate(!!data.empty_template);
    setAutoExtract(data.auto_extract !== false);
  };

  useEffect(() => {
    let cancelled = false;
    void api.fetchMemory().then(({ ok, data }) => {
      if (cancelled) return;
      if (!ok || !data) { showToast(M.loadFailed, 2500); return; }
      apply(data);
    });
    return () => { cancelled = true; };
  }, [M.loadFailed]);

  const openEdit = () => {
    setDraft(content);
    setEditing(true);
  };

  const save = async () => {
    setBusy(true);
    const { ok, data } = await api.saveMemoryApi(draft);
    setBusy(false);
    if (!ok || !data) {
      showToast((data as any)?.detail || M.loadFailed, 3000);
      return;
    }
    apply(data);
    setEditing(false);
    showToast(M.saved, 2500);
  };

  const toggleExtract = async () => {
    const next = !autoExtract;
    setAutoExtract(next);
    const { ok } = await api.savePrefsApi({ auto_extract_memory: next });
    if (!ok) {
      setAutoExtract(!next);
      showToast(S.settings.setFailed, 2500);
    }
  };

  if (editing) {
    return (
      <div className="mem-page">
        <div className="settings-block-title">{M.editTitle}</div>
        <textarea className="pf-input pf-textarea mem-editor" value={draft}
                  onChange={(e) => setDraft(e.target.value)} spellCheck={false} />
        {path ? <div className="settings-item-meta">{path}</div> : null}
        <div className="pf-actions">
          <button className="dp-btn" onClick={() => setEditing(false)} disabled={busy}>{S.common.cancel}</button>
          <button className="dp-btn primary" onClick={() => void save()} disabled={busy}>{S.common.save}</button>
        </div>
      </div>
    );
  }

  return (
    <div className="mem-page">
      <p className="mem-intro">{M.intro}</p>

      <div className="mem-section">
        <div className="mem-section-head">
          <div className="settings-block-title" style={{ marginBottom: 0 }}>{M.extractTitle}</div>
          <button type="button" className={"mem-switch" + (autoExtract ? " on" : "")}
                  role="switch" aria-checked={autoExtract} onClick={() => void toggleExtract()}>
            <i />
          </button>
        </div>
        <p className="mem-hint">{M.extractHint}</p>
        <div className="mem-preview" role="button" tabIndex={0}
             onClick={openEdit}
             onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openEdit(); } }}>
          {emptyTemplate || !content.trim() ? (
            <span className="mem-preview-empty">{M.empty}</span>
          ) : (
            <div className="mem-preview-md"><ChatMarkdown text={content} /></div>
          )}
          <span className="mem-preview-expand" title={M.previewLabel}><IconExpand /></span>
        </div>
        <div className="mem-preview-label">{M.previewLabel}</div>
      </div>
    </div>
  );
}

function GeneralTab() {
  const s = useAppState();
  const S = useStrings();
  return (
    <div className="settings-list">
      <div className="settings-block">
        <div className="settings-block-title">{S.settings.theme}</div>
        <div className="pf-toggle">
          {([["auto", S.settings.themeAuto], ["light", S.settings.themeLight], ["dark", S.settings.themeDark]] as const).map(([v, label]) => (
            <button key={v} className={"pf-toggle-btn" + (s.theme === v ? " active" : "")} onClick={() => setTheme(v)}>{label}</button>
          ))}
        </div>
      </div>

      {/* Each language is labelled in itself, so it is readable to someone who
          landed in the wrong one and cannot read the current UI. */}
      <div className="settings-block">
        <div className="settings-block-title">{S.settings.language}</div>
        <div className="pf-toggle">
          {LANGS.map(({ id, label }) => (
            <button key={id} className={"pf-toggle-btn" + (s.lang === id ? " active" : "")}
                    onClick={() => setLang(id)}>{label}</button>
          ))}
        </div>
        <div className="settings-item-meta">{S.settings.languageMeta}</div>
      </div>

      <div className="settings-block">
        <div className="settings-block-title">{S.settings.defaultDir}</div>
        <DirPicker
          value={s.dirModal.value || s.serverDir}
          onChange={(dir) => setState({ dirModal: { ...getState().dirModal, value: dir } })}
          onCommit={(dir) => void applyBaseDir(dir)}
          extraAction={
            <button className="dp-btn primary"
                    onClick={() => void applyBaseDir(s.dirModal.value || s.serverDir)}>{S.common.apply}</button>
          }
        />
      </div>

      <AccessBlock />
    </div>
  );
}

function ConfigFileBlock() {
  const s = useAppState();
  const S = useStrings();
  const path = s.serverConfigPath || "~/.agentica/config.yaml";
  const [before, after] = S.settings.localConfigHint.split("{path}");
  const open = async () => {
    if (!s.serverConfigPath) return;
    const { ok, data } = await api.openPathApi(s.serverConfigPath, "finder");
    if (!ok) showToast((data as any)?.detail || S.settings.openFailed, 3000);
  };
  return (
    <div className="settings-block">
      <div className="settings-block-title">{S.settings.localConfig}</div>
      <p className="settings-item-meta">
        {before}
        <button type="button" className="config-path-link" onClick={() => void open()} title={path}>
          {path}
        </button>
        {after}
      </p>
    </div>
  );
}

function ProfilesTab() {
  const s = useAppState();
  const S = useStrings();
  const active = s.serverProfile || s.profilesData.active;
  return (
    <>
      {s.profileForm && <ProfileForm />}
      <div className="settings-list">
        <ConfigFileBlock />
        {!(s.profilesData.profiles || []).length && <div className="settings-empty">{S.settings.noProfiles}</div>}
        {(s.profilesData.profiles || []).map((p: any) => (
          <div key={p.name} className={"settings-item" + (p.name === active ? " active" : "")}>
            <div className="settings-item-head">
              <span className="settings-name">
                {p.name}{p.name === active && <span className="settings-active"> ● {S.settings.inUse}</span>}
              </span>
              <div className="settings-item-actions">
                {p.name !== active && <button className="cron-act" onClick={() => void switchProfile(p.name)}>{S.settings.switch}</button>}
                <button className="cron-act" onClick={() => void editProfile(p.name)}>{S.common.edit}</button>
                <button className="cron-act danger" onClick={() => removeProfile(p.name)}>{S.common.delete}</button>
              </div>
            </div>
            <div className="settings-item-meta">
              {p.model_provider || "?"}/{p.model_name || "?"}{p.base_url ? " · " + p.base_url : ""}
              {p.has_api_key ? ` · key ${p.api_key_masked}` : ` · ${S.settings.noKey}`}
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
  const S = useStrings();
  const f = s.profileForm!;
  return (
    <div className="settings-form">
      <h4>{f.editing ? S.settings.editProfile(f.name) : S.settings.newProfile}</h4>
      <input className="pf-input" placeholder={S.settings.profileName} value={f.name} disabled={f.editing}
             onChange={(e) => patch({ name: e.target.value })} />
      <div className="pf-section">{S.settings.mainModel}</div>
      <input className="pf-input" list="provider-list" placeholder={S.settings.providerPlaceholder}
             value={f.model_provider} onChange={(e) => patch({ model_provider: e.target.value })} />
      <datalist id="provider-list">{s.providers.map((p) => <option key={p} value={p} />)}</datalist>
      <input className="pf-input" placeholder="model_name" value={f.model_name} onChange={(e) => patch({ model_name: e.target.value })} />
      <input className="pf-input" placeholder={S.settings.baseUrlOptional} value={f.base_url} onChange={(e) => patch({ base_url: e.target.value })} />
      <input className="pf-input" placeholder={f.editing ? S.settings.apiKeyKeep : "api_key"}
             value={f.api_key} onChange={(e) => patch({ api_key: e.target.value })} />
      <div className="pf-section">{S.settings.tuning}</div>
      <div className="pf-row">
        <input className="pf-input" placeholder="reasoning_effort（low/medium/high）" value={f.reasoning_effort} onChange={(e) => patch({ reasoning_effort: e.target.value })} />
        <input className="pf-input" type="number" placeholder="max_tokens" value={f.max_tokens} onChange={(e) => patch({ max_tokens: e.target.value })} />
      </div>
      <div className="pf-row">
        <input className="pf-input" type="number" placeholder="context_window" value={f.context_window} onChange={(e) => patch({ context_window: e.target.value })} />
        <input className="pf-input" type="number" placeholder="compact_token_limit" value={f.compact_token_limit} onChange={(e) => patch({ compact_token_limit: e.target.value })} />
      </div>
      <div className="pf-row">
        <input className="pf-input" type="number" step="0.1" placeholder="temperature" value={f.temperature} onChange={(e) => patch({ temperature: e.target.value })} />
        <input className="pf-input" type="number" step="0.05" placeholder="top_p" value={f.top_p} onChange={(e) => patch({ top_p: e.target.value })} />
      </div>
      <div className="pf-section">{S.settings.auxModel}</div>
      <input className="pf-input" placeholder="aux provider" value={f.aux_provider} onChange={(e) => patch({ aux_provider: e.target.value })} />
      <input className="pf-input" placeholder="aux model_name" value={f.aux_model} onChange={(e) => patch({ aux_model: e.target.value })} />
      <input className="pf-input" placeholder={S.settings.auxBaseUrl} value={f.aux_base_url} onChange={(e) => patch({ aux_base_url: e.target.value })} />
      <input className="pf-input" placeholder={S.settings.auxApiKey} value={f.aux_api_key} onChange={(e) => patch({ aux_api_key: e.target.value })} />
      <div className="pf-section">{S.settings.envBlock}</div>
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
      <button className="pf-add-env" onClick={() => patch({ envRows: [...f.envRows, { key: "", value: "" }] })}>{S.common.addVar}</button>
      <div className="pf-actions">
        <button className="dp-btn" onClick={() => setState({ profileForm: null })}>{S.common.cancel}</button>
        <button className="dp-btn primary" onClick={() => void saveProfile()}>{f.editing ? S.common.save : S.common.create}</button>
      </div>
    </div>
  );
}

/** Loaded lazily so the settings modal always opens with current server state
 *  rather than whatever the chat page fetched at boot. */
export async function primeSettings(tab: string) {
  setState({ settingsModal: { open: true }, settingsTab: tab });
  await Promise.all([
    loadStatus(), loadProfiles(), loadProviders(), loadDirHistory(), loadCronJobs(),
    loadAuthStatus(),
  ]);
}
