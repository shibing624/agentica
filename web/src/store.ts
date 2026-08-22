import { useSyncExternalStore } from "react";
import * as api from "./api";

/** Ordered stream of one assistant turn (text interleaves with tool groups). */
export type MsgPart =
  | { kind: "think"; text: string; t0?: number; ms?: number }
  | { kind: "tool"; name: string; argsStr: string; result?: string; diff?: string; t0?: number; ms?: number }
  | { kind: "text"; text: string }
  | { kind: "steer"; text: string; ts?: number };

export type ChatMsg = {
  role: "user" | "assistant";
  content: string;
  ts?: number;
  files?: string[];
  steps?: Array<Record<string, any>>;
  parts?: MsgPart[];
  durationSec?: number;
  durationMs?: number;
  llmMs?: number;
  tokIn?: number;
  tokOut?: number;
  costUsd?: number;
  cacheRead?: number;
  cacheHitPercent?: number | null;
  previews?: string[];
  aborted?: boolean;
  error?: string;
  /** Mid-run interrupt (CLI steer), not a new turn. */
  steer?: boolean;
};

export type Session = {
  title: string;
  msgs: ChatMsg[];
  /** First request time. Sidebar order and agoStr both use this. */
  ts: number;
  tokIn: number;
  tokOut: number;
  tokTotal: number;
  requests: number;
  totalTime: number;
  lastInputTokens: number;
  contextTokens: number;
  costUsd: number;
  dir: string;
  projectId?: string;
  archived?: boolean;
  unread?: boolean;
  /** Server says a run is in flight (refresh can reattach). */
  running?: boolean;
};

export type QueuedMessage = { id: string; sessionId: string; text: string; files: File[]; ts: number };

export type ProfileForm = {
  name: string;
  editing: boolean;
  model_provider: string;
  model_name: string;
  base_url: string;
  api_key: string;
  reasoning_effort: string;
  max_tokens: string;
  context_window: string;
  compact_token_limit: string;
  temperature: string;
  top_p: string;
  aux_provider: string;
  aux_model: string;
  aux_base_url: string;
  aux_api_key: string;
  envRows: { key: string; value: string }[];
};

export type CronForm = {
  id: string | null;
  name: string;
  prompt: string;
  schedule: string;
  timeout_seconds: string;
  max_retries: string;
  validate_run: boolean;
};

export type SkillForm = {
  name: string;
  editing: boolean;
  description: string;
  trigger: string;
  content: string;
};

export type McpForm = {
  name: string;
  kind: "stdio" | "sse";
  command: string;
  args: string;
  url: string;
  envRows: { key: string; value: string }[];
};

export type AppState = {
  /** Bumped on every change. This store is a mutable singleton — `state` and
   *  every object inside it keep their identity forever — so a `useMemo` over
   *  store data can only be keyed on this. Keying on `state.sessions` looks
   *  right and never recomputes. */
  rev: number;
  curSess: string | null;
  sessions: Record<string, Session>;
  streams: Record<string, {
    abortCtrl: AbortController;
    aiMsg: ChatMsg;
    userStopped?: boolean;
    preparing?: boolean;
    cancelling?: boolean;
    reconnecting?: boolean;
    runId?: string;
    lastSeq?: number;
  }>;
  goalRuns: Record<string, { status: string; objective: string; progress: string }>;
  /** Blocking slash commands (``/compact``) that hold the session lock. */
  commandRuns: Record<string, { kind: "compact" }>;
  pendingFiles: File[];
  messageQueue: QueuedMessage[];
  /** After reload, ChatPage reattaches to this in-flight session. */
  pendingResume: string | null;
  /** Composer is in /goal mode: set a token budget, then type the objective. */
  goalCompose: { budgetText: string } | null;
  serverModel: string;
  serverDir: string;
  serverProvider: string;
  serverModelName: string;
  serverVersion: string;
  serverConfigPath: string;
  serverReasoningEffort: string;
  serverContextWindow: number;
  serverProfile: string;
    serverSupportsImages: boolean;
    serverMediaModel: string;
  profilesData: { active: string; profiles: any[] };
  providers: string[];
  inputText: string;
  modelDDOpen: boolean;
  approvalMenuOpen: boolean;
  chatMenuOpen: boolean;
  selectedApprovalMode: string;
  sidebarCollapsed: boolean;
  theme: string;
  /** UI language. Lives here rather than in i18n.ts so a component reading
   *  strings re-renders through the same `rev` bump as everything else. */
  lang: "en" | "zh";
  toast: { show: boolean; msg: string };
  confirm: { open: boolean; title: string; msg: string; okLabel: string; onOk: (() => void) | null };
  dirModal: { open: boolean; forNewSession: boolean; value: string };
  dirHistory: string[];
  dirBrowse: { open: boolean; path: string; parent: string | null; dirs: { name: string; path: string }[] };
  pendingNewChatDir: string;
  accountPanelOpen: boolean;
  pluginsPanelOpen: boolean;
  pluginsTab: string;
  pluginsSearch: string;
  pluginsData: { tools: any[]; skills: any[]; mcpServers: any[] };
  skillForm: SkillForm | null;
  mcpForm: McpForm | null;
  settingsTab: string;
  settingsModal: { open: boolean };
  profileForm: ProfileForm | null;
  cronJobs: any[];
  cronForm: CronForm | null;
  cronRuns: Record<string, any[]>;
  cronRunsOpen: string[];
  cronBusy: string;
  sidebarSearch: string;
  /** Whether a web password exists. Decides what the access-control block
   *  offers ("set" vs "change") and whether logging out is a thing. */
  passwordSet: boolean;
  /** Still the password the gateway generated on first start. */
  passwordIsInitial: boolean;
  /** The signed-in account. Also names the ``users/<id>/`` partition its
   *  conversations and memory live in, so it is what the sidebar shows. */
  accountId: string;
  /** "admin" or "user". Only account management is admin-only. */
  accountRole: string;
  /** The seeded administrator. Its initial password is the one printed at first start. */
  defaultAccountId: string;
  /** How the current session was issued: "password" | "token" | "desktop". */
  sessionVia: string;
  minPasswordLength: number;
  /** Change-password overlay. `userId` is whose password; current password
   *  typed in the form is always the person at the keyboard's. */
  passwordDialog: { open: boolean; userId: string };
  /** The account table, admin only. Empty until the page loads it. */
  users: Array<{
    user_id: string; role: string; created_at: string;
    password_is_initial: boolean; has_password: boolean; is_admin: boolean;
  }>;
};

export function emptyProfileForm(): ProfileForm {
  return {
    name: "", editing: false, model_provider: "", model_name: "", base_url: "", api_key: "",
    reasoning_effort: "", max_tokens: "", context_window: "", compact_token_limit: "", temperature: "", top_p: "",
    aux_provider: "", aux_model: "", aux_base_url: "", aux_api_key: "",
    envRows: [],
  };
}

export function emptyCronForm(): CronForm {
  return {
    id: null, name: "", prompt: "", schedule: "",
    timeout_seconds: "", max_retries: "0", validate_run: true,
  };
}

export function emptySkillForm(): SkillForm {
  return { name: "", editing: false, description: "", trigger: "", content: "" };
}

export function emptyMcpForm(): McpForm {
  return { name: "", kind: "stdio", command: "", args: "", url: "", envRows: [] };
}

const listeners = new Set<() => void>();

const state: AppState = {
  rev: 0,
  curSess: null,
  sessions: {},
  streams: {},
  goalRuns: {},
  commandRuns: {},
  pendingFiles: [],
  messageQueue: [],
  pendingResume: null,
  goalCompose: null,
  serverModel: "-",
  serverDir: "",
  serverProvider: "",
  serverModelName: "",
  serverVersion: "",
  serverConfigPath: "",
  serverReasoningEffort: "",
  serverContextWindow: 128000,
  serverProfile: "",
    serverSupportsImages: false,
    serverMediaModel: "",
  profilesData: { active: "", profiles: [] },
  providers: [],
  inputText: "",
  modelDDOpen: false,
  approvalMenuOpen: false,
  chatMenuOpen: false,
  selectedApprovalMode: localStorage.getItem("ag_approval") || "auto",
  sidebarCollapsed: false,
  theme: localStorage.getItem("ag_theme") || "auto",
  lang: localStorage.getItem("ag_lang") === "zh" ? "zh" : "en",
  toast: { show: false, msg: "" },
  confirm: { open: false, title: "", msg: "", okLabel: "", onOk: null },
  dirModal: { open: false, forNewSession: false, value: "" },
  dirHistory: [],
  dirBrowse: { open: false, path: "", parent: null, dirs: [] },
  pendingNewChatDir: "",
  accountPanelOpen: false,
  pluginsPanelOpen: false,
  pluginsTab: "skills",
  pluginsSearch: "",
  pluginsData: { tools: [], skills: [], mcpServers: [] },
  skillForm: null,
  mcpForm: null,
  settingsTab: "settings",
  settingsModal: { open: false },
  profileForm: null,
  cronJobs: [],
  cronForm: null,
  cronRuns: {},
  cronRunsOpen: [],
  cronBusy: "",
  sidebarSearch: "",
  passwordSet: false,
  passwordIsInitial: false,
  accountId: "default",
  accountRole: "user",
  defaultAccountId: "default",
  sessionVia: "",
  minPasswordLength: 6,
  users: [],
  passwordDialog: { open: false, userId: "" },
};

export function getState() { return state; }

export function bump() {
  state.rev += 1;
  listeners.forEach((l) => l());
}

export function setState(patch: Partial<AppState>) {
  Object.assign(state, patch);
  bump();
}

export function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function useAppState(): AppState {
  useSyncExternalStore(subscribe, () => state.rev, () => state.rev);
  return state;
}

export function showToast(msg: string, ms = 1800) {
  state.toast = { show: true, msg };
  bump();
  window.setTimeout(() => { state.toast = { show: false, msg: "" }; bump(); }, ms);
}

/** Ask before something irreversible. The dialog is rendered by <ConfirmDialog />.
 *  An omitted `okLabel` is filled in there with the translated "Delete" — this
 *  module must not import the string table, since the table reads the store. */
export function askConfirm(opts: { title: string; msg: string; okLabel?: string; onOk: () => void }) {
  setState({
    confirm: {
      open: true, title: opts.title, msg: opts.msg,
      okLabel: opts.okLabel || "", onOk: opts.onOk,
    },
  });
}

export function closeConfirm() {
  setState({ confirm: { open: false, title: "", msg: "", okLabel: "", onOk: null } });
}

/** Cache is per account: the same browser logging in as `kk` must not
 *  resurrect `default`'s sidebar from a leftover `ag_s`. */
function accountCachePrefix() {
  return state.accountId || "default";
}

export function saveSessions() {
  localStorage.setItem("ag_s:" + accountCachePrefix(), JSON.stringify(state.sessions));
}

export function readSessionCache(): Record<string, Session> {
  const namespaced = localStorage.getItem("ag_s:" + accountCachePrefix());
  const raw = namespaced || localStorage.getItem("ag_s") || "{}";
  return JSON.parse(raw);
}

export function readLastSessionId(): string | null {
  return localStorage.getItem("ag_a:" + accountCachePrefix()) || localStorage.getItem("ag_a");
}

export function writeLastSessionId(id: string | null, persist = true) {
  const key = "ag_a:" + accountCachePrefix();
  if (id) localStorage.setItem(key, id);
  else localStorage.removeItem(key);
  if (persist) pushPrefs();
}

export function pushPrefs() {
  void api.savePrefsApi({
    theme: state.theme,
    lang: state.lang,
    approval_mode: state.selectedApprovalMode,
    last_session_id: state.curSess,
  });
}

export function resolvedTheme(raw: string) {
  if (raw === "dark" || raw === "light") return raw;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Theme is applied on <html> so it covers every route, not just the chat page. */
export function applyTheme(raw: string) {
  document.documentElement.dataset.theme = resolvedTheme(raw);
}

export function setTheme(raw: string) {
  localStorage.setItem("ag_theme", raw);
  applyTheme(raw);
  setState({ theme: raw });
  pushPrefs();
}

export const UNFILED_PROJECT_ID = "unfiled";

export function projectIdForDir(dir: string) {
  const d = (dir || "").trim();
  if (!d) return UNFILED_PROJECT_ID;
  return "dir:" + encodeURIComponent(d.replace(/\/+$/, ""));
}

export function projectNameForDir(dir: string) {
  const d = (dir || "").replace(/\/+$/, "");
  if (!d) return "Unfiled";
  const parts = d.split("/").filter(Boolean);
  return parts[parts.length - 1] || d;
}
