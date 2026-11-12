import { useSyncExternalStore } from "react";
import * as api from "./api";
import type { Automation } from "./api";
import { sessionStatsCache, type SessionStats } from "./lib/sessionCache";
import { createViewStore } from "./lib/sessionView";

/** Ordered stream of one assistant turn (text interleaves with tool groups). */
export type MsgPart =
  | { kind: "think"; text: string; t0?: number; ms?: number }
  | { kind: "tool"; name: string; argsStr: string; result?: string; diff?: string; t0?: number; ms?: number; toolCallId?: string }
  | { kind: "text"; text: string }
  | { kind: "steer"; text: string; ts?: number };

export type ApprovalDecision = "allow" | "allow_prefix" | "deny" | "deny_prefix";

/** SSE `approval_request` payload, keyed by the matching tool row. */
export type ApprovalRequest = {
  toolCallId: string;
  name: string;
  args: Record<string, unknown>;
  question: string;
  preview: string;
  similarLabel?: string;
  options: ApprovalDecision[];
};

/** SSE `question_request`: an open `ask_user_question`. */
export type QuestionRequest = {
  questionId: string;
  prompt: string;
  options: string[];
};

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
  /** Prompt tokens written to the provider cache this turn (Anthropic-style). */
  cacheWrite?: number;
  cacheHitPercent?: number | null;
  /** Model API calls this turn took (tool loops make it more than one). */
  requests?: number;
  /** Model ids this turn called, e.g. "deepseek-v4" or "a, b" after a fallback. */
  model?: string;
  previews?: string[];
  aborted?: boolean;
  error?: string;
  /** Wall clock when a blocking run error was set; used for the on-screen stamp. */
  errorTs?: number;
  /** Mid-run interrupt (CLI steer), not a new turn. */
  steer?: boolean;
  /** A turn no person wrote: a background command's or a delegated subagent's
   *  report, another session's message. The log marks the row `automated`;
   *  it renders as a notice, not as the user's query. */
  automated?: boolean;
  /** The automation that delivered this automated turn (log `cron_job`). */
  cronJob?: { id: string; name: string };
  /** SessionLog row this message ends at (assistant: its last row). Set when
   *  the message comes from the server log; fork and feedback address it. */
  uuid?: string;
  /** Thumbs up/down left on this reply. */
  feedback?: { rating: "up" | "down" | null; comment?: string };
  /** A `/skill args` turn: the log holds the expanded SKILL.md prompt; the
   *  bubble shows this chip instead. */
  skill?: { name: string; args: string };
  /** A Web `/compact` result. `pending` while the request runs; `notes` is the
   *  `<session>.notes.md` the new window starts from (rendered as markdown). */
  compact?: { pending?: boolean; before?: number; after?: number; notes?: string; notesPath?: string; auto?: boolean };
};

export type Session = {
  title: string;
  msgs: ChatMsg[];
  /** First request time. Project group order uses this, not last activity. */
  ts: number;
  /** Last query (or last log event if the transcript is not hydrated). */
  lastTs: number;
  tokIn: number;
  tokOut: number;
  tokTotal: number;
  requests: number;
  totalTime: number;
  lastInputTokens: number;
  contextTokens: number;
  costUsd: number;
  dir: string;
  /** The user picked this title; the first message must not replace it. */
  named?: boolean;
  projectId?: string;
  archived?: boolean;
  /** Kept at the top of its project in the sidebar (sidecar `pinned`). */
  pinned?: boolean;
  unread?: boolean;
  /** Server says a run is in flight (refresh can reattach). */
  running?: boolean;
  /** Number of running background jobs (`execute(background=True)` / `delegate`). */
  backgroundJobs?: number;
  source?: string;
  /** Where the session was started; `source` is where the latest turn came in. */
  origin?: string;
  preview?: string;
  userCount?: number;
  sizeBytes?: number;
  hydratedSize?: number;
  /** True when earlier turns exist before `msgs` (not loaded yet). */
  hasOlder?: boolean;
  /** Entry cursor passed as `before` when loading the next older page. */
  olderCursor?: number;
  /** Number of user turns before `msgs`, so turn numbers stay exact. */
  olderTurns?: number;
  /** True while an older page request is in flight. */
  loadingOlder?: boolean;
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
  enable_cache_control: boolean;
  cache_control_session_header: string;
  aux_provider: string;
  aux_model: string;
  aux_base_url: string;
  aux_api_key: string;
  envRows: { key: string; value: string }[];
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
  /** Until the first session catalog lands, the last open chat is unknown:
   *  showing the new-chat screen then would flash before it is restored. */
  booting: boolean;
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
    pendingApprovals?: ApprovalRequest[];
    decidingApproval?: boolean;
    pendingQuestions?: QuestionRequest[];
    answeringQuestion?: boolean;
  }>;
  goalRuns: Record<string, { status: string; objective: string; progress: string }>;
  /** Blocking slash commands (``/compact``) that hold the session lock. */
  commandRuns: Record<string, { kind: "compact" }>;
  pendingFiles: File[];
  messageQueue: QueuedMessage[];
  /** After reload, ChatPage reattaches to this in-flight session. */
  pendingResume: string | null;
  /** Composer is in /goal mode: type the objective; click the chip to set a token budget. */
  goalCompose: { budgetText: string } | null;
  serverModel: string;
  serverDir: string;
  serverProvider: string;
  serverModelName: string;
  serverVersion: string;
  latestVersion: string;
  updateAvailable: boolean;
  updateBusy: "idle" | "checking" | "updating";
  serverConfigPath: string;
  serverReasoningEffort: string;
  serverContextWindow: number;
  serverProfile: string;
  /** The model the open chat runs on; null until the gateway answers. */
  sessionModel: { id: string; profile: string; source: string; provider: string; model: string; effort: string } | null;
  /** A model picked on the new-chat screen, applied when the chat is created. */
  pendingModel: string;
    serverSupportsImages: boolean;
    serverMediaModel: string;
  profilesData: { active: string; profiles: any[] };
  providers: string[];
  inputText: string;
  modelDDOpen: boolean;
  /** Upward picker over the composer, opened from the `+` / `/` menus. */
  composerPicker: "model" | "permission" | "skill" | null;
  approvalMenuOpen: boolean;
  chatMenuOpen: boolean;
  selectedApprovalMode: string;
  sidebarCollapsed: boolean;
  /** Expanded sidebar width in px; dragged by the handle, remembered per browser. */
  sidebarWidth: number;
  /** Folders the user added or renamed. A workspace is a directory; sessions
   *  group under it by work dir, so this only carries what sessions cannot:
   *  an empty folder and a display name. Mirrors `prefs.workspaces`. */
  workspaces: Array<{ dir: string; name?: string }>;
  workspaceAdd: { open: boolean; value: string };
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
  pluginsTab: string;
  pluginsSearch: string;
  pluginsData: { tools: any[]; skills: any[]; mcpServers: any[] };
  skillForm: SkillForm | null;
  mcpForm: McpForm | null;
  settingsTab: string;
  settingsModal: { open: boolean };
  profileForm: ProfileForm | null;
  /** This account's automations, for the sidebar badge and the chat links. */
  cronJobs: Automation[];
  sidebarSearch: string;
  /** Jump target inside the open chat when the user clicks a search match or steps prev/next. */
  chatSearchTarget: { sessionId: string; query: string; turn: number; role?: "user" | "assistant"; seq: number } | null;
  sessionScope: "recent" | "all";
  /** List CLI/SDK sessions too (account pref; off keeps web history separate). */
  showCliSessions: boolean;
  sessionsLoading: boolean;
  sessionsError: string;
  /** Monotonic counter incremented when remote pulse / session catalog lands. */
  pulseSeq: number;
  /** Cached background jobs per session id for chat rows and header badge. */
  bgJobsBySession: Record<string, api.BackgroundJob[]>;
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
    enable_cache_control: false, cache_control_session_header: "",
    aux_provider: "", aux_model: "", aux_base_url: "", aux_api_key: "",
    envRows: [],
  };
}

export function emptySkillForm(): SkillForm {
  return { name: "", editing: false, description: "", trigger: "", content: "" };
}

export function emptyMcpForm(): McpForm {
  return { name: "", kind: "stdio", command: "", args: "", url: "", envRows: [] };
}

export const SIDEBAR_MIN = 220;
export const SIDEBAR_MAX = 480;
export const SIDEBAR_DEFAULT = 280;

export function clampSidebarWidth(w: number) {
  if (!Number.isFinite(w)) return SIDEBAR_DEFAULT;
  return Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, Math.round(w)));
}

function readSidebarWidth() {
  const raw = Number(localStorage.getItem("ag_sidebar_w"));
  return raw ? clampSidebarWidth(raw) : SIDEBAR_DEFAULT;
}

const listeners = new Set<() => void>();

const state: AppState = {
  rev: 0,
  curSess: null,
  booting: true,
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
  latestVersion: "",
  updateAvailable: false,
  updateBusy: "idle",
  serverConfigPath: "",
  serverReasoningEffort: "",
  serverContextWindow: 128000,
  serverProfile: "",
  sessionModel: null,
  pendingModel: "",
    serverSupportsImages: false,
    serverMediaModel: "",
  profilesData: { active: "", profiles: [] },
  providers: [],
  inputText: "",
  modelDDOpen: false,
  composerPicker: null,
  approvalMenuOpen: false,
  chatMenuOpen: false,
  selectedApprovalMode: localStorage.getItem("ag_approval") || "auto",
  sidebarCollapsed: false,
  sidebarWidth: readSidebarWidth(),
  workspaces: [],
  workspaceAdd: { open: false, value: "" },
  theme: localStorage.getItem("ag_theme") || "auto",
  lang: localStorage.getItem("ag_lang") === "zh" ? "zh" : "en",
  toast: { show: false, msg: "" },
  confirm: { open: false, title: "", msg: "", okLabel: "", onOk: null },
  dirModal: { open: false, forNewSession: false, value: "" },
  dirHistory: [],
  dirBrowse: { open: false, path: "", parent: null, dirs: [] },
  pendingNewChatDir: "",
  accountPanelOpen: false,
  pluginsTab: "skills",
  pluginsSearch: "",
  pluginsData: { tools: [], skills: [], mcpServers: [] },
  skillForm: null,
  mcpForm: null,
  settingsTab: "settings",
  settingsModal: { open: false },
  profileForm: null,
  cronJobs: [],
  sidebarSearch: "",
  chatSearchTarget: null,
  sessionScope: "recent",
  showCliSessions: false,
  sessionsLoading: false,
  sessionsError: "",
  pulseSeq: 0,
  bgJobsBySession: {},
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

/**
 * Each conversation owns its unsent composer text and attachments. The
 * composer reads the global `inputText` / `pendingFiles`; whenever the current
 * conversation changes, the outgoing one's draft is parked and the incoming
 * one's restored. Hooked into `bump` so every `curSess` assignment is covered.
 */
const NO_SESSION = "";
const composerDrafts = new Map<string, { text: string; files: File[] }>();
let composerOwner: string = NO_SESSION;

function swapComposerDraft() {
  const next = state.curSess || NO_SESSION;
  if (next === composerOwner) return;
  const prev = composerOwner;
  composerOwner = next;
  // Sending from the new-chat screen creates the session mid-submit: the
  // text typed there belongs to that new session, not to the next blank screen.
  if (prev === NO_SESSION && !composerDrafts.has(next) && !state.sessions[next]?.msgs.length) {
    return;
  }
  if (state.inputText || state.pendingFiles.length) {
    composerDrafts.set(prev, { text: state.inputText, files: state.pendingFiles });
  } else {
    composerDrafts.delete(prev);
  }
  const incoming = composerDrafts.get(next);
  composerDrafts.delete(next);
  state.inputText = incoming?.text ?? "";
  state.pendingFiles = incoming?.files ?? [];
}

export function bump() {
  swapComposerDraft();
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
  const key = "ag_s:" + accountCachePrefix();
  try {
    localStorage.setItem(key, JSON.stringify(sessionStatsCache(state.sessions)));
  } catch (e) {
    // The cache is optional (the server log is the record); a full or
    // disabled storage must not abort the send that triggered this save.
    console.warn("session cache not saved:", e);
    localStorage.removeItem(key);
  }
}

export function readSessionCache(): Record<string, Partial<SessionStats>> {
  const namespaced = localStorage.getItem("ag_s:" + accountCachePrefix());
  const raw = namespaced || localStorage.getItem("ag_s") || "{}";
  return JSON.parse(raw);
}

export function readLastSessionId(): string | null {
  return localStorage.getItem("ag_a:" + accountCachePrefix()) || localStorage.getItem("ag_a");
}

/** Last page and position per session, so reopening one returns there. */
export const sessionViews = createViewStore(localStorage, () => "ag_view:" + accountCachePrefix());

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

/** Display name of a workspace: the user's label if they gave one, else the folder name. */
export function workspaceName(dir: string) {
  const key = normDir(dir);
  return state.workspaces.find((w) => normDir(w.dir) === key)?.name || projectNameForDir(dir);
}

export function normDir(dir: string) {
  const d = (dir || "").trim();
  return d.replace(/[/\\]+$/, "") || d;
}

export function enqueueApproval(sessId: string, req: ApprovalRequest) {
  const live = state.streams[sessId];
  if (!live) return;
  const list = live.pendingApprovals || (live.pendingApprovals = []);
  const i = list.findIndex((x) => x.toolCallId === req.toolCallId);
  if (i >= 0) list[i] = req;
  else list.push(req);
  bump();
}

export function dequeueApproval(sessId: string, toolCallId: string) {
  const live = state.streams[sessId];
  if (!live?.pendingApprovals) return;
  live.pendingApprovals = live.pendingApprovals.filter((x) => x.toolCallId !== toolCallId);
  bump();
}

export function enqueueQuestion(sessId: string, q: QuestionRequest) {
  const live = state.streams[sessId];
  if (!live) return;
  const list = live.pendingQuestions || (live.pendingQuestions = []);
  if (!list.some((x) => x.questionId === q.questionId)) list.push(q);
  bump();
}

export function dequeueQuestion(sessId: string, questionId: string) {
  const live = state.streams[sessId];
  if (!live?.pendingQuestions) return;
  live.pendingQuestions = live.pendingQuestions.filter((x) => x.questionId !== questionId);
  bump();
}
