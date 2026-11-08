import * as api from "./api";
import { getStrings } from "./i18n";
import { uid, lastQueryTs } from "./lib/format";
import { eventsToChatMsgs, isSteerUserEvent, mergeClientSteers, rememberSteersFromMsgs } from "./lib/hydrateMessages";
import { emptyChatIds } from "./lib/sessionDrafts";
import { searchEventsConversation, searchMsgsConversation, type SessionContentHit } from "./lib/sessionRows";
import {
  askConfirm, bump, getState, projectIdForDir, readLastSessionId, readSessionCache,
  saveSessions, setState, showToast, writeLastSessionId,
  type ChatMsg, type Session,
} from "./store";

/** Session list state, shared by the sidebar and the settings modal's archive
 *  tab so restoring a session there updates the tree without a reload. */

let catalogGeneration = 0;
let catalogFlight: { scope: "recent" | "all"; promise: Promise<void> } | null = null;
let catalogAbort: AbortController | null = null;
const hydrateGenerations: Record<string, number> = {};
/** Names the user typed that the server has not confirmed yet: a catalog
 *  response already in flight would otherwise put the old title back. */
const pendingNames: Record<string, string> = {};
/** Default user turns loaded on session open and per "Load earlier" click (dsh style). */
export const CHAT_PAGE_TURNS = 6;
/** Minimum interval between passive focus/visibility catalog syncs. */
const FOCUS_SYNC_COOLDOWN_MS = 5000;
let lastCatalogLoadAt = 0;

export function loadSessions(scope: "recent" | "all" = getState().sessionScope): Promise<void> {
  if (catalogFlight?.scope === scope) return catalogFlight.promise;
  catalogAbort?.abort();
  const controller = new AbortController();
  catalogAbort = controller;
  const generation = ++catalogGeneration;
  const promise = (async () => {
    const cur = getState();
    const showSpinner = cur.sessionScope !== scope || Object.keys(cur.sessions).length === 0;
    if (showSpinner) setState({ sessionsLoading: true, sessionsError: "" });
    const { ok, data, error } = await api.fetchSessions(scope, controller.signal);
    if (generation !== catalogGeneration) return;
    if (!ok || !Array.isArray(data?.sessions)) {
      setState({
        sessionsLoading: false,
        sessionsError: error instanceof Error ? error.message : "Could not load sessions",
      });
      return;
    }
    lastCatalogLoadAt = Date.now();
    const local = readSessionCache();
    const merged: Record<string, Session> = {};
    const st = getState();
    for (const sv of data.sessions) {
      const id = sv.session_id;
      const lc = local[id] || {};
      const mem = st.sessions[id];
      const dir = sv.work_dir || st.serverDir || "";
      const firstTs = Date.parse(sv.first_timestamp) || Date.now();
      const serverLast = Date.parse(sv.last_timestamp) || 0;
      const next: Session = {
        title: pendingNames[id] || sv.name || "Chat",
        msgs: mem?.msgs || [],
        ts: firstTs,
        lastTs: Math.max(serverLast, mem ? lastQueryTs(mem) : 0, firstTs),
        tokIn: lc.tokIn || 0, tokOut: lc.tokOut || 0, tokTotal: lc.tokTotal || 0,
        requests: mem?.requests || lc.requests || 0, totalTime: mem?.totalTime || lc.totalTime || 0,
        lastInputTokens: mem?.lastInputTokens || lc.lastInputTokens || 0,
        contextTokens: mem?.contextTokens || lc.contextTokens || 0,
        costUsd: mem?.costUsd || lc.costUsd || 0,
        dir,
        projectId: projectIdForDir(dir),
        archived: !!sv.archived,
        pinned: !!sv.pinned,
        running: sessionIsLive(id) || !!sv.running,
        unread: !!sv.unread,
        source: sv.source || "legacy",
        preview: sv.preview || "",
        userCount: sv.user_count || 0,
        sizeBytes: Number(sv.size_bytes) || 0,
        hydratedSize: mem?.hydratedSize,
        hasOlder: mem?.hasOlder,
        olderCursor: mem?.olderCursor,
        olderTurns: mem?.olderTurns,
        loadingOlder: mem?.loadingOlder,
      };
      merged[id] = mem ? Object.assign(mem, next) : next;
    }
    if (st.curSess && !merged[st.curSess] && st.sessions[st.curSess]) {
      merged[st.curSess] = st.sessions[st.curSess];
    }
    setState({ sessions: merged, sessionScope: scope, sessionsLoading: false, sessionsError: "" });
    saveSessions();
    const last = readLastSessionId();
    // Restore the last open chat on start-up only: on a later refresh this
    // key may hold what another tab opened since.
    if (last && merged[last] && !getState().curSess) switchTo(last);
  })().finally(() => {
    if (catalogFlight?.promise === promise) catalogFlight = null;
    if (catalogAbort === controller) catalogAbort = null;
  });
  catalogFlight = { scope, promise };
  return promise;
}

export function switchTo(id: string) {
  const st = getState();
  st.curSess = id;
  const sess = st.sessions[id];
  if (sess) {
    sess.unread = false;
    markSessionRead(id);
  }
  writeLastSessionId(id);
  bump();
  if (!sess || sessionIsLive(id)) return;
  if (sess.msgs.length && sess.hydratedSize === sess.sizeBytes) return;
  void (async () => {
    await hydrateSession(id, true);
    if (getState().curSess !== id) return;
    if (getState().sessions[id]?.running && !sessionIsLive(id)) {
      setState({ pendingResume: id });
    }
  })();
}

export function markSessionRead(id: string) {
  const sess = getState().sessions[id];
  if (sess) sess.unread = false;
  void api.markSessionReadApi(id);
}

/** Poll only active runs; refresh the catalog once when a remote run finishes. */
export async function syncSessionStatus() {
  const st = getState();
  const ids = Object.entries(st.sessions)
    .filter(([id, sess]) => sess.running && !sessionIsLive(id))
    .map(([id]) => id);
  if (!ids.length) return;
  const results = await Promise.all(ids.map((id) => api.fetchActiveRun(id)));
  let changed = false;
  let finished = false;
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    const { ok, data } = results[i];
    if (!ok) continue;
    const sess = getState().sessions[id];
    if (!sess) continue;
    const running = sessionIsLive(id) || !!data?.run;
    if (sess.running !== running) {
      sess.running = running;
      changed = true;
    }
    if (!running) finished = true;
  }
  if (changed) bump();
  if (finished) await syncFromServer(false);
}

/**
 * Pick up what another client (a browser tab, the desktop window, the CLI)
 * did to these sessions: refresh the catalog, then bring the open session up
 * to date — attach to a run started elsewhere, or re-read a log that grew
 * since this client last read it. Each client otherwise only reads the
 * catalog once at start-up and an open transcript once, so a turn finished
 * in one window never showed in the other.
 */
export async function syncFromServer(force = false) {
  if (!force && Date.now() - lastCatalogLoadAt < FOCUS_SYNC_COOLDOWN_MS) return;
  await loadSessions();
  const id = getState().curSess;
  const sess = id ? getState().sessions[id] : null;
  if (!id || !sess || sessionIsLive(id) || !sess.sizeBytes) return;
  if (sess.hydratedSize !== sess.sizeBytes) await hydrateSession(id, true);
  const now = getState().sessions[id];
  if (getState().curSess === id && now?.running && !sessionIsLive(id)) {
    setState({ pendingResume: id });
  }
}

function sessionIsLive(id: string): boolean {
  const st = getState();
  return !!(st.streams[id] || st.goalRuns[id] || st.commandRuns[id]);
}

function countUserTurns(msgs: Session["msgs"]): number {
  let n = 0;
  for (const m of msgs) {
    if (m.role === "user" && !m.steer && !m.automated) n++;
  }
  return n;
}

const CHAT_ENTRY_TYPES = new Set(["user", "assistant", "tool", "tool_audit", "compact_boundary"]);

/** Client-side turn-aligned tail slice when talking to a gateway process started
 *  before ``/trace/events?turns=`` existed (which returns ``offset=0, limit=200``
 *  from the head of the log with no ``has_more`` field). */
function sliceTurnPageClient(all: any[], turns: number, before?: number): api.TraceEventsPage {
  const chatEntries = all.filter((e) => e && typeof e === "object" && CHAT_ENTRY_TYPES.has(String(e.type || "")));
  const total = chatEntries.length;
  const end = before != null ? Math.min(total, Math.max(0, before)) : total;
  const turnStarts: number[] = [];
  for (let i = 0; i < end; i++) {
    const e = chatEntries[i];
    if (e.type === "user" && !e.automated && !isSteerUserEvent(e)) {
      turnStarts.push(i);
    }
  }
  let start = 0;
  let olderTurns = 0;
  if (turnStarts.length > turns) {
    start = turnStarts[turnStarts.length - turns];
    olderTurns = turnStarts.length - turns;
    if (start > 0 && chatEntries[start - 1]?.type === "compact_boundary") start -= 1;
  }
  return {
    session_id: "",
    offset: start,
    limit: end - start,
    total,
    events: chatEntries.slice(start, end),
    has_more: start > 0,
    next_before: start,
    older_turns: olderTurns,
  };
}

async function fetchAllTraceEvents(id: string, firstPage?: api.TraceEventsPage): Promise<any[] | null> {
  const all: any[] = [];
  let offset = 0;
  const limit = 1000;
  if (firstPage && Array.isArray(firstPage.events) && firstPage.offset === 0) {
    all.push(...firstPage.events);
    const total = typeof firstPage.total === "number" ? firstPage.total : all.length;
    if (all.length >= total) return all;
    offset = all.length;
  }
  for (;;) {
    const { ok, data } = await api.fetchTraceEvents(id, offset, limit);
    if (!ok || !data?.events) return offset === 0 ? null : all;
    all.push(...data.events);
    const total = typeof data.total === "number" ? data.total : all.length;
    if (all.length >= total || data.events.length < limit) return all;
    offset += data.events.length;
  }
}

async function fetchTurnPageCompat(
  id: string,
  turns: number,
  before?: number,
): Promise<{ ok: boolean; data?: api.TraceEventsPage | null }> {
  const res = await api.fetchTraceTurnPage(id, turns, before);
  if (!res.ok || !res.data?.events) return res;
  if (typeof res.data.has_more === "boolean") return res;
  // Running gateway predates `?turns=`: it returned the first 200 raw rows from
  // offset 0 (only round 1). Read the full log and slice the tail turns here.
  const all = await fetchAllTraceEvents(id, before == null ? res.data : undefined);
  if (!all) return { ok: false };
  return { ok: true, data: { ...sliceTurnPageClient(all, turns, before), session_id: id } };
}

/** Replay a session's transcript from the server log so a reload (or another
 *  machine) still shows history localStorage never had — including tool
 *  calls, matching the live web turn. */
export async function hydrateSession(id: string, force = false) {
  if (sessionIsLive(id)) return;
  const existing = getState().sessions[id];
  if (!force && existing?.msgs.length && existing.hydratedSize === existing.sizeBytes) return;
  if (existing?.msgs.length) {
    rememberSteersFromMsgs(id, existing.msgs, existing.olderTurns || 0);
  }
  const generation = (hydrateGenerations[id] || 0) + 1;
  hydrateGenerations[id] = generation;
  const turns = existing && existing.hasOlder === false
    ? 200
    : Math.max(CHAT_PAGE_TURNS, countUserTurns(existing?.msgs || []));
  const { ok, data } = await fetchTurnPageCompat(id, turns);
  if (!ok || !data?.events || hydrateGenerations[id] !== generation) return;
  const sess = getState().sessions[id];
  if (!sess || sessionIsLive(id)) return;
  if (sess.msgs.length) {
    rememberSteersFromMsgs(id, sess.msgs, sess.olderTurns || 0);
  }
  const olderTurns = data.older_turns ?? 0;
  sess.msgs = mergeClientSteers(id, eventsToChatMsgs(data.events), olderTurns);
  sess.hasOlder = !!data.has_more;
  sess.olderCursor = data.next_before ?? 0;
  sess.olderTurns = olderTurns;
  sess.loadingOlder = false;
  sess.hydratedSize = sess.sizeBytes;
  sess.lastTs = lastQueryTs(sess);
  saveSessions();
  bump();
  if (sess.msgs.length) {
    void syncSessionRoundStats(id);
    void syncFeedback(id);
  }
}

/** Prepend one older page of turns above the currently loaded window. */
export async function loadOlderMessages(id: string): Promise<boolean> {
  const sess = getState().sessions[id];
  if (!sess || !sess.hasOlder || sess.loadingOlder) return false;
  const before = sess.olderCursor;
  if (before == null || before <= 0) {
    sess.hasOlder = false;
    bump();
    return false;
  }
  const generation = hydrateGenerations[id] || 0;
  sess.loadingOlder = true;
  bump();
  const { ok, data } = await fetchTurnPageCompat(id, CHAT_PAGE_TURNS, before);
  const cur = getState().sessions[id];
  if (!cur || (hydrateGenerations[id] || 0) !== generation) return false;
  cur.loadingOlder = false;
  if (!ok || !data?.events) {
    bump();
    return false;
  }
  const older = eventsToChatMsgs(data.events);
  const olderTurns = data.older_turns ?? 0;
  cur.msgs = mergeClientSteers(id, [...older, ...cur.msgs], olderTurns);
  cur.hasOlder = !!data.has_more;
  cur.olderCursor = data.next_before ?? 0;
  cur.olderTurns = olderTurns;
  bump();
  if (older.length) {
    void syncSessionRoundStats(id);
    void syncFeedback(id);
  }
  return older.length > 0;
}

/** Stamp thumbs up/down (kept in the session sidecar) onto the replies. */
async function syncFeedback(id: string) {
  const { ok, data } = await api.fetchFeedbackApi(id);
  const sess = getState().sessions[id];
  if (!ok || !data?.feedback || !sess) return;
  let changed = false;
  for (const m of sess.msgs) {
    const fb = m.uuid ? data.feedback[m.uuid] : undefined;
    if (!fb && !m.feedback) continue;
    m.feedback = fb ? { rating: fb.rating, comment: fb.comment } : undefined;
    changed = true;
  }
  if (changed) bump();
}

/** The log uuid of message ``idx``. A reply that just streamed in has none
 *  until the transcript is read back from the server log. */
async function msgUuid(id: string, idx: number): Promise<string | undefined> {
  const sess = getState().sessions[id];
  const cur = sess?.msgs[idx];
  if (cur?.uuid) return cur.uuid;
  if (!sess) return undefined;
  const fromEnd = sess.msgs.length - idx;
  const { ok, data } = await fetchTurnPageCompat(id, Math.max(1, CHAT_PAGE_TURNS));
  if (!ok || !data?.events) return undefined;
  const tail = eventsToChatMsgs(data.events);
  return tail[tail.length - fromEnd]?.uuid;
}

/** Branch the conversation after reply ``idx`` into a new session and open it. */
export async function forkAt(id: string, idx: number) {
  const S = getStrings();
  const at = await msgUuid(id, idx);
  if (!at) { showToast(S.chat.forkNotSaved, 2500); return; }
  const { ok, data } = await api.forkSessionApi(id, at);
  if (!ok || !data?.session_id) { showToast(S.chat.forkFailed, 2500); return; }
  await loadSessions();
  switchTo(data.session_id);
  showToast(S.chat.forkDone);
}

/** Thumbs up/down (click the active one again to clear) and an optional note. */
export async function rateMessage(
  id: string, idx: number, rating: "up" | "down" | null, comment = "",
) {
  const S = getStrings();
  const at = await msgUuid(id, idx);
  if (!at) { showToast(S.chat.forkNotSaved, 2500); return false; }
  const m = getState().sessions[id]?.msgs[idx];
  const before = m?.feedback;
  if (m) { m.uuid = at; m.feedback = rating || comment ? { rating, comment } : undefined; bump(); }
  const { ok } = await api.setFeedbackApi(id, at, rating, comment);
  if (!ok) {
    if (m) { m.feedback = before; bump(); }
    showToast(S.chat.feedbackFailed, 2500);
    return false;
  }
  return true;
}

/** Stamp each assistant footer with the same round the Trace page draws. */
export async function syncSessionRoundStats(id: string) {
  // New Chat has a client id and no jsonl yet; the analysis route 404s.
  const existing = getState().sessions[id];
  if (!existing || (!existing.sizeBytes && !existing.msgs.some((m) => m.role === "assistant"))) return;
  const { ok, data } = await api.fetchTraceAnalysis(id);
  if (!ok || !data?.rounds) return;
  const sess = getState().sessions[id];
  if (!sess) return;
  const rounds = (data.rounds as Array<{
    compaction?: boolean;
    durationMs: number;
    llmMs: number;
    requests?: number;
    tokens: {
      prompt: number;
      output: number;
      cacheRead: number;
      cacheWrite?: number;
      cacheHitPercent: number | null;
    };
    costUsd: number | null;
  }>).filter((r) => !r.compaction);
  // The log records the session's model once; per-round ids are not kept.
  const model = typeof data.meta?.model === "string" ? data.meta.model : "";
  // Compact cards are not rounds (the analysis drops compaction rounds too).
  const liveAi = getState().streams[id]?.aiMsg;
  const assistants = sess.msgs.filter((m) => m.role === "assistant" && !m.compact && m !== liveAi);
  const settledRounds = liveAi && rounds.length > assistants.length ? rounds.slice(0, rounds.length - 1) : rounds;
  const n = assistants.length;
  const roundOffset = Math.max(0, settledRounds.length - n);
  let changed = false;
  const lastRound = rounds[rounds.length - 1];
  if (lastRound?.tokens?.prompt && sess.lastInputTokens !== lastRound.tokens.prompt) {
    sess.lastInputTokens = lastRound.tokens.prompt;
    changed = true;
  }
  for (let i = 0; i < n && roundOffset + i < settledRounds.length; i++) {
    const m = assistants[i];
    const rd = settledRounds[roundOffset + i];
    if (
      m.llmMs === rd.llmMs
      && m.durationMs === rd.durationMs
      && m.tokIn === rd.tokens.prompt
      && m.tokOut === rd.tokens.output
      && m.cacheWrite === (rd.tokens.cacheWrite ?? m.cacheWrite)
      && m.requests === (rd.requests ?? m.requests)
      && (!model || m.model)
    ) continue;
    m.llmMs = rd.llmMs;
    m.durationMs = rd.durationMs;
    m.durationSec = rd.durationMs / 1000;
    m.tokIn = rd.tokens.prompt;
    m.tokOut = rd.tokens.output;
    m.cacheRead = rd.tokens.cacheRead;
    if (rd.tokens.cacheWrite != null) m.cacheWrite = rd.tokens.cacheWrite;
    m.cacheHitPercent = rd.tokens.cacheHitPercent;
    if (rd.requests != null) m.requests = rd.requests;
    if (model && !m.model) m.model = model;
    if (rd.costUsd != null) m.costUsd = rd.costUsd;
    changed = true;
  }
  if (!changed) return;
  saveSessions();
  bump();
}

/**
 * Create (or reuse) the workspace's single unsent "New Chat".
 *
 * Sidebar + buttons, the top New chat button and `ensureSession()` all land
 * here. Without reuse, clicking + N times left N identical empty rows in the
 * sidebar — none of them has a server-side log, so they disappear on refresh
 * and the user loses what looked like N conversations.
 */
export function createSession(dir: string) {
  const st = getState();
  const drafts = emptyChatIds(st.sessions, dir);
  if (drafts.length) {
    const [keep, ...stale] = drafts;
    for (const id of stale) delete st.sessions[id];
    saveSessions();
    switchTo(keep);
    return keep;
  }
  const id = uid();
  st.sessions[id] = {
    title: "New Chat", msgs: [], ts: Date.now(), lastTs: Date.now(), tokIn: 0, tokOut: 0, tokTotal: 0,
    requests: 0, totalTime: 0, lastInputTokens: 0, contextTokens: 0, costUsd: 0,
    dir, projectId: projectIdForDir(dir),
  };
  saveSessions();
  switchTo(id);
  return id;
}

export function newChat() {
  const st = getState();
  const cur = st.curSess ? st.sessions[st.curSess] : null;
  const dir = (cur && cur.dir) || st.pendingNewChatDir || st.serverDir;
  if (!dir) { setState({ dirModal: { open: true, forNewSession: true, value: st.serverDir } }); return; }
  // The top button and a workspace's + must land on the same unsent draft.
  // Otherwise the sidebar can show a "New Chat" row while the main pane shows
  // a second, unrelated empty welcome screen.
  const drafts = emptyChatIds(st.sessions, dir);
  if (drafts.length) {
    for (const id of drafts.slice(1)) delete st.sessions[id];
    setState({ pendingNewChatDir: dir, inputText: "" });
    switchTo(drafts[0]);
    return;
  }
  setState({ curSess: null, pendingNewChatDir: dir, inputText: "" });
  writeLastSessionId(null);
}

/** New empty conversation in a specific project (the sidebar + on that folder). */
export function newChatInDir(dir: string) {
  if (!dir) {
    setState({ dirModal: { open: true, forNewSession: true, value: "" } });
    return;
  }
  createSession(dir);
}

/** Rename a chat. A chat with no turn yet has no log on the server, so the
 *  name is kept on the session (`named`) and pushed once its first run exists
 *  (see `pushPendingName`); the first message then no longer overwrites it. */
export async function renameSession(id: string, name: string): Promise<void> {
  const sess = getState().sessions[id];
  const title = name.trim();
  if (!sess || !title || title === sess.title) return;
  const previous = sess.title;
  sess.title = title;
  sess.named = true;
  saveSessions();
  bump();
  if (!sess.msgs.length && !sess.sizeBytes) return;
  pendingNames[id] = title;
  const { ok } = await api.renameSessionApi(id, title);
  if (pendingNames[id] === title) delete pendingNames[id];
  if (ok) return;
  if (sess.title === title) {
    sess.title = previous;
    saveSessions();
    bump();
  }
  showToast(getStrings().chat.renameFailed, 2500);
}

/** After a renamed-before-first-message chat gets its first run. */
export function pushPendingName(id: string) {
  const sess = getState().sessions[id];
  if (!sess?.named) return;
  const title = sess.title;
  pendingNames[id] = title;
  void api.renameSessionApi(id, title).then(({ ok }) => {
    if (pendingNames[id] === title) delete pendingNames[id];
    if (!ok) showToast(getStrings().chat.renameFailed, 2500);
  });
}

export function archiveSession(id: string) {
  const st = getState();
  if (!st.sessions[id]) return;
  st.sessions[id].archived = true;
  saveSessions();
  void api.archiveSessionApi(id);
  if (st.curSess === id) {
    st.curSess = null;
    writeLastSessionId(null);
  }
  bump();
  showToast(getStrings().session.archived);
}

export function pinSession(id: string, pinned: boolean) {
  const sess = getState().sessions[id];
  if (!sess) return;
  sess.pinned = pinned;
  saveSessions();
  bump();
  // A session with no log yet has no sidecar to write; the flag stays local
  // until its first message creates one.
  if (!sess.msgs.length && !sess.sizeBytes) return;
  void api.pinSessionApi(id, pinned).then(({ ok }) => {
    if (ok) return;
    sess.pinned = !pinned;
    saveSessions();
    bump();
    showToast(getStrings().session.pinFailed, 2500);
  });
}

/** Branch the whole conversation into a new session and open it. */
export async function forkSession(id: string) {
  const S = getStrings();
  const { ok, data } = await api.forkSessionApi(id);
  if (!ok || !data?.session_id) { showToast(S.chat.forkFailed, 2500); return; }
  await loadSessions();
  switchTo(data.session_id);
  showToast(S.chat.forkDone);
}

export function unarchiveSession(id: string) {
  const sess = getState().sessions[id];
  if (!sess) return;
  sess.archived = false;
  saveSessions();
  void api.unarchiveSessionApi(id);
  bump();
  showToast(getStrings().session.restored);
}

export function deleteSession(id: string) {
  const sess = getState().sessions[id];
  if (!sess) return;
  const S = getStrings();
  askConfirm({
    title: S.session.removeSession,
    msg: S.session.removeSessionMsg(sess.title),
    onOk: async () => {
      const st = getState();
      delete st.sessions[id];
      if (st.curSess === id) {
        st.curSess = null;
        writeLastSessionId(null);
      }
      saveSessions();
      bump();
      await api.deleteSessionApi(id);
      showToast(S.session.deleted);
    },
  });
}

const rawEventsCache = new Map<string, { size: number; events: any[] }>();

/**
 * Search user queries and assistant answers across all visible sessions.
 * Uses `/api/sessions/search` when available, and falls back to client-side
 * event scanning if the running gateway process predates that endpoint.
 */
export async function searchAllSessionsContent(
  query: string,
  signal?: AbortSignal,
): Promise<Record<string, SessionContentHit[]>> {
  const q = query.trim();
  if (!q) return {};
  const st = getState();
  const out: Record<string, SessionContentHit[]> = {};

  const { ok, data } = await api.searchSessionsApi(q, st.sessionScope, signal);
  if (signal?.aborted) return {};
  if (ok && data?.results && typeof data.results === "object") {
    Object.assign(out, data.results);
  } else {
    // Client-side fallback when the gateway process has not been restarted yet.
    const entries = Object.entries(st.sessions).filter(([, s]) => !s.archived);
    const queue = [...entries];
    const workers = Array.from({ length: Math.min(4, queue.length) }, async () => {
      while (queue.length && !signal?.aborted) {
        const item = queue.shift();
        if (!item) break;
        const [id, sess] = item;
        if (!sess.sizeBytes && sess.msgs.length) {
          const hits = searchMsgsConversation(sess.msgs, sess.olderTurns || 0, q);
          if (hits.length) out[id] = hits;
          continue;
        }
        if (!sess.sizeBytes) continue;
        let cached = rawEventsCache.get(id);
        if (!cached || cached.size !== sess.sizeBytes) {
          const events = await fetchAllTraceEvents(id);
          if (signal?.aborted) return;
          if (events) {
            cached = { size: sess.sizeBytes, events };
            rawEventsCache.set(id, cached);
          }
        }
        if (cached?.events) {
          const hits = searchEventsConversation(cached.events, q);
          if (hits.length) out[id] = hits;
        }
      }
    });
    await Promise.all(workers);
  }
  // Merge any in-memory live messages that haven't been flushed or hydrated yet.
  for (const [id, sess] of Object.entries(getState().sessions)) {
    if (sess.archived || !sess.msgs.length) continue;
    const memHits = searchMsgsConversation(sess.msgs, sess.olderTurns || 0, q);
    if (!memHits.length) continue;
    const existing = out[id] || [];
    const byKey = new Set(existing.map((h) => `${h.turn}:${h.role}`));
    for (const h of memHits) {
      if (!byKey.has(`${h.turn}:${h.role}`)) existing.push(h);
    }
    existing.sort((a, b) => a.turn - b.turn);
    out[id] = existing;
  }
  return out;
}

/**
 * Open session `id`, load older pages if needed until `turn` is in the DOM,
 * and focus/highlight that turn's message.
 */
export async function openSessionAtTurn(
  id: string,
  turn: number,
  query: string,
  role?: "user" | "assistant",
) {
  switchTo(id);
  await hydrateSession(id);
  let guard = 0;
  while (guard++ < 35) {
    const sess = getState().sessions[id];
    if (!sess || (sess.olderTurns || 0) < turn || !sess.hasOlder) break;
    const loaded = await loadOlderMessages(id);
    if (!loaded) break;
  }
  setState({
    chatSearchTarget: { sessionId: id, query, turn, role, seq: Date.now() },
  });
}
