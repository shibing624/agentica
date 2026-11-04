import * as api from "./api";
import { getStrings } from "./i18n";
import { uid, lastQueryTs } from "./lib/format";
import { eventsToChatMsgs } from "./lib/hydrateMessages";
import { emptyChatIds } from "./lib/sessionDrafts";
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

export function loadSessions(scope: "recent" | "all" = getState().sessionScope): Promise<void> {
  if (catalogFlight?.scope === scope) return catalogFlight.promise;
  catalogAbort?.abort();
  const controller = new AbortController();
  catalogAbort = controller;
  const generation = ++catalogGeneration;
  const promise = (async () => {
    if (getState().sessionScope === scope) setState({ sessionsLoading: true, sessionsError: "" });
    const { ok, data, error } = await api.fetchSessions(scope, controller.signal);
    if (generation !== catalogGeneration) return;
    if (!ok || !Array.isArray(data?.sessions)) {
      setState({
        sessionsLoading: false,
        sessionsError: error instanceof Error ? error.message : "Could not load sessions",
      });
      return;
    }
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
        requests: lc.requests || 0, totalTime: lc.totalTime || 0, lastInputTokens: lc.lastInputTokens || 0,
        contextTokens: lc.contextTokens || 0, costUsd: lc.costUsd || 0,
        dir,
        projectId: projectIdForDir(dir),
        archived: !!sv.archived,
        running: sessionIsLive(id) || !!sv.running,
        unread: !!sv.unread,
        source: sv.source || "legacy",
        preview: sv.preview || "",
        userCount: sv.user_count || 0,
        sizeBytes: Number(sv.size_bytes) || 0,
        hydratedSize: mem?.hydratedSize,
      };
      merged[id] = mem ? Object.assign(mem, next) : next;
    }
    if (st.curSess && !merged[st.curSess] && st.sessions[st.curSess]) {
      merged[st.curSess] = st.sessions[st.curSess];
    }
    setState({ sessions: merged, sessionScope: scope, sessionsLoading: false, sessionsError: "" });
    saveSessions();
    const last = readLastSessionId();
    if (last && merged[last] && getState().curSess !== last) switchTo(last);
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
  if (finished) await loadSessions();
}

function sessionIsLive(id: string): boolean {
  const st = getState();
  return !!(st.streams[id] || st.goalRuns[id] || st.commandRuns[id]);
}

/** Replay a session's transcript from the server log so a reload (or another
 *  machine) still shows history localStorage never had — including tool
 *  calls, matching the live web turn. */
export async function hydrateSession(id: string, force = false) {
  if (sessionIsLive(id)) return;
  const existing = getState().sessions[id];
  if (!force && existing?.msgs.length && existing.hydratedSize === existing.sizeBytes) return;
  const generation = (hydrateGenerations[id] || 0) + 1;
  hydrateGenerations[id] = generation;
  const events = await fetchAllTraceEvents(id);
  if (!events || hydrateGenerations[id] !== generation) return;
  const sess = getState().sessions[id];
  if (!sess || sessionIsLive(id)) return;
  sess.msgs = eventsToChatMsgs(events);
  sess.hydratedSize = sess.sizeBytes;
  sess.lastTs = lastQueryTs(sess);
  saveSessions();
  bump();
  if (sess.msgs.length) {
    void syncSessionRoundStats(id);
    void syncFeedback(id);
  }
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
  const cur = getState().sessions[id]?.msgs[idx];
  if (cur?.uuid) return cur.uuid;
  if (!getState().sessions[id]) return undefined;
  const fresh = await fetchAllTraceEvents(id);
  if (!fresh) return undefined;
  return eventsToChatMsgs(fresh)[idx]?.uuid;
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

async function fetchAllTraceEvents(id: string): Promise<any[] | null> {
  const all: any[] = [];
  let offset = 0;
  const limit = 1000;
  for (;;) {
    const { ok, data } = await api.fetchTraceEvents(id, offset, limit);
    if (!ok || !data?.events) return offset === 0 ? null : all;
    all.push(...data.events);
    const total = typeof data.total === "number" ? data.total : all.length;
    if (all.length >= total || data.events.length < limit) return all;
    offset += limit;
  }
}

/** Stamp each assistant footer with the same round the Trace page draws. */
export async function syncSessionRoundStats(id: string) {
  // New Chat has a client id and no jsonl yet; the analysis route 404s.
  // A live round is not in the log either — `done` already stamped that footer.
  if (getState().streams[id]) return;
  const existing = getState().sessions[id];
  if (!existing?.msgs.some((m) => m.role === "assistant")) return;
  const { ok, data } = await api.fetchTraceAnalysis(id);
  if (!ok || !data?.rounds) return;
  const sess = getState().sessions[id];
  if (!sess) return;
  const rounds = (data.rounds as Array<{
    compaction?: boolean;
    durationMs: number;
    llmMs: number;
    tokens: {
      prompt: number;
      output: number;
      cacheRead: number;
      cacheHitPercent: number | null;
    };
    costUsd: number | null;
  }>).filter((r) => !r.compaction);
  const assistants = sess.msgs.filter((m) => m.role === "assistant");
  const n = assistants.length;
  let changed = false;
  for (let i = 0; i < n && i < rounds.length; i++) {
    const m = assistants[i];
    const rd = rounds[i];
    if (
      m.llmMs === rd.llmMs
      && m.durationMs === rd.durationMs
      && m.tokIn === rd.tokens.prompt
      && m.tokOut === rd.tokens.output
    ) continue;
    m.llmMs = rd.llmMs;
    m.durationMs = rd.durationMs;
    m.durationSec = rd.durationMs / 1000;
    m.tokIn = rd.tokens.prompt;
    m.tokOut = rd.tokens.output;
    m.cacheRead = rd.tokens.cacheRead;
    m.cacheHitPercent = rd.tokens.cacheHitPercent;
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
