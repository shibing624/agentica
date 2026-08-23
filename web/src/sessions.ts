import * as api from "./api";
import { getStrings } from "./i18n";
import { uid } from "./lib/format";
import {
  askConfirm, bump, getState, projectIdForDir, readLastSessionId, readSessionCache,
  saveSessions, setState, showToast, writeLastSessionId,
  type ChatMsg, type Session,
} from "./store";

/** Session list state, shared by the sidebar and the settings modal's archive
 *  tab so restoring a session there updates the tree without a reload. */

export async function loadSessions() {
  // Overlay only: token counts and the local dir live in the browser, but the
  // list itself is the server's. Merging local-only ids is how logging in as
  // `kk` used to resurrect `default`'s conversations from the same origin.
  const local = readSessionCache();
  const { ok, data } = await api.fetchSessions();
  const serverList = (ok && data?.sessions) ? data.sessions : [];
  const merged: Record<string, Session> = {};
  const st = getState();
  for (const sv of serverList) {
    const id = sv.session_id;
    const lc = local[id] || {};
    const dir = sv.work_dir || lc.dir || st.serverDir || "";
    merged[id] = {
      title: sv.name || lc.title || "Chat",
      msgs: lc.msgs || [],
      ts: Date.parse(sv.first_timestamp) || lc.ts || Date.now(),
      tokIn: lc.tokIn || 0, tokOut: lc.tokOut || 0, tokTotal: lc.tokTotal || 0,
      requests: lc.requests || 0, totalTime: lc.totalTime || 0, lastInputTokens: lc.lastInputTokens || 0,
      contextTokens: lc.contextTokens || 0, costUsd: lc.costUsd || 0,
      dir,
      projectId: projectIdForDir(dir),
      archived: !!sv.archived,
      running: !!sv.running,
    };
  }
  setState({ sessions: merged });
  saveSessions();
  const last = readLastSessionId();
  if (last && merged[last]) {
    switchTo(last);
    const sess = merged[last];
    if (sess.running) {
      setState({ pendingResume: last });
    } else if (shouldHydrateFromServer(sess, last)) {
      void hydrateSession(last, true);
    }
  }
}

export function switchTo(id: string) {
  const st = getState();
  st.curSess = id;
  const sess = st.sessions[id];
  if (sess) sess.unread = false;
  writeLastSessionId(id);
  bump();
  if (sess?.running && !getState().streams[id]) setState({ pendingResume: id });
  else if (sess && shouldHydrateFromServer(sess, id)) void hydrateSession(id, !!sess.msgs.length);
}

/** Replay a session's transcript from the server log so a reload (or another
 *  machine) still shows history localStorage never had. */
export async function hydrateSession(id: string, force = false) {
  const { ok, data } = await api.fetchTraceEvents(id, 0, 1000);
  if (!ok || !data?.events) return;
  const sess = getState().sessions[id];
  if (!sess || (sess.msgs.length && !force)) return;
  const msgs: ChatMsg[] = [];
  for (const e of data.events) {
    if (e.type === "user" || e.type === "assistant") {
      msgs.push({
        role: e.type,
        content: String(e.content || ""),
        ts: e.timestamp ? Date.parse(e.timestamp) : Date.now(),
      });
    }
  }
  if (!msgs.length) return;
  sess.msgs = msgs;
  saveSessions();
  bump();
  void syncSessionRoundStats(id);
}

function looksLikeDisconnectCasualty(sess: Session): boolean {
  const last = sess.msgs[sess.msgs.length - 1];
  if (!last || last.role !== "assistant" || !last.error) return false;
  const err = last.error.toLowerCase();
  return err.includes("network") || err.includes("failed to fetch") || err.includes("load failed");
}

function looksIncompleteAssistant(sess: Session): boolean {
  const last = sess.msgs[sess.msgs.length - 1];
  if (!last || last.role !== "assistant") return false;
  if (last.aborted || last.tokIn != null) return false;
  if (last.error) return looksLikeDisconnectCasualty(sess);
  const parts = last.parts || [];
  if (!last.content && !parts.length) return true;
  return parts.length > 0;
}

function shouldHydrateFromServer(sess: Session, id: string): boolean {
  if (getState().streams[id]) return false;
  if (!sess.msgs.length) return true;
  return looksLikeDisconnectCasualty(sess) || looksIncompleteAssistant(sess);
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

export function createSession(dir: string) {
  const id = uid();
  getState().sessions[id] = {
    title: "New Chat", msgs: [], ts: Date.now(), tokIn: 0, tokOut: 0, tokTotal: 0,
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

export function renameSession(id: string, name: string) {
  const sess = getState().sessions[id];
  if (!sess || !name.trim()) return;
  sess.title = name.trim();
  saveSessions();
  bump();
  void api.renameSessionApi(id, sess.title);
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
