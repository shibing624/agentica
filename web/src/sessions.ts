import * as api from "./api";
import { uid } from "./lib/format";
import {
  askConfirm, bump, getState, projectIdForDir, saveSessions, setState, showToast,
  type ChatMsg, type Session,
} from "./store";

/** Session list state, shared by the sidebar and the settings modal's archive
 *  tab so restoring a session there updates the tree without a reload. */

export async function loadSessions() {
  const local = JSON.parse(localStorage.getItem("ag_s") || "{}");
  const { ok, data } = await api.fetchSessions();
  const serverList = (ok && data?.sessions) ? data.sessions : [];
  const merged: Record<string, Session> = {};
  const st = getState();
  for (const sv of serverList) {
    const id = sv.session_id;
    const lc = local[id] || {};
    merged[id] = {
      title: sv.name || lc.title || "Chat",
      msgs: lc.msgs || [],
      ts: sv.last_timestamp ? new Date(sv.last_timestamp).getTime() : (lc.ts || Date.now()),
      tokIn: lc.tokIn || 0, tokOut: lc.tokOut || 0, tokTotal: lc.tokTotal || 0,
      requests: lc.requests || 0, totalTime: lc.totalTime || 0, lastInputTokens: lc.lastInputTokens || 0,
      dir: lc.dir || st.serverDir || "",
      archived: !!(sv.archived || lc.archived),
    };
  }
  for (const id of Object.keys(local)) if (!merged[id]) merged[id] = local[id];
  setState({ sessions: merged });
  saveSessions();
  const last = localStorage.getItem("ag_a");
  if (last && merged[last]) switchTo(last);
}

export function switchTo(id: string) {
  const st = getState();
  st.curSess = id;
  const sess = st.sessions[id];
  if (sess) sess.unread = false;
  localStorage.setItem("ag_a", id);
  bump();
  if (sess && !sess.msgs.length) void hydrateSession(id);
}

/** Replay a session's transcript from the server log so a reload (or another
 *  machine) still shows history localStorage never had. */
export async function hydrateSession(id: string) {
  const { ok, data } = await api.fetchTraceEvents(id, 0, 1000);
  if (!ok || !data?.events) return;
  const sess = getState().sessions[id];
  if (!sess || sess.msgs.length) return;
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
}

export function createSession(dir: string) {
  const id = uid();
  getState().sessions[id] = {
    title: "New Chat", msgs: [], ts: Date.now(), tokIn: 0, tokOut: 0, tokTotal: 0,
    requests: 0, totalTime: 0, lastInputTokens: 0, dir, projectId: projectIdForDir(dir),
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
  localStorage.removeItem("ag_a");
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
    localStorage.removeItem("ag_a");
  }
  bump();
  showToast("已归档");
}

export function unarchiveSession(id: string) {
  const sess = getState().sessions[id];
  if (!sess) return;
  sess.archived = false;
  saveSessions();
  void api.unarchiveSessionApi(id);
  bump();
  showToast("已恢复");
}

export function deleteSession(id: string) {
  const sess = getState().sessions[id];
  if (!sess) return;
  askConfirm({
    title: "删除会话",
    msg: `“${sess.title}” 及其服务端日志将被永久删除。`,
    onOk: async () => {
      const st = getState();
      delete st.sessions[id];
      if (st.curSess === id) {
        st.curSess = null;
        localStorage.removeItem("ag_a");
      }
      saveSessions();
      bump();
      await api.deleteSessionApi(id);
      showToast("会话已删除");
    },
  });
}
