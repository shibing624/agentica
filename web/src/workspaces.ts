import * as api from "./api";
import { getStrings } from "./i18n";
import { newChatInDir } from "./sessions";
import {
  askConfirm, bump, getState, normDir, projectNameForDir, saveSessions, setState, showToast,
  writeLastSessionId,
} from "./store";

/**
 * Workspace actions. A workspace is a directory: sessions belong to it through
 * their work dir, so adding one only registers the folder (and a label), and
 * "delete" removes the sessions under it — never the folder on disk.
 */

type Entry = { dir: string; name?: string };

async function saveWorkspaces(next: Entry[]): Promise<{ ok: boolean; detail: string }> {
  const { ok, data } = await api.savePrefsApi({ workspaces: next });
  if (!ok) return { ok: false, detail: (data as any)?.detail || "" };
  setState({ workspaces: Array.isArray((data as any)?.workspaces) ? (data as any).workspaces : next });
  return { ok: true, detail: "" };
}

export function openAddWorkspace() {
  setState({ workspaceAdd: { open: true, value: getState().serverDir || "" } });
}

export function closeAddWorkspace() {
  setState({ workspaceAdd: { open: false, value: "" }, dirBrowse: { open: false, path: "", parent: null, dirs: [] } });
}

/** Register a folder as a workspace and open a fresh chat in it. */
export async function addWorkspace(rawDir: string): Promise<boolean> {
  const S = getStrings();
  const dir = normDir(rawDir);
  if (!dir) { showToast(S.dir.empty); return false; }
  const known = getState().workspaces;
  if (!known.some((w) => normDir(w.dir) === dir)) {
    const res = await saveWorkspaces([...known, { dir }]);
    if (!res.ok) { showToast(res.detail || S.workspace.addFailed, 3500); return false; }
  }
  closeAddWorkspace();
  newChatInDir(dir);
  return true;
}

export async function renameWorkspace(dir: string, name: string) {
  const key = normDir(dir);
  const label = name.trim();
  if (label === workspaceLabel(key)) return;
  const known = getState().workspaces;
  const exists = known.some((w) => normDir(w.dir) === key);
  const next = exists
    ? known.map((w) => (normDir(w.dir) === key ? { dir: w.dir, ...(label ? { name: label } : {}) } : w))
    : [...known, { dir: key, ...(label ? { name: label } : {}) }];
  const res = await saveWorkspaces(next);
  if (!res.ok) showToast(res.detail || S().workspace.renameFailed, 3000);
}

function S() { return getStrings(); }

function workspaceLabel(key: string) {
  return getState().workspaces.find((w) => normDir(w.dir) === key)?.name || projectNameForDir(key);
}

export function deleteWorkspace(dir: string, name: string) {
  const T = getStrings();
  const key = normDir(dir);
  const st = getState();
  const ids = Object.entries(st.sessions)
    .filter(([, s]) => !s.archived && normDir(s.dir) === key)
    .map(([id]) => id);
  const busy = ids.some((id) => st.sessions[id].running || st.streams[id] || st.goalRuns[id] || st.commandRuns[id]);
  if (busy) { showToast(T.workspace.busy, 3000); return; }
  askConfirm({
    title: T.workspace.deleteTitle,
    msg: T.workspace.deleteMsg(name, ids.length),
    okLabel: T.workspace.deleteOk,
    onOk: async () => {
      const results = await Promise.all(ids.map((id) => api.deleteSessionApi(id)));
      const failed = results.filter((r) => !r.ok).length;
      const cur = getState();
      ids.forEach((id, i) => {
        if (!results[i].ok) return;
        delete cur.sessions[id];
        if (cur.curSess === id) {
          cur.curSess = null;
          writeLastSessionId(null);
        }
      });
      saveSessions();
      bump();
      await saveWorkspaces(cur.workspaces.filter((w) => normDir(w.dir) !== key));
      showToast(failed ? T.workspace.deletePartial(failed) : T.workspace.deleted, failed ? 3500 : 1800);
    },
  });
}
