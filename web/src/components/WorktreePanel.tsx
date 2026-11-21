import { useEffect } from "react";
import { getState, setState, showToast } from "../store";
import * as api from "../api";

async function loadWorktrees() {
  const sessionId = getState().curSess;
  if (!sessionId) return;
  setState({ worktreePanel: { ...getState().worktreePanel, loading: true, error: "" } });
  try {
    const res = await api.getWorktrees(sessionId);
    if (!res.ok || !res.data) throw new Error(res.error ? String(res.error) : `HTTP ${res.status}`);
    setState({ worktreePanel: { ...getState().worktreePanel, state: res.data, loading: false } });
  } catch (e) {
    setState({ worktreePanel: { ...getState().worktreePanel, loading: false, error: String(e) } });
  }
}

async function action(kind: "new" | "bind" | "merge" | "remove", name: string) {
  const sessionId = getState().curSess;
  if (!sessionId) return;
  const panel = getState().worktreePanel;
  setState({ worktreePanel: { ...panel, busy: true, error: "", output: "" } });
  try {
    const res = await api.postWorktree(sessionId, kind, name, panel.base);
    if (!res.ok || !res.data) throw new Error(res.error ? String(res.error) : `HTTP ${res.status}`);
    const session = getState().sessions[sessionId];
    if (kind === "bind" && session && res.data.work_dir) {
      session.projectDir ||= session.dir;
      session.dir = res.data.work_dir;
    }
    const refreshed = await api.getWorktrees(sessionId);
    if (!refreshed.ok || !refreshed.data) throw new Error(refreshed.error ? String(refreshed.error) : `HTTP ${refreshed.status}`);
    setState({
      worktreePanel: {
        ...getState().worktreePanel,
        state: refreshed.data,
        busy: false,
        output: res.data.output || refreshed.data.output || "",
        name: "",
      },
    });
    if (kind === "bind" || kind === "merge" || kind === "remove") showToast("Worktree updated");
  } catch (e) {
    setState({ worktreePanel: { ...getState().worktreePanel, busy: false, error: String(e) } });
  }
}

export function WorktreePanel() {
  const panel = getState().worktreePanel;
  useEffect(() => { void loadWorktrees(); }, []);
  if (!panel.open) return null;
  const entries = panel.state?.worktrees || [];
  return (
    <div className="dlg-overlay open" onMouseDown={(e) => {
      if (e.target === e.currentTarget) setState({ worktreePanel: { ...panel, open: false } });
    }}>
      <div className="dlg worktree-panel" role="dialog" aria-label="Worktrees" onMouseDown={(e) => e.stopPropagation()}>
        <div className="dlg-head">
          <strong>Worktrees</strong>
          <button className="wt-close" onClick={() => setState({ worktreePanel: { ...panel, open: false } })}>×</button>
        </div>
        <div className="worktree-current">Session work dir: {panel.state?.work_dir || getState().serverDir}</div>
        <div className="worktree-list">
          {panel.loading && <div className="worktree-empty">Loading…</div>}
          {!panel.loading && !entries.length && <div className="worktree-empty">No worktrees yet.</div>}
          {entries.map((entry) => (
            <div className="worktree-row" key={entry.name}>
              <div className="worktree-name">
                <strong>{entry.name}</strong>
                <span>{entry.describe}</span>
              </div>
              <div className="worktree-actions">
                {!entry.is_main && <button disabled={panel.busy} onClick={() => action("bind", entry.name)}>Bind</button>}
                {!entry.is_main && <button disabled={panel.busy} onClick={() => action("merge", entry.name)}>Merge</button>}
                {!entry.is_main && <button disabled={panel.busy} onClick={() => action("remove", entry.name)}>Remove</button>}
              </div>
            </div>
          ))}
        </div>
        <div className="worktree-form">
          <input
            aria-label="Worktree name"
            placeholder="task-name"
            value={panel.name}
            onChange={(e) => setState({ worktreePanel: { ...panel, name: e.currentTarget.value } })}
          />
          <button
            disabled={panel.busy || !panel.name.trim()}
            onClick={() => action("new", panel.name.trim())}
          >New</button>
          <button disabled={panel.busy} onClick={loadWorktrees}>Refresh</button>
        </div>
        {panel.output && <pre className="worktree-output">{panel.output}</pre>}
        {panel.error && <pre className="worktree-output worktree-error">{panel.error}</pre>}
      </div>
    </div>
  );
}
