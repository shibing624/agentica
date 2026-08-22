import { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { getStrings, useStrings } from "../i18n";
import { agoStr } from "../lib/format";
import { archiveSession, deleteSession, newChatInDir, renameSession, switchTo } from "../sessions";
import { IconArchive, IconClose, IconFolder, IconPencil, IconPlus } from "../icons";
import {
  getState, projectIdForDir, projectNameForDir, useAppState, type Session,
} from "../store";

/**
 * The conversation list in the left sidebar, grouped by working directory.
 *
 * It lives here rather than in ChatPage because it is the sidebar's content on
 * every page (chat, traces, users). Picking a session from a page that is not
 * chat navigates to chat — a click that silently changed the current session
 * off screen would look like a no-op.
 */
export function SessionTree() {
  const s = useAppState();
  const S = useStrings();
  const nav = useNavigate();
  const { pathname } = useLocation();
  const q = s.sidebarSearch.toLowerCase();
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const groups = useMemo(() => {
    const by: Record<string, { id: string; name: string; dir: string; sessions: { id: string; session: Session }[] }> = {};
    for (const [id, sess] of Object.entries(s.sessions)) {
      if (sess.archived) continue;
      if (q && !sess.title.toLowerCase().includes(q)) continue;
      const pid = sess.projectId || projectIdForDir(sess.dir);
      if (!by[pid]) by[pid] = { id: pid, name: projectNameForDir(sess.dir), dir: sess.dir, sessions: [] };
      by[pid].sessions.push({ id, session: sess });
    }
    const grouped = Object.values(by);
    for (const g of grouped) {
      g.sessions.sort((a, b) => b.session.ts - a.session.ts);
    }
    // Project order is when that directory first got a chat, not the newest
    // session — otherwise an old project jumps to the top on "new chat".
    grouped.sort((a, b) => {
      const aFirst = a.sessions[a.sessions.length - 1]?.session.ts || 0;
      const bFirst = b.sessions[b.sessions.length - 1]?.session.ts || 0;
      return bFirst - aFirst;
    });
    return grouped;
  }, [s.rev, s.sidebarSearch]);

  const pick = (id: string) => {
    switchTo(id);
    if (pathname !== "/chat") nav("/chat");
  };

  const addIn = (dir: string) => {
    newChatInDir(dir);
    if (pathname !== "/chat") nav("/chat");
  };

  return (
    <div className="s-list">
      {!groups.length && <div className="s-empty">{q ? S.chat.noMatch : S.chat.noSessions}</div>}
      {groups.map((g) => {
        const shut = !!collapsed[g.id];
        return (
          <div className="p-group" key={g.id}>
            <div className="p-head">
              <button type="button" className="p-caret" title={g.name}
                      onClick={() => setCollapsed((c) => ({ ...c, [g.id]: !c[g.id] }))}>
                {shut ? "▸" : "▾"}
              </button>
              <button type="button" className="p-main" onClick={() => setCollapsed((c) => ({ ...c, [g.id]: !c[g.id] }))}>
                <div className="p-title">
                  <span className="p-icon"><IconFolder /></span>
                  <span className="p-title-text">{g.name}</span>
                  <span className="p-count">{g.sessions.length}</span>
                </div>
              </button>
              <button type="button" className="p-add" title={S.chat.newInProject}
                      onClick={(e) => { e.stopPropagation(); addIn(g.dir); }}>
                <IconPlus />
              </button>
            </div>
            {!shut && g.sessions.map(({ id, session }) => (
              <div key={id} className={"s-item" + (id === s.curSess ? " active" : "")} onClick={() => pick(id)}>
                <div className="s-main">
                  <span className="ti">{session.title}</span>
                </div>
                <span className="mt">{agoStr(session.ts)}</span>
                <div className="s-actions">
                  <button className="db" title={S.chat.rename} onClick={(e) => { e.stopPropagation(); promptRename(id); }}><IconPencil /></button>
                  <button className="db" title={S.chat.archive} onClick={(e) => { e.stopPropagation(); archiveSession(id); }}><IconArchive /></button>
                  <button className="db" title={S.common.delete} onClick={(e) => { e.stopPropagation(); deleteSession(id); }}><IconClose /></button>
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export function promptRename(id: string) {
  const sess = getState().sessions[id];
  if (!sess) return;
  const name = window.prompt(getStrings().chat.renamePrompt, sess.title);
  if (name) renameSession(id, name);
}
