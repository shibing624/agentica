import { useMemo } from "react";
import { useLocation, useNavigate } from "react-router";
import { getStrings, useStrings } from "../i18n";
import { agoStr, shortenPath } from "../lib/format";
import { archiveSession, deleteSession, renameSession, switchTo } from "../sessions";
import { IconArchive, IconClose, IconFolder, IconPencil } from "../icons";
import {
  getState, projectIdForDir, projectNameForDir, useAppState, type Session,
} from "../store";

/**
 * The conversation list in the left sidebar, grouped by working directory.
 *
 * It lives here rather than in ChatPage because it is the sidebar's content on
 * every page, Traces included: the nav is the same nav, and a user who has just
 * read a trace usually wants to go back to that conversation. Picking one from
 * a page that is not chat therefore navigates to chat — the alternative is a
 * click that silently changes which session is current somewhere off screen.
 */
export function SessionTree() {
  const s = useAppState();
  const S = useStrings();
  const nav = useNavigate();
  const { pathname } = useLocation();
  const q = s.sidebarSearch.toLowerCase();
  const groups = useMemo(() => {
    const by: Record<string, { id: string; name: string; dir: string; sessions: { id: string; session: Session }[] }> = {};
    for (const [id, sess] of Object.entries(s.sessions)) {
      if (sess.archived) continue;
      if (q && !sess.title.toLowerCase().includes(q)) continue;
      const pid = sess.projectId || projectIdForDir(sess.dir);
      if (!by[pid]) by[pid] = { id: pid, name: projectNameForDir(sess.dir), dir: sess.dir, sessions: [] };
      by[pid].sessions.push({ id, session: sess });
    }
    return Object.values(by).sort((a, b) => (b.sessions[0]?.session.ts || 0) - (a.sessions[0]?.session.ts || 0));
  }, [s.rev, s.sidebarSearch]);

  const pick = (id: string) => {
    switchTo(id);
    if (pathname !== "/chat") nav("/chat");
  };

  return (
    <div className="s-list">
      {!groups.length && <div className="s-empty">{q ? S.chat.noMatch : S.chat.noSessions}</div>}
      {groups.map((g) => (
        <div className="p-group" key={g.id}>
          <div className="p-head">
            <div className="p-main">
              <div className="p-title">
                <span className="p-icon"><IconFolder /></span>
                <span className="p-title-text">{g.name}</span>
                <span className="p-count">{g.sessions.length}</span>
              </div>
              <div className="p-dir" title={g.dir}>{shortenPath(g.dir)}</div>
            </div>
          </div>
          {g.sessions.sort((a, b) => b.session.ts - a.session.ts).map(({ id, session }) => (
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
      ))}
    </div>
  );
}

export function promptRename(id: string) {
  const sess = getState().sessions[id];
  if (!sess) return;
  const name = window.prompt(getStrings().chat.renamePrompt, sess.title);
  if (name) renameSession(id, name);
}
