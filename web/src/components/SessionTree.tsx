import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { getStrings, useStrings } from "../i18n";
import { agoStr, lastQueryTs, sessionSideStatus } from "../lib/format";
import { collapseSessionRows, sessionMatches } from "../lib/sessionRows";
import { archiveSession, deleteSession, loadSessions, newChatInDir, renameSession, switchTo, syncSessionStatus } from "../sessions";
import { IconArchive, IconClose, IconFolder, IconPencil, IconPlus, IconSpinner } from "../icons";
import {
  getState, projectIdForDir, projectNameForDir, setState, useAppState, type Session,
} from "../store";

type Group = {
  id: string;
  name: string;
  dir: string;
  sessions: { id: string; session: Session }[];
};

export function SessionTree() {
  const s = useAppState();
  const S = useStrings();
  const nav = useNavigate();
  const { pathname } = useLocation();
  const q = s.sidebarSearch.trim();
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const groups = useMemo(() => {
    const by: Record<string, Group> = {};
    for (const [id, sess] of Object.entries(s.sessions)) {
      if (sess.archived) continue;
      const pid = sess.projectId || projectIdForDir(sess.dir);
      const name = projectNameForDir(sess.dir);
      if (!sessionMatches({ id, session: sess }, name, q)) continue;
      if (!by[pid]) by[pid] = { id: pid, name, dir: sess.dir, sessions: [] };
      const running = !!(sess.running || s.streams[id] || s.goalRuns[id] || s.commandRuns[id]);
      by[pid].sessions.push({ id, session: { ...sess, running } });
    }
    const grouped = Object.values(by);
    for (const g of grouped) {
      g.sessions.sort((a, b) => lastQueryTs(b.session) - lastQueryTs(a.session));
    }
    grouped.sort((a, b) => {
      const aFirst = Math.min(...a.sessions.map((x) => x.session.ts));
      const bFirst = Math.min(...b.sessions.map((x) => x.session.ts));
      return bFirst - aFirst;
    });
    return grouped;
  }, [s.rev, s.sidebarSearch]);

  useEffect(() => {
    const remoteBusy = Object.entries(s.sessions).some(
      ([id, sess]) => sess.running && !s.streams[id] && !s.goalRuns[id] && !s.commandRuns[id],
    );
    if (!remoteBusy) return;
    void syncSessionStatus();
    const timer = window.setInterval(() => { void syncSessionStatus(); }, 2500);
    return () => window.clearInterval(timer);
  }, [s.rev]);

  const pick = (id: string) => {
    switchTo(id);
    if (pathname !== "/chat") nav("/chat");
  };

  const addIn = (dir: string) => {
    newChatInDir(dir);
    if (pathname !== "/chat") nav("/chat");
  };

  const toggleScope = () => {
    const scope = s.sessionScope === "recent" ? "all" : "recent";
    setState({ sessionScope: scope });
    void loadSessions(scope);
  };

  return (
    <div className="s-list">
      <div className="session-scope-row">
        <span>{s.sessionScope === "recent" ? S.chat.recentSessions : S.chat.allSessions}</span>
        <button type="button" onClick={toggleScope} disabled={s.sessionsLoading}>
          {s.sessionScope === "recent" ? S.chat.showAllSessions : S.chat.showRecentSessions}
        </button>
      </div>
      {s.sessionsError && (
        <div className="session-error" role="alert">
          <span>{S.chat.loadSessionsFailed}</span>
          <button type="button" onClick={() => void loadSessions()}>{S.common.refresh}</button>
        </div>
      )}
      {!groups.length && (
        <div className="s-empty">
          {s.sessionsLoading ? S.common.loading : q ? S.chat.noMatch : S.chat.noSessions}
        </div>
      )}
      {groups.map((g) => {
        const shut = !!collapsed[g.id];
        const isExpanded = !!expanded[g.id];
        const rows = collapseSessionRows(g.sessions, s.curSess, isExpanded || !!q);
        return (
          <section className="p-group" key={g.id}>
            <div className="p-head">
              <button type="button" className="p-caret" title={g.name}
                      aria-label={shut ? S.chat.expandProject : S.chat.collapseProject}
                      onClick={() => setCollapsed((c) => ({ ...c, [g.id]: !c[g.id] }))}>
                {shut ? "▸" : "▾"}
              </button>
              <button type="button" className="p-main" title={g.dir}
                      onClick={() => setCollapsed((c) => ({ ...c, [g.id]: !c[g.id] }))}>
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
            {!shut && rows.visible.map(({ id, session }) => {
              const side = sessionSideStatus({
                id, curSess: s.curSess, unread: session.unread, running: session.running,
              });
              return (
                <div key={id} className={"s-item" + (id === s.curSess ? " active" : "")}
                     role="button" tabIndex={0}
                     onClick={() => pick(id)}
                     onKeyDown={(e) => {
                       if (e.key === "Enter" || e.key === " ") {
                         e.preventDefault();
                         pick(id);
                       }
                     }}>
                  <div className="s-main">
                    <span className="ti">{session.title}</span>
                  </div>
                  {session.source === "cli" && <span className="s-source">CLI</span>}
                  {side === "busy" ? (
                    <span className="mt busy" title={S.chat.sessionBusy}><IconSpinner /></span>
                  ) : side === "unread" ? (
                    <span className="mt unread" title={S.chat.sessionUnread} />
                  ) : (
                    <span className="mt">{agoStr(lastQueryTs(session), S.chat)}</span>
                  )}
                  <div className="s-actions">
                    <button className="db" title={S.chat.rename} onClick={(e) => { e.stopPropagation(); promptRename(id); }}><IconPencil /></button>
                    <button className="db" title={S.chat.archive} onClick={(e) => { e.stopPropagation(); archiveSession(id); }}><IconArchive /></button>
                    <button className="db danger" title={S.common.delete} onClick={(e) => { e.stopPropagation(); deleteSession(id); }}><IconClose /></button>
                  </div>
                </div>
              );
            })}
            {!shut && !q && (rows.hiddenCount > 0 || isExpanded) && (
              <button type="button" className="s-more"
                      onClick={() => setExpanded((value) => ({ ...value, [g.id]: !value[g.id] }))}>
                {isExpanded ? S.chat.showLessSessions : S.chat.showMore(rows.hiddenCount)}
              </button>
            )}
          </section>
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
