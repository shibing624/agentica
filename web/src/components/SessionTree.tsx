import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { getStrings, useStrings } from "../i18n";
import { agoStr, lastQueryTs, sessionSideStatus } from "../lib/format";
import { collapseSessionRows, sessionMatches } from "../lib/sessionRows";
import { archiveSession, deleteSession, loadSessions, newChatInDir, renameSession, switchTo, syncSessionStatus } from "../sessions";
import {
  IconArchive, IconChevronRight, IconClose, IconFolder, IconMore, IconPencil, IconPlus, IconSpinner, IconTrash,
} from "../icons";
import {
  getState, normDir, projectIdForDir, projectNameForDir, setState, useAppState, type Session,
} from "../store";
import { deleteWorkspace, renameWorkspace } from "../workspaces";

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
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");
  const [editingSession, setEditingSession] = useState<string | null>(null);
  const labels = useMemo(() => {
    const m: Record<string, string> = {};
    for (const w of s.workspaces) if (w.name) m[projectIdForDir(w.dir)] = w.name;
    return m;
  }, [s.workspaces]);
  const groups = useMemo(() => {
    const by: Record<string, Group> = {};
    for (const [id, sess] of Object.entries(s.sessions)) {
      if (sess.archived) continue;
      const pid = sess.projectId || projectIdForDir(sess.dir);
      const name = labels[pid] || projectNameForDir(sess.dir);
      if (!sessionMatches({ id, session: sess }, name, q)) continue;
      if (!by[pid]) by[pid] = { id: pid, name, dir: normDir(sess.dir), sessions: [] };
      const running = !!(sess.running || s.streams[id] || s.goalRuns[id] || s.commandRuns[id]);
      by[pid].sessions.push({ id, session: { ...sess, running } });
    }
    // A workspace the user added is listed even before its first chat.
    for (const w of s.workspaces) {
      const pid = projectIdForDir(w.dir);
      if (by[pid]) continue;
      const name = w.name || projectNameForDir(w.dir);
      if (q && !name.toLocaleLowerCase().includes(q.toLocaleLowerCase())) continue;
      by[pid] = { id: pid, name, dir: normDir(w.dir), sessions: [] };
    }
    const grouped = Object.values(by);
    for (const g of grouped) {
      g.sessions.sort((a, b) => lastQueryTs(b.session) - lastQueryTs(a.session));
    }
    const first = (g: Group) => g.sessions.length ? Math.min(...g.sessions.map((x) => x.session.ts)) : Date.now();
    grouped.sort((a, b) => first(b) - first(a));
    return grouped;
  }, [s.rev, s.sidebarSearch]);

  const menuRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!menuFor) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuFor(null);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMenuFor(null); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuFor]);

  const commitRename = (g: Group) => {
    const text = renameText;
    setRenaming(null);
    void renameWorkspace(g.dir, text);
  };

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
            <div className={"p-head" + (menuFor === g.id ? " menu-open" : "")}>
              <button type="button" className={"p-caret" + (shut ? "" : " open")} title={g.name}
                      aria-label={shut ? S.chat.expandProject : S.chat.collapseProject}
                      onClick={() => setCollapsed((c) => ({ ...c, [g.id]: !c[g.id] }))}>
                <IconChevronRight />
              </button>
              {renaming === g.id ? (
                <input className="p-rename-input" autoFocus value={renameText}
                       placeholder={S.workspace.renamePrompt}
                       onChange={(e) => setRenameText(e.target.value)}
                       onBlur={() => commitRename(g)}
                       onKeyDown={(e) => {
                         if (e.nativeEvent.isComposing) return;
                         if (e.key === "Enter") { e.preventDefault(); commitRename(g); }
                         if (e.key === "Escape") { e.preventDefault(); setRenaming(null); }
                       }} />
              ) : (
                <button type="button" className="p-main" title={g.dir}
                        onClick={() => setCollapsed((c) => ({ ...c, [g.id]: !c[g.id] }))}
                        onDoubleClick={() => { setRenameText(g.name); setRenaming(g.id); }}>
                  <div className="p-title">
                    <span className="p-icon"><IconFolder /></span>
                    <span className="p-title-text">{g.name}</span>
                  </div>
                </button>
              )}
              {renaming !== g.id && (
                <div className="p-tools">
                  <span className="p-count">{g.sessions.length || ""}</span>
                  <div className="p-menu-wrap" ref={menuFor === g.id ? menuRef : undefined}>
                    <button type="button" className="p-more" title={S.workspace.more} aria-haspopup="menu"
                            aria-expanded={menuFor === g.id}
                            onClick={(e) => { e.stopPropagation(); setMenuFor(menuFor === g.id ? null : g.id); }}>
                      <IconMore />
                    </button>
                    {menuFor === g.id && (
                      <div className="p-menu" role="menu">
                        <button type="button" role="menuitem" className="p-menu-item"
                                onClick={() => { setMenuFor(null); setRenameText(g.name); setRenaming(g.id); }}>
                          <IconPencil /><span>{S.workspace.rename}</span>
                        </button>
                        <button type="button" role="menuitem" className="p-menu-item danger"
                                onClick={() => { setMenuFor(null); deleteWorkspace(g.dir, g.name); }}>
                          <IconTrash /><span>{S.workspace.delete}</span>
                        </button>
                      </div>
                    )}
                  </div>
                  <button type="button" className="p-add" title={S.chat.newInProject}
                          onClick={(e) => { e.stopPropagation(); addIn(g.dir); }}>
                    <IconPlus />
                  </button>
                </div>
              )}
            </div>
            {!shut && !g.sessions.length && <div className="s-hint">{S.workspace.emptyHint}</div>}
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
                    {editingSession === id ? (
                      <RenameInput className="s-rename-input" initial={session.title}
                                   onDone={(name) => {
                                     setEditingSession(null);
                                     if (name !== null) void renameSession(id, name);
                                   }} />
                    ) : (
                      <span className="ti" onDoubleClick={(e) => { e.stopPropagation(); setEditingSession(id); }}>
                        {session.title}
                      </span>
                    )}
                  </div>
                  {editingSession !== id && session.source === "cli" && <span className="s-source">CLI</span>}
                  {editingSession === id ? null : side === "busy" ? (
                    <span className="mt busy" title={S.chat.sessionBusy}><IconSpinner /></span>
                  ) : side === "unread" ? (
                    <span className="mt unread" title={S.chat.sessionUnread} />
                  ) : (
                    <span className="mt">{agoStr(lastQueryTs(session), S.chat)}</span>
                  )}
                  {editingSession !== id && (
                    <div className="s-actions">
                      <button className="db" title={S.chat.rename} onClick={(e) => { e.stopPropagation(); setEditingSession(id); }}><IconPencil /></button>
                      <button className="db" title={S.chat.archive} onClick={(e) => { e.stopPropagation(); archiveSession(id); }}><IconArchive /></button>
                      <button className="db danger" title={S.common.delete} onClick={(e) => { e.stopPropagation(); deleteSession(id); }}><IconClose /></button>
                    </div>
                  )}
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

/** Inline title editor. Enter or blur saves, Esc cancels (`onDone(null)`).
 *  Clicks and keys stay inside so the row underneath does not open or toggle. */
export function RenameInput({ initial, className, onDone }: {
  initial: string;
  className: string;
  onDone: (name: string | null) => void;
}) {
  const S = getStrings();
  const [text, setText] = useState(initial);
  const done = useRef(false);
  const finish = (name: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(name === null || !name.trim() ? null : name.trim());
  };
  return (
    <input className={className} autoFocus value={text} placeholder={S.chat.renamePrompt}
           aria-label={S.chat.rename}
           onFocus={(e) => e.currentTarget.select()}
           onChange={(e) => setText(e.target.value)}
           onClick={(e) => e.stopPropagation()}
           onDoubleClick={(e) => e.stopPropagation()}
           onBlur={() => finish(text)}
           onKeyDown={(e) => {
             e.stopPropagation();
             if (e.nativeEvent.isComposing) return;
             if (e.key === "Enter") { e.preventDefault(); finish(text); }
             if (e.key === "Escape") { e.preventDefault(); finish(null); }
           }} />
  );
}
