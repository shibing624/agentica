import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { getStrings, useStrings } from "../i18n";
import { agoStr, lastQueryTs, sessionSideStatus } from "../lib/format";
import { collapseSessionRows, sessionMatches, sessionSourceBadge, sessionSourceTitle, type SessionContentHit } from "../lib/sessionRows";
import {
  archiveSession, deleteSession, forkSession, loadSessions, newChatInDir, openSessionAtTurn,
  pinSession, renameSession, searchAllSessionsContent, sessionPath, switchTo, syncSessionStatus,
} from "../sessions";
import {
  IconArchive, IconChevronRight, IconCopyOutline, IconFolder, IconFork, IconMore, IconPencil, IconPin, IconPlus,
  IconClock, IconSpinner, IconTrash,
} from "../icons";
import {
  getState, normDir, projectIdForDir, projectNameForDir, setState, showToast, useAppState, type Session,
} from "../store";
import { deleteWorkspace, renameWorkspace } from "../workspaces";
import { isImeKey } from "../lib/ime";

type Group = {
  id: string;
  name: string;
  dir: string;
  sessions: { id: string; session: Session }[];
};

/** Matches listed under a session before "N more". */
const HITS_PREVIEW = 3;

export function SessionTree() {
  const s = useAppState();
  // Conversations an automation delivers into (not ended), for the row mark.
  const automatedSessions = useMemo(
    () => new Set(s.cronJobs.filter((j) => j.session_id && j.status !== "completed").map((j) => j.session_id)),
    [s.cronJobs],
  );
  const S = useStrings();
  const nav = useNavigate();
  const { pathname, search } = useLocation();
  const q = s.sidebarSearch.trim();
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");
  const [editingSession, setEditingSession] = useState<string | null>(null);
  const [rowMenu, setRowMenu] = useState<{ id: string; anchor: DOMRect } | null>(null);
  const [contentHits, setContentHits] = useState<Record<string, SessionContentHit[]>>({});
  const [searchingContent, setSearchingContent] = useState(false);
  // Sessions whose hit list is shown in full instead of the first few.
  const [allHits, setAllHits] = useState<Record<string, boolean>>({});
  // Read at request time: opening a hit switches the chat, and re-running the
  // search for that would blank the list the user is clicking through.
  const focusSessRef = useRef(s.curSess);
  focusSessRef.current = s.curSess;

  useEffect(() => {
    setAllHits({});
    if (!q) {
      setContentHits({});
      setSearchingContent(false);
      if (getState().chatSearchTarget) setState({ chatSearchTarget: null });
      return;
    }
    setSearchingContent(true);
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => {
      void searchAllSessionsContent(q, ctrl.signal, focusSessRef.current).then((res) => {
        if (ctrl.signal.aborted) return;
        setContentHits(res);
        setSearchingContent(false);
      });
    }, 180);
    return () => {
      window.clearTimeout(timer);
      ctrl.abort();
    };
  }, [q, s.sessionScope]);

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
      if (!sessionMatches({ id, session: sess }, name, q, contentHits[id])) continue;
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
      // Pinned first, then most recent.
      g.sessions.sort((a, b) => (Number(!!b.session.pinned) - Number(!!a.session.pinned))
        || (lastQueryTs(b.session) - lastQueryTs(a.session)));
    }
    const first = (g: Group) => g.sessions.length ? Math.min(...g.sessions.map((x) => x.session.ts)) : Date.now();
    grouped.sort((a, b) => first(b) - first(a));
    return grouped;
  }, [s.rev, s.sidebarSearch, contentHits]);

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

  const remoteBusyKey = Object.entries(s.sessions)
    .filter(([id, sess]) => sess.running && !s.streams[id] && !s.goalRuns[id] && !s.commandRuns[id])
    .map(([id]) => id)
    .sort()
    .join(",");
  useEffect(() => {
    if (!remoteBusyKey) return;
    const timer = window.setInterval(() => { void syncSessionStatus(); }, 2500);
    return () => window.clearInterval(timer);
  }, [remoteBusyKey]);

  const pick = (id: string) => {
    const firstHit = q ? contentHits[id]?.[0] : undefined;
    if (firstHit) {
      void openSessionAtTurn(id, firstHit.turn, q, firstHit.role);
    } else {
      switchTo(id);
      // Back to the page this session was left on (chat or its trace).
      const path = sessionPath(id);
      if (pathname + search !== path) nav(path);
      return;
    }
    if (pathname !== "/chat") nav("/chat");
  };

  const pickHit = (id: string, hit: SessionContentHit) => {
    void openSessionAtTurn(id, hit.turn, q, hit.role);
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
          {s.sessionsLoading || searchingContent ? S.common.loading : q ? S.chat.noMatch : S.chat.noSessions}
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
                         if (isImeKey(e.nativeEvent)) return;
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
              const sourceBadge = sessionSourceBadge(session.source, S.chat.sourceLabels);
              const hits = q ? contentHits[id] : undefined;
              // The session being read lists every match so the next one is a
              // click away; the others show a few and expand on request.
              const showAll = !!allHits[id] || id === s.curSess;
              const shown = hits ? (showAll ? hits : hits.slice(0, HITS_PREVIEW)) : [];
              return (
                <div key={id} className="s-item-block">
                <div
                     className={"s-item" + (id === s.curSess ? " active" : "") + (session.pinned ? " pinned" : "")
                       + (rowMenu?.id === id ? " menu-open" : "") + (editingSession === id ? " editing" : "")}
                     role="button" tabIndex={0}
                     onClick={() => pick(id)}
                     onKeyDown={(e) => {
                       if (e.key === "Enter" || e.key === " ") {
                         e.preventDefault();
                         pick(id);
                       }
                     }}>
                  {/* dsh's leading status slot: running spinner / unread dot
                      sit before the title; the time stays on the right. */}
                  {editingSession !== id && side !== "idle" && (
                    <span className={"s-lead " + side} title={side === "busy" ? S.chat.sessionBusy : S.chat.sessionUnread}>
                      {side === "busy" && <IconSpinner />}
                    </span>
                  )}
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
                  {editingSession !== id && automatedSessions.has(id) && (
                    <span className="s-auto-mark" title={S.automation.sessionMark}><IconClock /></span>
                  )}
                  {editingSession !== id && sourceBadge && (
                    <span className={"s-source" + (sourceBadge.isIm ? " im" : "")}
                          title={sessionSourceTitle(session.source, session.origin, S.chat.sourceLabels, S.chat)}>
                      {sourceBadge.label}
                    </span>
                  )}
                  {editingSession !== id && (
                    <span className="mt">{agoStr(lastQueryTs(session), S.chat)}</span>
                  )}
                  {editingSession !== id && session.pinned && (
                    <span className="s-pin-mark" title={S.chat.pinnedTip}><IconPin filled /></span>
                  )}
                  {editingSession !== id && (
                    <div className="s-actions">
                      <button type="button" className="db" title={S.chat.sessionMore} aria-label={S.chat.sessionMore}
                              aria-haspopup="menu" aria-expanded={rowMenu?.id === id}
                              onClick={(e) => {
                                e.stopPropagation();
                                const anchor = e.currentTarget.getBoundingClientRect();
                                setRowMenu(rowMenu?.id === id ? null : { id, anchor });
                              }}>
                        <IconMore />
                      </button>
                      <button type="button" className="db" title={S.chat.archiveSession} aria-label={S.chat.archiveSession}
                              onClick={(e) => { e.stopPropagation(); archiveSession(id); }}>
                        <IconArchive />
                      </button>
                      <button type="button" className={"db" + (session.pinned ? " on" : "")}
                              title={session.pinned ? S.chat.unpinSession : S.chat.pinSession}
                              aria-label={session.pinned ? S.chat.unpinSession : S.chat.pinSession} aria-pressed={!!session.pinned}
                              onClick={(e) => { e.stopPropagation(); pinSession(id, !session.pinned); }}>
                        <IconPin filled={!!session.pinned} />
                      </button>
                    </div>
                  )}
                </div>
                {hits && hits.length > 0 && (
                  <div className="s-hits">
                    {shown.map((hit, idx) => (
                      <button
                        key={`${hit.turn}:${hit.role}:${idx}`}
                        type="button"
                        className={"s-hit" + (s.chatSearchTarget?.sessionId === id && s.chatSearchTarget?.turn === hit.turn && s.chatSearchTarget?.role === hit.role ? " active" : "")}
                        onClick={(e) => { e.stopPropagation(); pickHit(id, hit); }}
                        title={hit.snippet}
                      >
                        <span className={"s-hit-tag " + hit.role}>
                          {hit.role === "user" ? S.chat.searchTurnQ(hit.turn) : S.chat.searchTurnA(hit.turn)}
                        </span>
                        <span className="s-hit-text">{highlightSnippet(hit.snippet, q)}</span>
                      </button>
                    ))}
                    {!showAll && hits.length > HITS_PREVIEW && (
                      <button
                        type="button"
                        className="s-hit-more"
                        onClick={(e) => { e.stopPropagation(); setAllHits((v) => ({ ...v, [id]: true })); }}
                      >
                        {S.chat.searchMoreHits(hits.length - HITS_PREVIEW)}
                      </button>
                    )}
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
      {rowMenu && s.sessions[rowMenu.id] && (
        <SessionMenu
          id={rowMenu.id}
          anchor={rowMenu.anchor}
          pinned={!!s.sessions[rowMenu.id].pinned}
          onClose={() => setRowMenu(null)}
          onRename={() => setEditingSession(rowMenu.id)}
        />
      )}
    </div>
  );
}

/**
 * The row's `···` menu (deepseek-harness style). Fixed-positioned so the
 * sidebar's scroll box cannot clip it; opens below the button, or above it
 * when there is no room, and closes on outside click, Esc or scroll.
 */
function SessionMenu({ id, anchor, pinned, onClose, onRename }: {
  id: string;
  anchor: DOMRect;
  pinned: boolean;
  onClose: () => void;
  onRename: () => void;
}) {
  const S = useStrings();
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number }>({ left: anchor.left, top: anchor.bottom + 4 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const h = el.offsetHeight;
    const w = el.offsetWidth;
    const below = anchor.bottom + 4;
    const top = below + h > window.innerHeight - 8 ? Math.max(8, anchor.top - h - 4) : below;
    const left = Math.min(Math.max(8, anchor.left - 8), window.innerWidth - w - 8);
    setPos({ left, top });
    el.querySelector<HTMLButtonElement>(".p-menu-item")?.focus();
  }, [anchor]);
  useEffect(() => {
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) onClose(); };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); return; }
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      e.preventDefault();
      const items = [...(ref.current?.querySelectorAll<HTMLButtonElement>(".p-menu-item") || [])];
      const at = items.indexOf(document.activeElement as HTMLButtonElement);
      const next = (at + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items[next]?.focus();
    };
    const onScroll = (e: Event) => { if (!ref.current?.contains(e.target as Node)) onClose(); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onClose);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onClose);
    };
  }, [onClose]);
  const run = (fn: () => void) => () => { onClose(); fn(); };
  return (
    <div ref={ref} className="p-menu s-menu" role="menu" style={{ left: pos.left, top: pos.top }}>
      <button type="button" role="menuitem" className="p-menu-item" onClick={run(() => pinSession(id, !pinned))}>
        <IconPin filled={pinned} /><span>{pinned ? S.chat.unpinSession : S.chat.pinSession}</span>
      </button>
      <button type="button" role="menuitem" className="p-menu-item" onClick={run(onRename)}>
        <IconPencil /><span>{S.chat.rename}</span>
      </button>
      <button type="button" role="menuitem" className="p-menu-item" onClick={run(() => { void forkSession(id); })}>
        <IconFork /><span>{S.chat.forkSession}</span>
      </button>
      <button type="button" role="menuitem" className="p-menu-item" onClick={run(() => archiveSession(id))}>
        <IconArchive /><span>{S.chat.archiveSession}</span>
      </button>
      <button type="button" role="menuitem" className="p-menu-item"
              onClick={run(() => { void navigator.clipboard.writeText(id).then(() => showToast(S.chat.sessionIdCopied)); })}>
        <IconCopyOutline /><span>{S.chat.copySessionId}</span>
      </button>
      <div className="p-menu-sep" role="separator" />
      <button type="button" role="menuitem" className="p-menu-item danger" onClick={run(() => deleteSession(id))}>
        <IconTrash /><span>{S.chat.deleteSession}</span>
      </button>
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
             if (isImeKey(e.nativeEvent)) return;
             if (e.key === "Enter") { e.preventDefault(); finish(text); }
             if (e.key === "Escape") { e.preventDefault(); finish(null); }
           }} />
  );
}

function highlightSnippet(snippet: string, query: string) {
  const needle = query.trim();
  if (!needle) return snippet;
  const lower = snippet.toLocaleLowerCase();
  const qLower = needle.toLocaleLowerCase();
  const at = lower.indexOf(qLower);
  if (at < 0) return snippet;
  return (
    <>
      {snippet.slice(0, at)}
      <mark>{snippet.slice(at, at + needle.length)}</mark>
      {snippet.slice(at + needle.length)}
    </>
  );
}
