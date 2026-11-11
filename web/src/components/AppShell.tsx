import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { Outlet, useLocation, useNavigate } from "react-router";
import * as api from "../api";
import {
  loadAuthStatus, loadCronJobs, loadPlugins, loadPrefs, loadProfiles,
  loadStatus, loadUpdateCheck, triggerUpdateOrCheck,
} from "../data";
import { useStrings } from "../i18n";
import { fmtN, fmtTime } from "../lib/format";
import { primeSettings, SettingsModal } from "../panels/SettingsModal";
import { loadSessions, newChat, pollSessionsPulse, sessionPath, switchTo, syncFromServer } from "../sessions";
import {
  Logo, IconPlus, IconSearch, IconClose, IconClock, IconPlug,
  IconSidebar, IconArchive, IconGear, IconProfiles, IconUser, IconLogout,
  IconChat, IconDatabase, IconFolderPlus, IconDownload, IconSpinner, DefaultAvatar,
} from "../icons";
import { clampSidebarWidth, getState, sessionViews, setState, useAppState, type Session } from "../store";
import { openAddWorkspace } from "../workspaces";
import { AddWorkspaceDialog } from "./AddWorkspaceDialog";
import { ChangePasswordDialog } from "./ChangePasswordDialog";
import { DirModal } from "./DirModal";
import { SessionTree } from "./SessionTree";

/**
 * The frame every page inside the app shares: left navigation, the account
 * popover, and the modals any of that navigation can open.
 *
 * Trace is a session accessory, not a top-level tab: the only way in is the
 * "view trace" chip next to the conversation title, so this nav has no traces
 * item. The session tree still sits here on `/traces` so leaving a trace for
 * that conversation is one click.
 *
 * Chat / traces / users share one shell via `<AppLayout>` so navigating to a
 * trace does not remount this tree and refetch `/api/sessions` (that refetch
 * used to replace the sidebar and drop a conversation the list had not yet
 * included).
 *
 * `list` is the column under the nav — chat, traces, and users all put the
 * same session tree there.
 */
export function AppLayout() {
  const { pathname } = useLocation();
  const active = pathname.startsWith("/traces")
    ? "traces"
    : pathname.startsWith("/automations") ? "automations" : "chat";
  return (
    <AppShell
      active={active}
      list={<SessionTree />}
    >
      <Outlet />
    </AppShell>
  );
}

export function AppShell({
  active, list, children,
}: {
  active: "chat" | "traces" | "automations";
  list?: React.ReactNode;
  children: React.ReactNode;
}) {
  const s = useAppState();
  const S = useStrings();
  const nav = useNavigate();
  const activeAutomations = s.cronJobs.filter((j) => j.status === "active").length;

  // Every page needs the same server state, and any of them can be the first
  // one loaded (a deep link to /traces is a cold start too).
  useEffect(() => {
    void (async () => {
      try {
        await Promise.all([loadStatus(), loadAuthStatus(), loadUpdateCheck()]);
        await loadPrefs();
        await loadSessions();
        // `?new=1` / `?session=` are the chat's deep links. The automations
        // page uses the same names for its own create view and must keep them.
        const params = new URLSearchParams(
          window.location.pathname.startsWith("/automations") ? "" : window.location.search,
        );
        const targetSess = params.get("session");
        if (params.get("new") === "1") {
          newChat();
          nav("/chat", { replace: true });
        } else if (targetSess && getState().sessions[targetSess]) {
          switchTo(targetSess);
          nav(sessionPath(targetSess), { replace: true });
        } else {
          // The last open chat was restored by loadSessions; if it was left on
          // its trace, reopen the trace (the desktop app always loads /chat).
          const restored = getState().curSess;
          if (restored && window.location.pathname === "/chat" && sessionPath(restored) !== "/chat") {
            nav(sessionPath(restored), { replace: true });
          }
        }
      } finally {
        // After the redirect: the chat page records "left on chat" only once
        // booting is over, so it cannot overwrite the trace it is leaving for.
        setState({ booting: false });
      }
      await Promise.all([loadProfiles(), loadPlugins(), loadCronJobs()]);
    })();
  }, []);

  // Desktop Dock menu actions arrive as a DOM event when the window is already
  // open so switching sessions or starting a new chat does not reload the SPA.
  useEffect(() => {
    const onDesktopNav = (ev: Event) => {
      const detail = (ev as CustomEvent<{ action?: string; sessionId?: string }>).detail;
      if (!detail) return;
      if (detail.action === "new-chat") {
        nav("/chat");
        newChat();
      } else if (detail.action === "session" && detail.sessionId) {
        const sid = detail.sessionId;
        if (getState().sessions[sid]) {
          switchTo(sid);
          nav(sessionPath(sid));
        } else {
          nav("/chat");
          void loadSessions().then(() => {
            if (!getState().sessions[sid]) return;
            switchTo(sid);
            nav(sessionPath(sid));
          });
        }
      }
    };
    window.addEventListener("agentica:desktop-nav", onDesktopNav);
    return () => window.removeEventListener("agentica:desktop-nav", onDesktopNav);
  }, [nav]);

  // Keep the window / tab title and <html> session dataset in sync with the
  // active conversation so browser tabs, the macOS Window list, and the
  // Desktop Dock menu all reflect what is currently open.
  const curSess = s.curSess ? s.sessions[s.curSess] : null;
  useEffect(() => {
    const title = curSess?.title?.trim() || "";
    const prefix = curSess?.running ? "● " : "";
    document.title = title ? `${prefix}${title} — Agentica` : "Agentica";
    const root = document.documentElement;
    if (s.curSess) root.dataset.sessionId = s.curSess;
    else delete root.dataset.sessionId;
    if (title) root.dataset.sessionTitle = title;
    else delete root.dataset.sessionTitle;
    if (curSess?.running) root.dataset.sessionRunning = "1";
    else delete root.dataset.sessionRunning;
  }, [s.curSess, curSess?.title, curSess?.running]);

  // Another client (browser tab, desktop window, CLI) may have run a turn.
  // Sync on window focus / tab visibility rather than polling `/api/sessions`
  // every 5s in the background; active remote runs already poll their own
  // `/api/sessions/{id}/run` status in `SessionTree`.
  useEffect(() => {
    const sync = () => { if (document.visibilityState === "visible") void syncFromServer(); };
    // Positions are written on a short timer; a closing window must not lose the last one.
    const flushViews = () => sessionViews.flush();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void pollSessionsPulse();
    }, 2000);
    window.addEventListener("focus", sync);
    document.addEventListener("visibilitychange", sync);
    window.addEventListener("pagehide", flushViews);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", sync);
      document.removeEventListener("visibilitychange", sync);
      window.removeEventListener("pagehide", flushViews);
    };
  }, []);

  const usage = accountUsage(s.sessions);
  const [searchOpen, setSearchOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const asideRef = useRef<HTMLElement | null>(null);
  const showSearch = active === "chat" || active === "traces";

  // On <html> so every page (chat topbar, trace page, users) can clear the
  // floating seat without each one reading the store for it.
  useEffect(() => {
    const root = document.documentElement;
    if (s.sidebarCollapsed) root.dataset.sidebarCollapsed = "";
    else delete root.dataset.sidebarCollapsed;
  }, [s.sidebarCollapsed]);

  const startResize = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (s.sidebarCollapsed) return;
    e.preventDefault();
    const left = asideRef.current?.getBoundingClientRect().left ?? 0;
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    setDragging(true);
    let width = s.sidebarWidth;
    const move = (ev: PointerEvent) => {
      width = clampSidebarWidth(ev.clientX - left);
      asideRef.current?.style.setProperty("--sidebar-w", width + "px");
    };
    const up = () => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      setDragging(false);
      localStorage.setItem("ag_sidebar_w", String(width));
      setState({ sidebarWidth: width });
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
  };

  return (
    <>
      <aside ref={asideRef}
             className={"sidebar" + (s.sidebarCollapsed ? " collapsed" : "") + (dragging ? " resizing" : "")}
             style={{ "--sidebar-w": s.sidebarWidth + "px" } as React.CSSProperties}>
        <div className="side-head">
          <div className="brand"><Logo /><span>Agentica</span></div>
          <button className="ib" onClick={() => setState({ sidebarCollapsed: !s.sidebarCollapsed })}
                  title={s.sidebarCollapsed ? S.nav.expand : S.nav.collapse}><IconSidebar /></button>
        </div>
        <button className="side-new" onClick={() => { nav("/chat"); newChat(); }} title={S.nav.newChat}>
          <IconPlus /><span className="nav-label">{S.nav.newChat}</span>
        </button>
        <nav className="side-nav" aria-label="Primary">
          <button className={"side-nav-item" + (active === "automations" ? " active" : "")}
                  onClick={() => nav("/automations")} title={S.nav.cron}>
            <span className="nav-icon"><IconClock /></span>
            <span className="nav-label">{S.nav.cron}</span>
            {activeAutomations ? <span className="nav-badge">{activeAutomations}</span> : null}
          </button>
          <button className="side-nav-item" onClick={() => void primeSettings("plugins")}
                  title={S.nav.plugins}>
            <span className="nav-icon"><IconPlug /></span>
            <span className="nav-label">{S.nav.plugins}</span>
          </button>
        </nav>
        <div className="side-section">
          <span className="side-section-title">{S.workspace.title}</span>
          <div className="side-section-actions">
            {showSearch && (
              <button className={"ib sm" + (searchOpen || s.sidebarSearch ? " on" : "")}
                      title={S.nav.search} onClick={() => setSearchOpen((v) => !v)}>
                <IconSearch />
              </button>
            )}
            <button className="ib sm" title={S.workspace.add} onClick={() => openAddWorkspace()}>
              <IconFolderPlus />
            </button>
          </div>
        </div>
        {showSearch && (searchOpen || !!s.sidebarSearch) && (
          <label className="side-search">
            <IconSearch />
            <input autoFocus placeholder={S.nav.searchSessions} value={s.sidebarSearch}
                   onChange={(e) => setState({ sidebarSearch: e.target.value })}
                   onKeyDown={(e) => {
                     if (e.key === "Escape") { setState({ sidebarSearch: "" }); setSearchOpen(false); }
                   }} />
            {s.sidebarSearch ? (
              <button className="search-clear"
                      onClick={(e) => { e.preventDefault(); setState({ sidebarSearch: "" }); }}>
                <IconClose />
              </button>
            ) : null}
          </label>
        )}
        {list}
        <div className="account-wrap">
          <div className={"account-pop" + (s.accountPanelOpen ? " open" : "")} onClick={(e) => e.stopPropagation()}>
            <div className="account-pop-usage">
              <div className="ctx-tip-header">{S.nav.usageTitle} <span className="account-usage-scope">{S.nav.allSessions}</span></div>
              <div className="ctx-tip-row"><span>{S.nav.input}</span><span>{fmtN(usage.tokIn)}</span></div>
              <div className="ctx-tip-row"><span>{S.nav.output}</span><span>{fmtN(usage.tokOut)}</span></div>
              <div className="ctx-tip-row ctx-tip-total"><span>{S.nav.total}</span><span>{fmtN(usage.tokTotal)}</span></div>
              {usage.totalTime > 0 && (
                <div className="ctx-tip-row"><span>{S.nav.elapsed}</span><span>{fmtTime(usage.totalTime)}</span></div>
              )}
            </div>
            {([["settings", S.nav.generalSettings, <IconGear key="g" />, "settings"],
               ["profiles", S.nav.profile, <IconProfiles key="p" />, "profiles"],
               ["memory", S.nav.memory, <IconDatabase key="m" />, "memory"],
               ["plugins", S.nav.plugins, <IconPlug key="pl" />, "plugins"],
               ["assistant", S.nav.assistant, <IconChat key="asst" />, "assistant"],
               ["users", S.nav.users, <IconUser key="u" />, "users"],
               ["archived", S.nav.archivedSessions, <IconArchive key="a" />, "archived"]] as const)
              .filter(([, , , tab]) => tab !== "users" || s.accountRole === "admin")
              .map(([key, label, icon, tab]) => (
                <button key={key} className="account-action"
                        onClick={() => {
                          setState({ accountPanelOpen: false });
                          void primeSettings(tab);
                        }}>
                  {icon}<span>{label}</span>
                </button>
              ))}
            <button className="account-action"
                    onClick={() => { setState({ accountPanelOpen: false }); void signOut(); }}>
              <IconLogout /><span>{S.settings.logout}</span>
            </button>
          </div>
          <div className="side-foot">
            <button
              type="button"
              className={"account-entry" + (s.accountPanelOpen ? " active" : "")}
              onClick={() => setState({ accountPanelOpen: !s.accountPanelOpen })}
              title={S.nav.account}
              aria-label={S.nav.account}
              aria-expanded={s.accountPanelOpen}
            >
              <span className="account-avatar"><DefaultAvatar /></span>
            </button>
            <button
              type="button"
              className={
                "side-update-btn"
                + (s.updateAvailable ? " has-update" : "")
                + (s.updateBusy !== "idle" ? " busy" : "")
              }
              disabled={s.updateBusy !== "idle"}
              title={
                s.updateAvailable && s.latestVersion
                  ? S.nav.updateHint(s.serverVersion || "-", s.latestVersion)
                  : S.nav.checkUpdateTitle
              }
              onClick={() => void triggerUpdateOrCheck()}
            >
              {s.updateBusy === "updating" ? (
                <>
                  <IconSpinner />
                  <span>{S.nav.updating}</span>
                </>
              ) : s.updateBusy === "checking" ? (
                <>
                  <IconSpinner />
                  <span>{S.nav.checkingUpdate}</span>
                </>
              ) : s.updateAvailable && s.latestVersion ? (
                <>
                  <IconDownload />
                  <span>{S.nav.updateTo(s.latestVersion)}</span>
                  <span className="side-update-dot" aria-hidden="true" />
                </>
              ) : (
                <span>{s.serverVersion ? "v" + s.serverVersion : S.nav.checkUpdate}</span>
              )}
            </button>
          </div>
        </div>
        <div className="sidebar-resize-handle" role="separator" aria-orientation="vertical"
             title={S.workspace.resize} onPointerDown={startResize}
             onDoubleClick={() => { localStorage.removeItem("ag_sidebar_w"); setState({ sidebarWidth: clampSidebarWidth(280) }); }} />
      </aside>
      <div className={"account-pop-backdrop" + (s.accountPanelOpen ? " open" : "")}
           onClick={() => setState({ accountPanelOpen: false })} />
      {children}
      {/* After the page on purpose: on macOS the window's drag regions are
          applied in document order, so a control has to come after every
          drag row it overlaps (sidebar head, page topbar) to stay clickable. */}
      {s.sidebarCollapsed && (
        <div className="side-seat">
          <button className="ib" onClick={() => setState({ sidebarCollapsed: false })}
                  title={S.nav.expand} aria-label={S.nav.expand}><IconSidebar /></button>
          <button className="ib" onClick={() => { nav("/chat"); newChat(); }}
                  title={S.nav.newChat} aria-label={S.nav.newChat}><IconPlus /></button>
        </div>
      )}
      {s.dirModal.open && <DirModal />}
      {s.workspaceAdd.open && <AddWorkspaceDialog />}
      {s.settingsModal.open && <SettingsModal />}
      <ChangePasswordDialog />
    </>
  );
}

async function signOut() {
  await api.logoutApi();
  window.location.href = "/login";
}

function accountUsage(sessions: Record<string, Session>) {
  let tokIn = 0, tokOut = 0, tokTotal = 0, totalTime = 0;
  for (const sess of Object.values(sessions)) {
    tokIn += sess.tokIn || 0;
    tokOut += sess.tokOut || 0;
    tokTotal += sess.tokTotal || 0;
    totalTime += sess.totalTime || 0;
  }
  return { tokIn, tokOut, tokTotal, totalTime };
}
