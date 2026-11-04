import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { Outlet, useLocation, useNavigate } from "react-router";
import * as api from "../api";
import { loadAuthStatus, loadCronJobs, loadPlugins, loadPrefs, loadProfiles, loadStatus } from "../data";
import { useStrings } from "../i18n";
import { fmtN, fmtTime } from "../lib/format";
import { PluginsPanel } from "../panels/PluginsPanel";
import { primeSettings, SettingsModal } from "../panels/SettingsModal";
import { loadSessions, newChat } from "../sessions";
import {
  Logo, IconPlus, IconSearch, IconClose, IconClock, IconPlug,
  IconSidebar, IconArchive, IconGear, IconProfiles, IconUser, IconLogout,
  IconChat, IconDatabase, IconFolderPlus,
} from "../icons";
import { clampSidebarWidth, setState, useAppState, type Session } from "../store";
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
  const S = useStrings();
  const active = pathname.startsWith("/traces")
    ? "traces"
    : pathname.startsWith("/users")
      ? "users"
      : pathname.startsWith("/assistant")
        ? "assistant"
        : "chat";
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
  active: "chat" | "traces" | "users" | "assistant";
  list?: React.ReactNode;
  children: React.ReactNode;
}) {
  const s = useAppState();
  const S = useStrings();
  const nav = useNavigate();

  // Every page needs the same server state, and any of them can be the first
  // one loaded (a deep link to /traces is a cold start too).
  useEffect(() => {
    void (async () => {
      await Promise.all([loadStatus(), loadAuthStatus()]);
      await loadPrefs();
      await loadSessions();
      await Promise.all([loadProfiles(), loadPlugins(), loadCronJobs()]);
    })();
  }, []);

  const usage = accountUsage(s.sessions);
  const [searchOpen, setSearchOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const asideRef = useRef<HTMLElement | null>(null);
  const showSearch = active === "chat" || active === "traces";

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
          <button className="side-nav-item" onClick={() => void primeSettings("cron")} title={S.nav.cron}>
            <span className="nav-icon"><IconClock /></span>
            <span className="nav-label">{S.nav.cron}</span>
            {s.cronJobs.length ? <span className="nav-badge">{s.cronJobs.length}</span> : null}
          </button>
          <button className="side-nav-item" onClick={() => { setState({ pluginsPanelOpen: true }); void loadPlugins(); }}
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
               ["assistant", S.nav.assistant, <IconChat key="asst" />, "assistant"],
               ["memory", S.nav.memory, <IconDatabase key="m" />, "memory"],
               ["profiles", S.nav.profile, <IconProfiles key="p" />, "profiles"],
               ["users", S.nav.users, <IconUser key="u" />, "users"],
               ["cron", S.nav.cron, <IconClock key="c" />, "cron"],
               ["archived", S.nav.archivedSessions, <IconArchive key="a" />, "archived"]] as const)
              .filter(([, , , tab]) => tab !== "users" || s.accountRole === "admin")
              .map(([key, label, icon, tab]) => (
                <button key={key} className="account-action"
                        onClick={() => {
                          setState({ accountPanelOpen: false });
                          if (tab === "users") nav("/users");
                          else void primeSettings(tab);
                        }}>
                  {icon}<span>{label}</span>
                </button>
              ))}
            <button className="account-action"
                    onClick={() => { setState({ accountPanelOpen: false }); void signOut(); }}>
              <IconLogout /><span>{S.settings.logout}</span>
            </button>
          </div>
          <button className="account-entry" onClick={() => setState({ accountPanelOpen: !s.accountPanelOpen })} title={S.nav.account}>
            <span className="account-avatar"><IconUser /></span>
            {/* The signed-in account, not the model profile. The profile used
                to be here and answered a question nobody asks of an avatar —
                and now that the same install can have several accounts, whose
                conversations these are is the thing worth naming. */}
            <span className="account-meta">
              <span className="account-name">{s.accountId}</span>
              <span className="account-sub">
                {s.accountRole === "admin" ? S.users.roleAdmin : S.users.roleUser}
              </span>
            </span>
          </button>
        </div>
        <div className="sidebar-resize-handle" role="separator" aria-orientation="vertical"
             title={S.workspace.resize} onPointerDown={startResize}
             onDoubleClick={() => { localStorage.removeItem("ag_sidebar_w"); setState({ sidebarWidth: clampSidebarWidth(280) }); }} />
      </aside>
      <div className={"account-pop-backdrop" + (s.accountPanelOpen ? " open" : "")}
           onClick={() => setState({ accountPanelOpen: false })} />
      {children}
      {s.dirModal.open && <DirModal />}
      {s.workspaceAdd.open && <AddWorkspaceDialog />}
      {s.settingsModal.open && <SettingsModal />}
      {s.pluginsPanelOpen && <PluginsPanel />}
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
