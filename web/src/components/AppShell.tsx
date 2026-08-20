import { useEffect } from "react";
import { useNavigate } from "react-router";
import { loadAuthStatus, loadCronJobs, loadPlugins, loadProfiles, loadStatus } from "../data";
import { useStrings } from "../i18n";
import { fmtN, fmtTime } from "../lib/format";
import { PluginsPanel } from "../panels/PluginsPanel";
import { primeSettings, SettingsModal } from "../panels/SettingsModal";
import { loadSessions, newChat } from "../sessions";
import {
  Logo, IconPlus, IconSearch, IconClose, IconClock, IconPlug, IconTrace,
  IconSidebar, IconArchive, IconGear, IconProfiles, IconUser,
} from "../icons";
import { setState, useAppState, type Session } from "../store";
import { DirModal } from "./DirModal";

/**
 * The frame every page inside the app shares: left navigation, the account
 * popover, and the modals any of that navigation can open.
 *
 * It exists because Traces is a view of this app and not a different app. It
 * used to be a whole second layout with its own brand block and its own "back
 * to chat" link, which meant the nav went away exactly when the user wanted to
 * act on what they had just read, and every panel reachable from the nav
 * (scheduled jobs, plugins, settings) was unreachable from there.
 *
 * `list` is the column under the nav. Chat puts its session tree there; traces
 * puts nothing, because its own session picker is a middle column of its own —
 * two session lists in one sidebar is a puzzle, not a shortcut.
 */
export function AppShell({
  active, list, children,
}: {
  active: "chat" | "traces";
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
      await loadStatus();
      await loadSessions();
      await Promise.all([loadProfiles(), loadPlugins(), loadCronJobs(), loadAuthStatus()]);
    })();
  }, []);

  const usage = accountUsage(s.sessions);

  return (
    <>
      <aside className={"sidebar" + (s.sidebarCollapsed ? " collapsed" : "")}>
        <div className="side-head">
          <div className="brand"><Logo /><span>Agentica</span></div>
          <button className="ib" onClick={() => setState({ sidebarCollapsed: !s.sidebarCollapsed })}
                  title={S.nav.collapse}><IconSidebar /></button>
        </div>
        <nav className="side-nav" aria-label="Primary">
          <button className="side-nav-item" onClick={() => { nav("/chat"); newChat(); }} title={S.nav.newChat}>
            <span className="nav-icon"><IconPlus /></span>
            <span className="nav-label">{S.nav.newChat}</span>
          </button>
          {active === "chat" && (
            <label className="side-nav-item side-search-item" title={S.nav.search}>
              <span className="nav-icon"><IconSearch /></span>
              <input placeholder={S.nav.searchSessions} value={s.sidebarSearch}
                     onChange={(e) => setState({ sidebarSearch: e.target.value })} />
              {s.sidebarSearch ? (
                <button className="search-clear"
                        onClick={(e) => { e.preventDefault(); setState({ sidebarSearch: "" }); }}>
                  <IconClose />
                </button>
              ) : null}
            </label>
          )}
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
          <button className={"side-nav-item" + (active === "traces" ? " active" : "")}
                  onClick={() => nav("/traces")} title={S.nav.traces}>
            <span className="nav-icon"><IconTrace /></span>
            <span className="nav-label">{S.nav.traces}</span>
          </button>
        </nav>
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
            {([["settings", S.nav.generalSettings, <IconGear key="g" />], ["profiles", S.nav.profile, <IconProfiles key="p" />],
               ["cron", S.nav.cron, <IconClock key="c" />], ["archived", S.nav.archivedSessions, <IconArchive key="a" />]] as const).map(
              ([tab, label, icon]) => (
                <button key={tab} className="account-action"
                        onClick={() => { setState({ accountPanelOpen: false }); void primeSettings(tab); }}>
                  {icon}<span>{label}</span>
                </button>
              ))}
          </div>
          <button className="account-entry" onClick={() => setState({ accountPanelOpen: !s.accountPanelOpen })} title={S.nav.account}>
            <span className="account-avatar"><IconUser /></span>
            <span className="account-meta">
              <span className="account-name">{s.serverProfile || "User"}</span>
              <span className="account-sub">{S.nav.accountSub}</span>
            </span>
          </button>
        </div>
      </aside>
      <div className={"account-pop-backdrop" + (s.accountPanelOpen ? " open" : "")}
           onClick={() => setState({ accountPanelOpen: false })} />
      {children}
      {s.dirModal.open && <DirModal />}
      {s.settingsModal.open && <SettingsModal />}
      {s.pluginsPanelOpen && <PluginsPanel />}
    </>
  );
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
