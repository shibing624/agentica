import { useState } from "react";
import { useNavigate } from "react-router";
import { useStrings } from "../i18n";
import { IconClock, IconPlus } from "../icons";
import { useAppState } from "../store";

/**
 * The conversation header's way to its automations (after dsh's task chip):
 * how many run here, a menu to open one, and "create one that runs here".
 * Without any it is just the clock button that creates one.
 */
export function SessionAutomations({ sessionId }: { sessionId: string }) {
  const S = useStrings();
  const A = S.automation;
  const s = useAppState();
  const nav = useNavigate();
  const [open, setOpen] = useState(false);
  const here = s.cronJobs.filter((j) => j.session_id === sessionId && j.status !== "completed");
  const create = () => nav(`/automations?new=1&session=${encodeURIComponent(sessionId)}`);
  if (!here.length) {
    return (
      <button type="button" className="ib" title={A.emptyAction} aria-label={A.emptyAction} onClick={create}>
        <IconClock />
      </button>
    );
  }
  return (
    <div className="sess-auto">
      <button type="button" className={"sess-auto-chip" + (open ? " on" : "")} title={A.chatBadgeTitle}
              aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <IconClock /><span>{A.chatBadge(here.length)}</span>
      </button>
      {open && (
        <>
          <div className="auto-menu-scrim" onClick={() => setOpen(false)} />
          <div className="auto-menu sess-auto-menu" role="menu">
            {here.map((j) => (
              <button key={j.id} type="button" role="menuitem"
                      onClick={() => { setOpen(false); nav(`/automations?task=${encodeURIComponent(j.id)}`); }}>
                <span className={"sess-auto-dot " + j.status + (j.running ? " running" : "")} />
                <span className="auto-menu-item"><span>{j.name}</span><span className="auto-menu-help">{j.schedule}</span></span>
              </button>
            ))}
            <div className="auto-menu-sep" />
            <button type="button" role="menuitem" onClick={() => { setOpen(false); create(); }}>
              <IconPlus /><span>{A.emptyAction}</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
}
