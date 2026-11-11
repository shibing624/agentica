import { useState } from "react";
import { useNavigate } from "react-router";
import { IconChevronDown, IconClock, IconTerminal } from "../icons";
import { useStrings } from "../i18n";
import { useAppState } from "../store";

/**
 * A turn no person wrote — a background command's or a delegated subagent's
 * report, another session's message — drawn as a quiet system notice on the
 * assistant's side, never as the user's bubble. The user's queries are the
 * one thing a reader scans for; a report the agent got is context for the
 * reply under it.
 *
 * The first line is the report's own header; the rest folds.
 */
export function AutomatedNotice({ text, ts }: { text: string; ts?: number }) {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  const nl = text.indexOf("\n");
  const head = (nl < 0 ? text : text.slice(0, nl)).replace(/^\[|\]$/g, "").trim();
  const body = nl < 0 ? "" : text.slice(nl + 1).trim();
  return (
    <div className={"auto-notice" + (open ? " open" : "")}>
      <button type="button" className="auto-notice-head" aria-expanded={open} disabled={!body}
              onClick={() => setOpen((v) => !v)}>
        <span className="auto-notice-icon"><IconTerminal /></span>
        <span className="auto-notice-label">{S.chat.automatedLabel}</span>
        <span className="auto-notice-title" title={head}>{head}</span>
        {ts != null && <span className="auto-notice-ts">{S.chat.msgStamp(ts)}</span>}
        {body && <span className="auto-notice-chev"><IconChevronDown /></span>}
      </button>
      {open && body && <pre className="auto-notice-body">{body}</pre>}
    </div>
  );
}

/**
 * A turn an automation delivered: the job's name and when, with a way to its
 * detail page. The instruction is the job's own (from the catalog); the raw
 * delivered message, which also carries the run's standing guidance for the
 * model, is what folds open when the job is gone.
 */
export function ScheduledRunCard({ job, text, ts }: { job: { id: string; name: string }; text: string; ts?: number }) {
  const S = useStrings();
  const A = S.automation;
  const nav = useNavigate();
  const s = useAppState();
  const [open, setOpen] = useState(false);
  const live = s.cronJobs.find((j) => j.id === job.id);
  const body = live ? live.prompt : text;
  return (
    <div className={"sched-card" + (open ? " open" : "")}>
      <div className="sched-card-head">
        <span className="sched-card-icon"><IconClock /></span>
        <span className="sched-card-label">{A.scheduledRun}</span>
        <span className="sched-card-name" title={job.name}>{live?.name || job.name}</span>
        {ts != null && <span className="sched-card-ts">{S.chat.msgStamp(ts)}</span>}
        {live ? (
          <button type="button" className="sched-card-link" onClick={() => nav(`/automations?task=${encodeURIComponent(job.id)}`)}>
            {A.openAutomation}
          </button>
        ) : (
          <span className="sched-card-gone">{A.automationGone}</span>
        )}
      </div>
      <button type="button" className={"sched-card-body" + (open ? " open" : "")} aria-expanded={open}
              onClick={() => setOpen((v) => !v)}>
        <span>{body}</span>
        <span className="sched-card-chev"><IconChevronDown /></span>
      </button>
    </div>
  );
}
