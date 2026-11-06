import { useState } from "react";
import { IconChevronDown, IconTerminal } from "../icons";
import { useStrings } from "../i18n";

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
