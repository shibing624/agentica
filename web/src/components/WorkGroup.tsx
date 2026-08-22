import { useEffect, useRef, useState } from "react";
import { fmtDurationMs } from "../lib/format";
import { workSummary, type MsgPart } from "../lib/msgParts";
import { formatToolDisplay, layoutToolDisplay, parseToolArgs } from "../lib/toolDisplay";
import { IconCheck, IconChevronDown, IconSpinner, IconUser } from "../icons";
import { getState } from "../store";
import { useStrings } from "../i18n";
import { CodeFrame } from "./CodeFrame";
import { HIGHLIGHT_LIMIT } from "../lib/highlight";

function thinkLabel(
  streaming: boolean,
  S: { chat: { thinking: string; thinkingNow: string; stopping: string; reconnecting: string; preparingAttachments: string } },
): string {
  if (!streaming) return S.chat.thinking;
  const st = getState();
  const live = st.streams[st.curSess || ""];
  if (live?.cancelling) return S.chat.stopping;
  if (live?.reconnecting) return S.chat.reconnecting;
  if (live?.preparing) return S.chat.preparingAttachments;
  return S.chat.thinkingNow;
}

export function WorkGroup({
  items,
  isLast,
}: {
  items: MsgPart[];
  isLast: boolean;
}) {
  const S = useStrings();
  const { steps, ms, running } = workSummary(items);
  const [open, setOpen] = useState(isLast);
  const userToggled = useRef(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const [, tick] = useState(0);

  useEffect(() => {
    if (!userToggled.current) setOpen(isLast);
  }, [isLast]);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => tick((n) => n + 1), 250);
    return () => window.clearInterval(id);
  }, [running]);

  return (
    <div ref={rootRef} className={"work-group" + (running ? " running" : "")}>
      <button
        type="button"
        className="work-head"
        aria-expanded={open}
        onClick={() => {
          const willClose = open;
          userToggled.current = true;
          setOpen((v) => !v);
          if (willClose) {
            requestAnimationFrame(() => rootRef.current?.scrollIntoView({ block: "nearest" }));
          }
        }}
      >
        <span className="work-icon">{running ? <IconSpinner /> : <IconCheck />}</span>
        <span className="work-status">{running ? S.chat.workRunning : S.chat.workDone}</span>
        {steps > 0 && <span className="work-meta">{S.chat.workGroupSteps(steps)}</span>}
        {ms > 0 && <span className="work-meta">{fmtDurationMs(ms)}</span>}
        <span className="work-spacer" />
        <span className={"work-chevron" + (open ? " open" : "")}><IconChevronDown /></span>
      </button>
      {open && (
        <div className="work-body">
          {items.map((p, i) => {
            if (p.kind === "think") {
              return <ThinkRow key={i} part={p} streaming={p.ms == null && running} />;
            }
            if (p.kind === "tool") {
              return <ToolRow key={i} part={p} />;
            }
            return null;
          })}
        </div>
      )}
    </div>
  );
}

function ThinkRow({ part, streaming }: { part: Extract<MsgPart, { kind: "think" }>; streaming: boolean }) {
  const S = useStrings();
  const [open, setOpen] = useState(streaming);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (streaming) setOpen(true);
  }, [streaming]);

  const duration = part.ms != null
    ? part.ms
    : (streaming && part.t0 != null ? Date.now() - part.t0 : 0);

  return (
    <div ref={rootRef} className={"work-step" + (streaming ? " running" : "")}>
      <button
        type="button"
        className="work-step-head"
        aria-expanded={open}
        onClick={() => {
          const willClose = open;
          setOpen((v) => !v);
          if (willClose) {
            requestAnimationFrame(() => rootRef.current?.scrollIntoView({ block: "nearest" }));
          }
        }}
      >
        <span className="work-icon">{streaming ? <IconSpinner /> : <IconCheck />}</span>
        <span className="work-step-name">{thinkLabel(streaming, S)}</span>
        {duration > 0 && <span className="work-meta">{fmtDurationMs(duration)}</span>}
        <span className="work-spacer" />
        <span className={"work-chevron" + (open ? " open" : "")}><IconChevronDown /></span>
      </button>
      {open && part.text ? (
        <div className="think-text">{part.text}</div>
      ) : null}
    </div>
  );
}

/** Local file-read tools: call line is enough. write_todos success is the list. */
const HIDE_RESULT_TOOLS = new Set(["read_file", "glob", "grep"]);

function sessionCwd(): string {
  const st = getState();
  const cur = st.curSess ? st.sessions[st.curSess] : null;
  return (cur && cur.dir) || st.serverDir || "";
}

function ToolRow({ part }: { part: Extract<MsgPart, { kind: "tool" }> }) {
  const running = part.result == null;
  const resultText = part.result != null ? String(part.result) : "";
  const isError = resultText.startsWith("Error: ");
  const diff = part.diff || "";
  const showDiff = Boolean(diff) && !isError;
  const hideResult = HIDE_RESULT_TOOLS.has(part.name) || (part.name === "write_todos" && !isError);
  const display = formatToolDisplay(part.name, parseToolArgs(part.argsStr), sessionCwd());
  const displayLayout = layoutToolDisplay(part.name, display);
  const showArgsBody = displayLayout.body.trim().length > 0;
  const showResult = Boolean(resultText) && !showDiff && (isError || !hideResult);
  const duration = part.ms != null
    ? part.ms
    : (running && part.t0 != null ? Date.now() - part.t0 : 0);
  return (
    <details className={"work-step work-tool" + (running ? " running" : "")} open={running || showArgsBody || showResult || showDiff}>
      <summary className="work-step-head">
        <span className="work-icon">{running ? <IconSpinner /> : <IconCheck />}</span>
        <span className="work-step-name">{part.name}</span>
        {displayLayout.header ? <span className="step-tool-args" title={display}>{displayLayout.header}</span> : null}
        {duration > 0 && <span className="work-tool-ms">{fmtDurationMs(duration)}</span>}
      </summary>
      {showArgsBody ? (
        <pre className={displayLayout.bodyKind === "call" ? "step-tool-call" : "step-pre"}>{displayLayout.body}</pre>
      ) : null}
      {showDiff ? (
        <div className="work-tool-diff">
          <CodeFrame code={diff} language="diff" highlight={diff.length <= HIGHLIGHT_LIMIT} />
        </div>
      ) : null}
      {showResult ? <pre className="step-pre out">{resultText}</pre> : null}
    </details>
  );
}

export function SteerChip({ text, ts }: { text: string; ts?: number }) {
  const S = useStrings();
  return (
    <div className="steer-chip">
      <div className="steer-chip-box">
        <div className="steer-chip-row">
          <IconUser />
          <p className="steer-chip-text">
            <span className="steer-chip-label">{S.chat.userSteering}</span>
            {text}
          </p>
        </div>
      </div>
      {ts != null && (
        <span className="steer-chip-meta">{S.chat.msgStamp(ts)}</span>
      )}
    </div>
  );
}
