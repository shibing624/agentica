import { useEffect, useRef, useState, type JSX } from "react";
import { useNavigate } from "react-router";
import { fmtDurationMs } from "../lib/format";
import { workSummary, type MsgPart } from "../lib/msgParts";
import { formatToolDisplay, layoutToolDisplay, parseToolArgs } from "../lib/toolDisplay";
import {
  diffStats, doneTitle, firstLine, groupCategories, parseTaskResult, parseTodos, runningTool, todoProgress,
  toolCategory, type Category,
} from "../lib/workActivity";
import {
  IconAlert, IconBook, IconBrain, IconChevronDown, IconFile, IconGlobe, IconListCheck, IconPencil, IconSearch,
  IconExternal, IconSpinner, IconTerminal, IconUser, IconUsers,
} from "../icons";
import { getState, type ApprovalDecision, type ApprovalRequest } from "../store";
import { useStrings } from "../i18n";
import { CodeFrame } from "./CodeFrame";
import { HIGHLIGHT_LIMIT } from "../lib/highlight";
import { parseSkillTurn } from "../lib/skillInvocation";
import { SkillTurn } from "./MessageExtras";
import { invocableSkills } from "./SlashMenu";
import { TaskCard, TodoList } from "./TaskCard";

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

const CATEGORY_ICON: Record<Category, () => JSX.Element> = {
  thinking: IconBrain, read: IconFile, search: IconSearch, edit: IconPencil, write: IconPencil,
  commands: IconTerminal, webSearch: IconGlobe, webFetch: IconGlobe, subagents: IconUsers,
  plan: IconListCheck, questions: IconUser, tools: IconBook,
};

function CategoryIcon({ cat }: { cat: Category }) {
  const I = CATEGORY_ICON[cat];
  return <I />;
}

function sessionCwd(): string {
  const st = getState();
  const cur = st.curSess ? st.sessions[st.curSess] : null;
  return (cur && cur.dir) || st.serverDir || "";
}

function isErrorResult(text: string): boolean {
  return text.startsWith("Error: ");
}

/** One-line description of a call: what it touched, not its JSON. */
function toolSummary(part: Extract<MsgPart, { kind: "tool" }>, S: ReturnType<typeof useStrings>): string {
  const args = parseToolArgs(part.argsStr);
  if (part.name === "task") return firstLine(String(args.description ?? ""));
  if (part.name === "write_todos") {
    const p = todoProgress(parseTodos(args));
    if (!p.total) return "";
    return S.chat.todoProgress(p.done, p.total) + (p.active ? " · " + p.active.content : "");
  }
  const display = formatToolDisplay(part.name, args, sessionCwd());
  if (part.name === "execute") return firstLine(display);
  return firstLine(layoutToolDisplay(part.name, display).header || display);
}

/**
 * A run of thinking and tool calls between two pieces of answer text
 * (deepseek-harness style). Folded by default — one quiet line saying what it
 * did ("Ran commands and read files") or, while running, what it is doing now.
 * Opens to one 24px row per step; each row opens in place, and a tool row
 * links to the same call on the Trace page.
 */
export function WorkGroup({
  items,
  isLast: _isLast,
  live = false,
  pendingToolCallId,
}: {
  items: MsgPart[];
  isLast: boolean;
  live?: boolean;
  pendingToolCallId?: string;
}) {
  const S = useStrings();
  const { ms, running: unfinished } = workSummary(items, Date.now(), live);
  const running = live && unfinished;
  const containsPending = !!pendingToolCallId && items.some(
    (p) => p.kind === "tool" && p.toolCallId === pendingToolCallId,
  );
  // Folded unless a call in here waits on the user's approval.
  const [open, setOpen] = useState(containsPending);
  const userToggled = useRef(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const [, tick] = useState(0);

  useEffect(() => {
    if (!userToggled.current && containsPending) setOpen(true);
  }, [containsPending]);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => tick((n) => n + 1), 250);
    return () => window.clearInterval(id);
  }, [running]);

  const errors = items.filter((p) => p.kind === "tool" && p.result != null && isErrorResult(String(p.result))).length;
  let title: string;
  let summary = "";
  let icon: JSX.Element;
  if (running) {
    const tool = runningTool(items);
    if (tool) {
      title = S.chat.actRunning[toolCategory(tool.name)];
      summary = toolSummary(tool, S);
    } else {
      const think = [...items].reverse().find((p) => p.kind === "think");
      title = thinkLabel(true, S);
      summary = think && think.kind === "think" ? lastLine(think.text) : "";
    }
    icon = <IconSpinner />;
  } else {
    title = doneTitle(items, S.chat);
    icon = <CategoryIcon cat={groupCategories(items)[0] || "thinking"} />;
  }

  return (
    <div ref={rootRef} className={"wg" + (running ? " running" : "") + (open ? " open" : "")}>
      <button
        type="button"
        className="wg-head"
        aria-expanded={open}
        onClick={() => {
          const willClose = open;
          userToggled.current = true;
          setOpen((v) => !v);
          if (willClose) requestAnimationFrame(() => rootRef.current?.scrollIntoView({ block: "nearest" }));
        }}
      >
        <span className="wg-icon">{icon}<span className="wg-chev"><IconChevronDown /></span></span>
        <span className={"wg-title" + (running ? " shimmer" : "")}>{title}</span>
        {summary && <><span className="wr-dot" /><span className="wg-sum">{summary}</span></>}
        {errors > 0 && <span className="wg-err">{S.chat.workFailed(errors)}</span>}
        {ms > 0 && <span className="wg-ms">{fmtDurationMs(ms)}</span>}
      </button>
      {open && (
        <div className="wg-body">
          {items.map((p, i) => {
            if (p.kind === "think") {
              return <ThinkRow key={i} part={p} streaming={p.ms == null && running} />;
            }
            if (p.kind === "tool") {
              return (
                <ToolRow
                  key={p.toolCallId || i}
                  part={p}
                  live={live}
                  awaitingApproval={!!pendingToolCallId && p.toolCallId === pendingToolCallId}
                />
              );
            }
            return null;
          })}
        </div>
      )}
    </div>
  );
}

function lastLine(text: string): string {
  const lines = text.split("\n").map((l) => l.replace(/\*\*|__|`/g, "").trim()).filter(Boolean);
  return lines.length ? lines[lines.length - 1] : "";
}

function ThinkRow({ part, streaming }: { part: Extract<MsgPart, { kind: "think" }>; streaming: boolean }) {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  const duration = part.ms != null
    ? part.ms
    : (streaming && part.t0 != null ? Date.now() - part.t0 : 0);
  const preview = streaming ? lastLine(part.text) : firstLine(part.text);
  return (
    <div className={"wr" + (streaming ? " running" : "") + (open ? " open" : "")}>
      <div className="wr-line">
        <button type="button" className="wr-head" aria-expanded={open} disabled={!part.text}
                onClick={() => setOpen((v) => !v)}>
          <span className="wr-icon">{streaming ? <IconSpinner /> : <IconBrain />}<span className="wr-chev"><IconChevronDown /></span></span>
          <span className="wr-title">{thinkLabel(streaming, S)}</span>
          {preview && <><span className="wr-dot" /><span className="wr-sum">{preview}</span></>}
          {duration > 0 && <span className="wr-ms">{fmtDurationMs(duration)}</span>}
        </button>
      </div>
      {open && part.text ? <div className="wr-body think-text">{part.text}</div> : null}
    </div>
  );
}

/** Local file-read tools: the call line is enough. */
const HIDE_RESULT_TOOLS = new Set(["read_file", "glob", "grep"]);

function ToolRow({
  part,
  live,
  awaitingApproval,
}: {
  part: Extract<MsgPart, { kind: "tool" }>;
  live: boolean;
  awaitingApproval: boolean;
}) {
  const S = useStrings();
  const nav = useNavigate();
  const running = live && part.result == null;
  const resultText = part.result != null ? String(part.result) : "";
  const isError = isErrorResult(resultText);
  const cat = toolCategory(part.name);
  const args = parseToolArgs(part.argsStr);
  const diff = part.diff || "";
  const stats = !isError ? diffStats(diff) : null;
  const [open, setOpen] = useState(awaitingApproval);
  const userToggled = useRef(false);
  useEffect(() => {
    if (!userToggled.current && awaitingApproval) setOpen(true);
  }, [awaitingApproval]);
  const duration = part.ms != null
    ? part.ms
    : (running && part.t0 != null ? Date.now() - part.t0 : 0);
  const sessionId = getState().curSess;
  const summary = isError ? firstLine(resultText.slice("Error: ".length)) : toolSummary(part, S);
  const title = S.chat.toolTitle[cat] || part.name;

  return (
    <div
      className={"wr wr-tool" + (running ? " running" : "") + (open ? " open" : "") + (isError ? " error" : "")
        + (awaitingApproval ? " awaiting-approval" : "")}
      data-tool-call-id={part.toolCallId || undefined}
    >
      <div className="wr-line">
        <button type="button" className="wr-head" aria-expanded={open}
                onClick={() => { userToggled.current = true; setOpen((v) => !v); }}>
          <span className="wr-icon">
            {running ? <IconSpinner /> : isError ? <IconAlert /> : <CategoryIcon cat={cat} />}
            <span className="wr-chev"><IconChevronDown /></span>
          </span>
          <span className="wr-title" title={part.name}>{title}</span>
          {summary && <><span className="wr-dot" /><span className="wr-sum" title={summary}>{summary}</span></>}
          {awaitingApproval && <span className="work-awaiting">{S.chat.approvalWaiting}</span>}
          {stats && (
            <span className="wr-diff"><span className="add">+{stats.added}</span> <span className="del">-{stats.removed}</span></span>
          )}
          {duration > 0 && <span className="wr-ms">{fmtDurationMs(duration)}</span>}
        </button>
        {sessionId && part.toolCallId && !running && (
          <button type="button" className="wr-trace" title={S.chat.viewInTraceTip}
                  onClick={() => nav(`/traces?sessionId=${encodeURIComponent(sessionId)}&call=${encodeURIComponent(part.toolCallId!)}`)}>
            <IconExternal />{S.chat.viewInTrace}
          </button>
        )}
      </div>
      {open && <ToolDetail part={part} args={args} running={running} resultText={resultText} isError={isError} duration={duration} />}
    </div>
  );
}

function ToolDetail({
  part, args, running, resultText, isError, duration,
}: {
  part: Extract<MsgPart, { kind: "tool" }>;
  args: Record<string, unknown>;
  running: boolean;
  resultText: string;
  isError: boolean;
  duration: number;
}) {
  if (part.name === "task") {
    const result = parseTaskResult(resultText);
    return (
      <div className="wr-body">
        <TaskCard brief={String(args.description ?? "")} subagentType={String(args.subagent_type ?? "")}
                  result={result} running={running} elapsedMs={duration} />
        {!result && resultText && !running && <pre className="step-pre out">{resultText}</pre>}
      </div>
    );
  }
  if (part.name === "write_todos" && !isError) {
    return <div className="wr-body"><TodoList todos={parseTodos(args)} /></div>;
  }
  const display = formatToolDisplay(part.name, args, sessionCwd());
  const layout = layoutToolDisplay(part.name, display);
  const diff = part.diff || "";
  const showDiff = Boolean(diff) && !isError;
  // The call itself: full command / every argument, since the row only had room for one line.
  const call = layout.body.trim() ? layout.body : (layout.header.length > 60 || part.name === "execute" ? display : "");
  const showResult = Boolean(resultText) && !showDiff && (isError || !HIDE_RESULT_TOOLS.has(part.name));
  return (
    <div className="wr-body">
      {call ? <pre className={layout.bodyKind === "call" || part.name === "execute" ? "step-tool-call" : "step-pre"}>{call}</pre> : null}
      {showDiff ? (
        <div className="work-tool-diff">
          <CodeFrame code={diff} language="diff" highlight={diff.length <= HIGHLIGHT_LIMIT} />
        </div>
      ) : null}
      {showResult ? <pre className={"step-pre out" + (isError ? " err" : "")}>{resultText}</pre> : null}
      {!call && !showDiff && !showResult && !running && <pre className="step-pre">{part.argsStr || "{}"}</pre>}
    </div>
  );
}

export function ApprovalCard({
  req,
  queueIndex,
  queueTotal,
  busy,
  onDecide,
}: {
  req: ApprovalRequest;
  queueIndex: number;
  queueTotal: number;
  busy: boolean;
  onDecide: (decision: ApprovalDecision) => void;
}) {
  const S = useStrings();
  const [similarOpen, setSimilarOpen] = useState(false);
  const [denyOpen, setDenyOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const allowPrefix = req.options.includes("allow_prefix");
  const denyPrefix = req.options.includes("deny_prefix");

  useEffect(() => {
    if (!similarOpen && !denyOpen) return;
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) {
        setSimilarOpen(false);
        setDenyOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [similarOpen, denyOpen]);

  return (
    <div className="approval-card" ref={wrapRef} role="dialog" aria-label={req.question || req.name}>
      <div className="approval-card-head">
        <div className="approval-card-q">{req.question || S.chat.approvalWaiting}</div>
        {queueTotal > 1 && (
          <div className="approval-card-n">{S.chat.approvalQueue(queueIndex + 1, queueTotal)}</div>
        )}
      </div>
      {req.preview ? <pre className="approval-card-preview">{req.preview}</pre> : null}
      <div className="approval-card-acts">
        <div className={"approval-deny-split" + (denyPrefix ? "" : " solo")}>
          <button
            type="button"
            className="approval-deny"
            disabled={busy}
            onClick={() => onDecide("deny")}
          >
            {S.chat.approvalDeny} <kbd>Esc</kbd>
          </button>
          {denyPrefix && (
            <>
              <button
                type="button"
                className="approval-deny-more"
                disabled={busy}
                aria-label={S.chat.approvalDenySimilar(req.name, req.similarLabel)}
                aria-expanded={denyOpen}
                onClick={() => {
                  setDenyOpen((v) => !v);
                  setSimilarOpen(false);
                }}
              >
                <IconChevronDown />
              </button>
              {denyOpen && (
                <div className="approval-similar-dd left">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => {
                      setDenyOpen(false);
                      onDecide("deny_prefix");
                    }}
                  >
                    {S.chat.approvalDenySimilar(req.name, req.similarLabel)}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
        <div className={"approval-allow-split" + (allowPrefix ? "" : " solo")}>
          <button
            type="button"
            className="approval-allow"
            disabled={busy}
            onClick={() => onDecide("allow")}
          >
            {S.chat.approvalAllowOnce} <kbd>⏎</kbd>
          </button>
          {allowPrefix && (
            <>
              <button
                type="button"
                className="approval-allow-more"
                disabled={busy}
                aria-label={S.chat.approvalAllowSimilar(req.name, req.similarLabel)}
                aria-expanded={similarOpen}
                onClick={() => {
                  setSimilarOpen((v) => !v);
                  setDenyOpen(false);
                }}
              >
                <IconChevronDown />
              </button>
              {similarOpen && (
                <div className="approval-similar-dd">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => {
                      setSimilarOpen(false);
                      onDecide("allow_prefix");
                    }}
                  >
                    {S.chat.approvalAllowSimilar(req.name, req.similarLabel)}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function SteerChip({ text, ts, onOpenSkill }: { text: string; ts?: number; onOpenSkill: (name: string) => void }) {
  const S = useStrings();
  const skill = parseSkillTurn(text, invocableSkills(getState().pluginsData.skills || []));
  return (
    <div className="steer-chip">
      <div className="steer-chip-box">
        <div className="steer-chip-row">
          <IconUser />
          <p className="steer-chip-text">
            <span className="steer-chip-label">{S.chat.userSteering}</span>
            {skill ? <SkillTurn name={skill.name} args={skill.args} source={text} onOpen={onOpenSkill} /> : text}
          </p>
        </div>
      </div>
      {ts != null && (
        <span className="steer-chip-meta">{S.chat.msgStamp(ts)}</span>
      )}
    </div>
  );
}
