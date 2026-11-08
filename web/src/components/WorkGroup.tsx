import { lazy, Suspense, useEffect, useRef, useState, type JSX } from "react";
import { useNavigate } from "react-router";
import { fmtDurationMs } from "../lib/format";
import { workSummary, type MsgPart } from "../lib/msgParts";
import { formatToolDisplay, layoutToolDisplay, parseToolArgs } from "../lib/toolDisplay";
import {
  diffStats, doneTitle, firstLine, groupCategories, parseTaskResult, parseTodos, runningTool, todoProgress,
  toolCategory, toolResultPreview, type Category,
} from "../lib/workActivity";
import {
  IconAlert, IconBook, IconBrain, IconChevronDown, IconCopy, IconFile, IconGlobe, IconListCheck, IconPencil, IconSearch,
  IconExternal, IconSpinner, IconTerminal, IconUser, IconUsers,
} from "../icons";
import { getState, showToast, type ApprovalDecision, type ApprovalRequest } from "../store";
import { useStrings } from "../i18n";
import { CodeFrame } from "./CodeFrame";
import { HIGHLIGHT_LIMIT } from "../lib/highlight";
import { parseSkillTurn } from "../lib/skillInvocation";
import { SkillTurn } from "./MessageExtras";
import { invocableSkills } from "./SlashMenu";
import { TaskCard, TodoList } from "./TaskCard";

const ChatMarkdown = lazy(() => import("./ChatMarkdown").then((m) => ({ default: m.ChatMarkdown })));

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
  if (part.name === "execute" || part.name === "bash" || part.name === "shell") {
    const desc = firstLine(String(args.description ?? ""));
    if (desc) return desc;
  }
  const display = formatToolDisplay(part.name, args, sessionCwd());
  if (part.name === "execute") return display.replace(/\s+/g, " ").trim();
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
  defaultOpen = false,
}: {
  items: MsgPart[];
  isLast: boolean;
  live?: boolean;
  pendingToolCallId?: string;
  defaultOpen?: boolean;
}) {
  const S = useStrings();
  const { running: unfinished } = workSummary(items, Date.now(), live);
  const running = live && unfinished;
  const containsPending = !!pendingToolCallId && items.some(
    (p) => p.kind === "tool" && p.toolCallId === pendingToolCallId,
  );
  // Folded unless a call in here waits on the user's approval.
  const [open, setOpen] = useState(defaultOpen || containsPending);
  const userToggled = useRef(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const [, tick] = useState(0);

  useEffect(() => {
    if (!userToggled.current && containsPending) setOpen(true);
  }, [containsPending]);

  useEffect(() => {
    if (!userToggled.current && defaultOpen) setOpen(true);
  }, [defaultOpen]);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => tick((n) => n + 1), 250);
    return () => window.clearInterval(id);
  }, [running]);

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
        {!open && summary && <><span className="wr-dot" /><span className="wg-sum">{summary}</span></>}
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
  const preview = streaming ? lastLine(part.text) : firstLine(part.text);
  return (
    <div className={"wr" + (streaming ? " running" : "") + (open ? " open" : "")}>
      <div className="wr-line">
        <button type="button" className="wr-head" aria-expanded={open} disabled={!part.text}
                onClick={() => setOpen((v) => !v)}>
          <span className="wr-icon">{streaming ? <IconSpinner /> : <IconBrain />}<span className="wr-chev"><IconChevronDown /></span></span>
          <span className="wr-title">{thinkLabel(streaming, S)}</span>
          {!open && preview && (
            <>
              <span className="wr-dot" />
              <span className={"wr-sum" + (streaming ? " streaming" : "")}>{preview}</span>
            </>
          )}
        </button>
      </div>
      {open && part.text ? (
        <div className="wr-body think-text">
          <Suspense fallback={<span className="markdown-fallback">{part.text}</span>}>
            <ChatMarkdown text={part.text} streaming={streaming} />
          </Suspense>
        </div>
      ) : null}
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
  const isShell = part.name === "execute" || part.name === "bash" || part.name === "shell";
  const hasExplicitDesc = isShell && Boolean(firstLine(String(args.description ?? "")));
  const summary = isError ? firstLine(resultText.slice("Error: ".length)) : toolSummary(part, S);
  const showSummary = Boolean(summary) && (!open || !isShell || hasExplicitDesc);
  const resultPreview = !open && !running && !isError ? toolResultPreview(part.name, resultText, Boolean(stats)) : "";
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
          {showSummary && (
            <>
              <span className="wr-dot" />
              {resultPreview ? (
                <span className="wr-sum has-result" title={`${summary} → ${resultPreview}`}>
                  <span className="wr-call-req">{summary}</span>
                  <span className="wr-call-res">
                    <span className="wr-call-arr">{" → "}</span>
                    <span className="wr-call-res-text">{resultPreview}</span>
                  </span>
                </span>
              ) : (
                <span className="wr-sum" title={summary}>{summary}</span>
              )}
            </>
          )}
          {awaitingApproval && <span className="work-awaiting">{S.chat.approvalWaiting}</span>}
          {stats && (
            <span className="wr-diff"><span className="add">+{stats.added}</span> <span className="del">-{stats.removed}</span></span>
          )}
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
  if (part.name === "execute" || part.name === "bash" || part.name === "shell") {
    return (
      <div className="wr-body">
        <TerminalCard
          command={String(args.command ?? "")}
          workDir={String(args.work_dir ?? "") || sessionCwd()}
          resultText={resultText}
          running={running}
          isError={isError}
        />
      </div>
    );
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

function promptCwdLabel(cwd: string): string {
  const trimmed = cwd.replace(/[/\\]+$/, "");
  const seg = trimmed.split(/[/\\]/).pop();
  return seg || "$";
}

function parseTerminalOutput(raw: string, isError: boolean): { output: string; exitCode: number | null } {
  const text = isError && raw.startsWith("Error: ") ? raw.slice("Error: ".length) : raw;
  const m = /\n*\[Exit code:\s*(-?\d+)\]\s*$/i.exec(text);
  if (m) {
    return { output: text.slice(0, m.index).replace(/\n$/, ""), exitCode: Number(m[1]) };
  }
  return { output: text.replace(/\n$/, ""), exitCode: isError ? 1 : 0 };
}

/** deepseek-harness `TerminalBlock` style card for `execute` / `bash` tool calls. */
function TerminalCard({
  command, workDir, resultText, running, isError,
}: {
  command: string;
  workDir: string;
  resultText: string;
  running: boolean;
  isError: boolean;
}) {
  const S = useStrings();
  const [copied, setCopied] = useState(false);
  const { output, exitCode } = parseTerminalOutput(resultText, isError);
  const failed = !running && (isError || (exitCode !== null && exitCode !== 0));
  const stateLabel = S.chat.taskStatus(running ? "running" : failed ? "error" : "completed");
  const cwdLabel = promptCwdLabel(workDir);
  const cmdBody = command.endsWith("\n") ? command.slice(0, -1) : command;
  const cmdLines = (cmdBody || "").split("\n");
  const empty = !output.trim();
  const showBody = !running || !empty;
  const copyPayload = !empty ? output : command;
  const onCopy = () => {
    if (!copyPayload) return;
    void navigator.clipboard.writeText(copyPayload).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className={"term-card" + (running ? " running" : "") + (failed ? " error" : "") + (showBody ? " has-body" : "")}>
      <div className="term-card-head">
        <div className="term-prompt">
          <span className="term-sr-state">{stateLabel}</span>
          {cmdLines.map((line, i) => (
            <div key={i} className="term-prompt-line">
              {i === 0 && (
                <span className={"term-state-dot " + (running ? "running" : failed ? "error" : "done")} aria-hidden />
              )}
              <span className="term-cwd">{i === 0 ? cwdLabel : "$"}</span>
              <span className="term-cmd">{line}</span>
            </div>
          ))}
        </div>
        {!running && exitCode !== null && exitCode !== 0 && (
          <span className="term-exit">exit {exitCode}</span>
        )}
        {copyPayload && (
          <button type="button" className="term-copy" onClick={onCopy}>
            {copied ? S.common.copied : S.common.copy}
          </button>
        )}
      </div>
      {showBody && (
        empty ? (
          <div className="term-empty">(no output)</div>
        ) : (
          <pre className={"term-output" + (failed ? " err" : "")}>{output}</pre>
        )
      )}
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
  const copy = () => {
    if (!text) return;
    void navigator.clipboard.writeText(text).then(() => showToast(S.common.copied));
  };
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
      <div className="steer-chip-footer">
        {text && (
          <button type="button" className="msg-act" title={S.common.copy} aria-label={S.common.copy} onClick={copy}>
            <IconCopy />
          </button>
        )}
        {ts != null && (
          <span className="steer-chip-meta">{S.chat.msgStamp(ts)}</span>
        )}
      </div>
    </div>
  );
}
