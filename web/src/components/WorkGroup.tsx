import { lazy, Suspense, useEffect, useRef, useState, type JSX } from "react";
import { useNavigate } from "react-router";
import { fmtDurationMs } from "../lib/format";
import { jobDuration, parseBackgroundStart } from "../lib/backgroundJobs";
import { normalizeMentionedPath } from "../lib/filePath";
import { workSummary, type MsgPart } from "../lib/msgParts";
import { formatToolDisplay, layoutToolDisplay, parseToolArgs } from "../lib/toolDisplay";
import {
  diffStats, doneTitle, firstLine, groupCategories, parseShellOutput, parseTaskResult, parseTodos, runningTool,
  todoProgress, toolCategory, toolResultPreview, type Category,
} from "../lib/workActivity";
import {
  IconBook, IconBrain, IconChat, IconChevronDown, IconCopy, IconFile, IconGlobe, IconListCheck, IconPencil, IconSearch,
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
  idle: string = S.chat.thinkingNow,
): string {
  if (!streaming) return S.chat.thinking;
  const st = getState();
  const live = st.streams[st.curSess || ""];
  if (live?.cancelling) return S.chat.stopping;
  if (live?.reconnecting) return S.chat.reconnecting;
  if (live?.preparing) return S.chat.preparingAttachments;
  return idle;
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
 * (deepseek-harness `detailed` style). While the turn is streaming (`live`),
 * step rows render flat without a group header so every command description
 * and file read is visible in real time; once settled, historical groups keep
 * their collapsible summary header ("Ran commands and read files").
 */
export function WorkGroup({
  items,
  isLast,
  live = false,
  pendingToolCallId,
  onOpenFile,
}: {
  items: MsgPart[];
  isLast: boolean;
  live?: boolean;
  pendingToolCallId?: string;
  onOpenFile?: (path: string) => void;
}) {
  const S = useStrings();
  const { running: unfinished } = workSummary(items, Date.now(), live);
  // The tail group of a live turn stays "running" between tool batches too:
  // the turn is waiting on the next LLM step, which reads as analysing.
  const running = live && (unfinished || isLast);
  const containsPending = !!pendingToolCallId && items.some(
    (p) => p.kind === "tool" && p.toolCallId === pendingToolCallId,
  );
  // Second level of the process (dsh): folded to one line until clicked,
  // except while one of its tools waits on approval.
  const [open, setOpen] = useState(containsPending);
  const userToggled = useRef(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const [, tick] = useState(0);

  useEffect(() => {
    if (!userToggled.current && containsPending) setOpen(true);
  }, [containsPending]);

  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => tick((n) => n + 1), 250);
    return () => window.clearInterval(id);
  }, [live]);

  const rows = items.map((p, i) => {
    if (p.kind === "think") {
      const streamingThink = p.ms == null && running;
      if (!streamingThink && !p.text.trim()) return null;
      return <ThinkRow key={i} part={p} streaming={streamingThink} />;
    }
    if (p.kind === "text") {
      if (!p.text.trim()) return null;
      return <ReplyRow key={i} text={p.text} />;
    }
    if (p.kind === "tool") {
      return (
        <ToolRow
          key={p.toolCallId || i}
          part={p}
          live={live}
          awaitingApproval={!!pendingToolCallId && p.toolCallId === pendingToolCallId}
          onOpenFile={onOpenFile}
        />
      );
    }
    return null;
  });

  // dsh standard mode: a live group is one folded line — the current step's
  // category plus its detail (tool argument or latest thought) — and opens
  // to the full rows on click.
  let title: string;
  let summary = "";
  let icon: JSX.Element;
  if (running) {
    const tool = unfinished ? runningTool(items) : undefined;
    if (tool) {
      const cat = toolCategory(tool.name);
      title = S.chat.actRunning[cat];
      summary = toolSummary(tool, S);
      icon = <CategoryIcon cat={cat} />;
    } else {
      const last = items[items.length - 1];
      title = thinkLabel(true, S, S.chat.analyzingNow);
      summary = last?.kind === "think" ? latestCompletedParagraphFirstLine(last.text) : "";
      icon = <IconBrain />;
    }
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
          {rows}
        </div>
      )}
    </div>
  );
}

/** First line of the latest paragraph that is already closed by a blank line. */
function latestCompletedParagraphFirstLine(text: string): string {
  let summary = "";
  let paragraphStart = 0;
  const separator = /\r?\n(?:[\t ]*\r?\n)+/g;
  while (true) {
    const next = separator.exec(text);
    const paragraphEnd = next === null ? text.length : next.index + next[0].indexOf("\n");
    const newline = text.indexOf("\n", paragraphStart);
    if (newline !== -1 && newline <= paragraphEnd) {
      const candidate = text.slice(paragraphStart, newline).trim();
      if (candidate) summary = candidate.replaceAll("**", "");
    }
    if (next === null) return summary;
    paragraphStart = next.index + next[0].length;
  }
}

function ThinkRow({ part, streaming }: { part: Extract<MsgPart, { kind: "think" }>; streaming: boolean }) {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  // Streaming advances one completed paragraph at a time (dsh `ReasoningRow`),
  // so the preview never shows a half-written line; settled rows take line one.
  const preview = streaming ? latestCompletedParagraphFirstLine(part.text) : firstLine(part.text);
  return (
    <div className={"wr" + (streaming ? " running" : "") + (open ? " open" : "")}>
      <div className="wr-line">
        <button type="button" className="wr-head" aria-expanded={open} disabled={!part.text}
                onClick={() => setOpen((v) => !v)}>
          <span className="wr-icon">{streaming ? <IconSpinner /> : <IconBrain />}<span className="wr-chev"><IconChevronDown /></span></span>
          <span className={"wr-title" + (streaming ? " shimmer" : "")}>{thinkLabel(streaming, S)}</span>
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

/**
 * An assistant reply spoken between tool calls (models that narrate in
 * `content` instead of reasoning): one line inside the process group, opening
 * to the rendered text — the same shape as a thought.
 */
function ReplyRow({ text }: { text: string }) {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  const preview = firstLine(text);
  return (
    <div className={"wr" + (open ? " open" : "")}>
      <div className="wr-line">
        <button type="button" className="wr-head" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          <span className="wr-icon"><IconChat /><span className="wr-chev"><IconChevronDown /></span></span>
          <span className="wr-title">{S.chat.replyRow}</span>
          {!open && preview && (
            <>
              <span className="wr-dot" />
              <span className="wr-sum">{preview}</span>
            </>
          )}
        </button>
      </div>
      {open ? (
        <div className="wr-body think-text">
          <Suspense fallback={<span className="markdown-fallback">{text}</span>}>
            <ChatMarkdown text={text} streaming={false} />
          </Suspense>
        </div>
      ) : null}
    </div>
  );
}

/** Local file-read tools: the call line is enough. */
const HIDE_RESULT_TOOLS = new Set(["read_file", "glob", "grep"]);
const OPEN_FILE_TOOLS = new Set(["read_file", "write_file", "edit_file", "multi_edit"]);

function ToolRow({
  part,
  live,
  awaitingApproval,
  onOpenFile,
}: {
  part: Extract<MsgPart, { kind: "tool" }>;
  live: boolean;
  awaitingApproval: boolean;
  onOpenFile?: (path: string) => void;
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
  const cwd = sessionCwd();
  const isShell = part.name === "execute" || part.name === "bash" || part.name === "shell";
  const bgStart = isShell && !isError ? parseBackgroundStart(resultText) : null;
  const bgJob = bgStart && sessionId
    ? getState().bgJobsBySession[sessionId]?.find((j) => j.id === bgStart.id)
    : undefined;
  const bgRunning = bgJob?.status === "running";
  const shellParsed = isShell && !running && part.result != null ? parseShellOutput(resultText, isError) : null;
  const shellExitFailed = Boolean(
    (shellParsed && shellParsed.exitCode !== null && shellParsed.exitCode !== 0)
    || bgJob?.status === "failed",
  );
  const rowFailed = isError || shellExitFailed;
  const hasExplicitDesc = isShell && Boolean(firstLine(String(args.description ?? "")));
  // A failed call keeps its own description; the failure is the neutral badge
  // after it (dsh `ExitCodeBadge`), and the error text lives in the detail.
  const summary = toolSummary(part, S);
  const failBadge = bgJob || running
    ? ""
    : shellExitFailed && shellParsed?.exitCode != null
      ? S.chat.bgExitCode(shellParsed.exitCode)
      : isError ? S.chat.toolFailed : "";
  const cmdPreview = hasExplicitDesc ? formatToolDisplay("execute", args, cwd).replace(/\s+/g, " ").trim() : "";
  const summaryTip = hasExplicitDesc && cmdPreview ? `${summary}\n$ ${cmdPreview}` : summary;
  const showSummary = Boolean(summary) && (!open || !isShell || hasExplicitDesc);
  const resultPreview = !open && !running && !rowFailed
    ? (bgStart
      ? (bgRunning ? (bgJob?.progress || "") : "")
      : toolResultPreview(part.name, resultText, Boolean(stats)))
    : "";
  const bgBadge = bgJob
    ? (bgJob.status === "running"
      ? S.chat.bgInChatRunning(jobDuration(bgJob, Date.now()))
      : bgJob.status === "completed"
        ? S.chat.bgInChatDone(jobDuration(bgJob, Date.now()))
        : bgJob.status === "stopped"
          ? S.chat.bgInChatStopped
          : S.chat.bgInChatFailed(bgJob.exit_code))
    : "";
  const rawFilePath = OPEN_FILE_TOOLS.has(part.name) ? String(args.file_path ?? args.path ?? "").trim() : "";
  const openablePath = !running && !rowFailed && onOpenFile && rawFilePath
    ? normalizeMentionedPath(rawFilePath, cwd || null)
    : null;
  const title = S.chat.toolTitle[cat] || part.name;

  return (
    <div
      className={"wr wr-tool" + ((running || bgRunning) ? " running" : "") + (open ? " open" : "")
        + (awaitingApproval ? " awaiting-approval" : "")}
      data-tool-call-id={part.toolCallId || undefined}
    >
      <div className="wr-line">
        <button type="button" className="wr-head" aria-expanded={open}
                onClick={() => { userToggled.current = true; setOpen((v) => !v); }}>
          <span className="wr-icon">
            {(running || bgRunning) ? <IconSpinner /> : <CategoryIcon cat={cat} />}
            <span className="wr-chev"><IconChevronDown /></span>
          </span>
          <span className={"wr-title" + (running ? " shimmer" : "")} title={part.name}>{title}</span>
          {showSummary && (
            <>
              <span className="wr-dot" />
              {resultPreview ? (
                <span className="wr-sum has-result" title={`${summaryTip} → ${resultPreview}`}>
                  <span className="wr-call-req">{summary}</span>
                  <span className="wr-call-res">
                    <span className="wr-call-arr">{" → "}</span>
                    <span className="wr-call-res-text">{resultPreview}</span>
                  </span>
                </span>
              ) : openablePath && onOpenFile ? (
                <span
                  role="link"
                  tabIndex={0}
                  className="wr-sum wr-file-link"
                  title={summaryTip}
                  onClick={(e) => { e.stopPropagation(); onOpenFile(openablePath); }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      e.stopPropagation();
                      onOpenFile(openablePath);
                    }
                  }}
                >
                  {summary}
                </span>
              ) : (
                <span className={"wr-sum" + (running ? " shimmer" : "")} title={summaryTip}>{summary}</span>
              )}
            </>
          )}
          {bgBadge && bgJob && (
            <span className={"wr-bg-pill " + bgJob.status}>{bgBadge}</span>
          )}
          {failBadge && <span className="wr-exit"><span className="wr-exit-dot" aria-hidden />{failBadge}</span>}
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
      {open && (
        <ToolDetail
          part={part}
          args={args}
          running={running || bgRunning}
          resultText={resultText}
          isError={isError}
          duration={duration}
          bgLogPath={bgJob?.log_path || bgStart?.logPath}
          onOpenFile={onOpenFile}
        />
      )}
    </div>
  );
}

function ToolDetail({
  part, args, running, resultText, isError, duration, bgLogPath, onOpenFile,
}: {
  part: Extract<MsgPart, { kind: "tool" }>;
  args: Record<string, unknown>;
  running: boolean;
  resultText: string;
  isError: boolean;
  duration: number;
  bgLogPath?: string;
  onOpenFile?: (path: string) => void;
}) {
  if (part.name === "task") {
    const result = parseTaskResult(resultText);
    return (
      <div className="wr-body">
        <TaskCard brief={String(args.description ?? "")} subagentType={String(args.subagent_type ?? "")}
                  model={args.model ? String(args.model) : undefined}
                  modelTier={args.model_tier ? String(args.model_tier) : undefined}
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
          bgLogPath={bgLogPath}
          onOpenFile={onOpenFile}
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

/** deepseek-harness `TerminalBlock` style card for `execute` / `bash` tool calls. */
function TerminalCard({
  command, workDir, resultText, running, isError, bgLogPath, onOpenFile,
}: {
  command: string;
  workDir: string;
  resultText: string;
  running: boolean;
  isError: boolean;
  bgLogPath?: string;
  onOpenFile?: (path: string) => void;
}) {
  const S = useStrings();
  const [copied, setCopied] = useState(false);
  const { output, exitCode } = parseShellOutput(resultText, isError);
  const failed = !running && (isError || (exitCode !== null && exitCode !== 0));
  const stateLabel = S.chat.taskStatus(running ? "running" : failed ? "error" : "completed");
  const cwdLabel = promptCwdLabel(workDir);
  const cmdBody = command.endsWith("\n") ? command.slice(0, -1) : command;
  const cmdLines = (cmdBody || "").split("\n");
  const empty = !output.trim();
  const showBody = !running || !empty;
  const copyPayload = !empty ? output : command;
  const openableBgLog = bgLogPath && onOpenFile
    ? normalizeMentionedPath(bgLogPath, workDir || sessionCwd() || null)
    : null;
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
          <span className="term-exit"><span className="wr-exit-dot" aria-hidden />{S.chat.bgExitCode(exitCode)}</span>
        )}
        {openableBgLog && onOpenFile && (
          <button type="button" className="term-copy" title={S.chat.bgOpenLogTip} onClick={() => onOpenFile(openableBgLog)}>
            {S.chat.bgOpenLog}
          </button>
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
