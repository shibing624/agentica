import { lazy, Suspense, useState } from "react";
import { IconChevronDown, IconListCheck } from "../icons";
import { useStrings } from "../i18n";
import { fmtDurationMs } from "../lib/format";
import type { TaskResult, Todo } from "../lib/workActivity";

const ChatMarkdown = lazy(() => import("./ChatMarkdown").then((m) => ({ default: m.ChatMarkdown })));

/** A subagent run (`task` tool): status line, brief, the steps it took, its report.
 *  Chat rows and Trace entries render the same card. */
export function TaskCard({
  brief, subagentType, result, running = false, elapsedMs,
}: {
  brief: string;
  subagentType?: string;
  result: TaskResult | null;
  running?: boolean;
  elapsedMs?: number;
}) {
  const S = useStrings();
  const status = running ? "running" : result ? result.status : "";
  const tone = running ? "run" : !result ? "" : result.success ? "ok" : result.status === "error" ? "err" : "warn";
  const ms = result?.executionMs ?? elapsedMs;
  return (
    <div className="task-card">
      <div className="task-meta">
        {status && <span className={"task-status " + tone}>{S.chat.taskStatus(status)}</span>}
        {(result?.subagentName || subagentType) && (
          <span className="task-agent">{result?.subagentName || subagentType}</span>
        )}
        {ms != null && ms > 0 && <span className="task-num">{fmtDurationMs(ms)}</span>}
        {!!result?.toolCount && <span className="task-num">{S.chat.taskMeta(result.toolCount)}</span>}
        {result?.runId && <span className="task-run" title={S.chat.taskRunId}>{result.runId}</span>}
      </div>
      {brief && (
        <details className="task-sec" open={!result && !running}>
          <summary>{S.chat.taskBrief}</summary>
          <div className="task-brief">{brief}</div>
        </details>
      )}
      {!!result?.steps.length && (
        <details className="task-sec">
          <summary>{S.chat.taskSteps}<span className="task-sec-n">{result.steps.length + (result.stepsTruncated || 0)}</span></summary>
          <ol className="task-steps">
            {result.steps.map((st, i) => (
              <li key={i}><span className="task-step-name">{st.name}</span><span className="task-step-info">{st.info}</span></li>
            ))}
            {!!result.stepsTruncated && <li className="task-steps-more">{S.chat.taskStepsMore(result.stepsTruncated)}</li>}
          </ol>
        </details>
      )}
      {result && !result.success && result.error && <div className="task-error">{result.error}</div>}
      {result?.result && (
        <div className="task-report">
          <div className="task-report-h">{S.chat.taskReport}</div>
          <Suspense fallback={<pre className="markdown-fallback">{result.result}</pre>}>
            <ChatMarkdown text={result.result} streaming={false} />
          </Suspense>
        </div>
      )}
    </div>
  );
}

export function TodoList({ todos }: { todos: Todo[] }) {
  return (
    <ul className="todo-list">
      {todos.map((t, i) => (
        <li key={i} className={"todo-item " + t.status}>
          <span className={"state-dot " + t.status} aria-hidden />
          <span className="todo-text">{t.content}</span>
        </li>
      ))}
    </ul>
  );
}

/**
 * The plan above the composer: the newest `write_todos` list of this session,
 * folded to one line ("Tasks  2 done · 1 in progress · 3 pending — <current>").
 * Hidden once every item is done, so a finished plan does not sit there.
 */
export function TodoDock({ todos }: { todos: Todo[] }) {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  if (!todos.length || todos.every((t) => t.status === "completed")) return null;
  const done = todos.filter((t) => t.status === "completed").length;
  const active = todos.filter((t) => t.status === "in_progress");
  const pending = todos.length - done - active.length;
  return (
    <div className={"todo-dock" + (open ? " open" : "")}>
      <button type="button" className="todo-dock-head" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <span className="todo-dock-icon"><IconListCheck /></span>
        <span className="todo-dock-title">{S.chat.todoTitle}</span>
        <span className="todo-dock-counts">{S.chat.todoCounts(done, active.length, pending)}</span>
        {!open && active[0] && <span className="todo-dock-cur">{active[0].content}</span>}
        <span className="todo-dock-chev"><IconChevronDown /></span>
      </button>
      {open && <div className="todo-dock-body"><TodoList todos={todos} /></div>}
    </div>
  );
}
