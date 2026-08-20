import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import * as api from "../api";
import { agoStr, fmtFileSize, fmtN, shortenPath } from "../lib/format";
import { IconChat, IconClose, IconCopy, IconSearch, Logo } from "../icons";
import { showToast } from "../store";

type Entry = {
  index: number;
  ts: string;
  kind: string;
  summary: string;
  detail: string;
  toolName?: string;
  toolCallId?: string;
  isError?: boolean;
  durationMs?: number;
};

type Tokens = { input: number; cacheRead: number; cacheWrite: number; output: number };

type Round = {
  taskIndex: number;
  title: string;
  compaction: boolean;
  startTs: string;
  endTs: string;
  durationMs: number;
  llmMs: number;
  requests: number;
  waitMs: number;
  toolCalls: number;
  toolResults: number;
  toolErrors: number;
  tokens: Tokens;
  costUsd: number | null;
  tps: number;
  phases: Record<string, number>;
  entries: Entry[];
};

type Analysis = {
  session_id: string;
  hasTimeline: boolean;
  elapsedMs: number;
  compactionCount: number;
  reconnectCount: number;
  meta: {
    model: string | null;
    provider: string | null;
    contextWindow: number | null;
    cwd: string | null;
    gitBranch: string | null;
    version: string | null;
    tools: string[];
    systemPromptChars: number;
  };
  totals: {
    rounds: number;
    requests: number;
    toolCalls: number;
    toolOk: number;
    toolErrors: number;
    compactions: number;
    tokens: Tokens;
    elapsedMs: number;
    llmMs: number;
    waitMs: number;
    costUsd: number | null;
    tps: number;
  };
  rounds: Round[];
  modelSegments: Array<{ kind: string; startTs: string; endTs: string; taskIndex: number; key: string; name?: string }>;
  toolSpans: Array<{ toolCallId: string; name: string; callTs: string; approvalTs?: string; outputTs?: string; taskIndex: number }>;
  file: { path: string; sizeBytes: number; modifiedAt: string; name: string };
};

const PHASES: Array<{ key: string; label: string; cls: string }> = [
  { key: "thinking", label: "思考", cls: "ph-thinking" },
  { key: "text", label: "模型回复", cls: "ph-text" },
  { key: "toolArgs", label: "工具参数生成", cls: "ph-args" },
  { key: "toolWait", label: "审批等待", cls: "ph-wait" },
  { key: "toolExec", label: "工具执行", cls: "ph-exec" },
  { key: "other", label: "其它", cls: "ph-other" },
];

const KIND_LABEL: Record<string, string> = {
  session_meta: "session_meta",
  tool_list_ready: "tool_list_ready",
  system_prompt: "system_prompt",
  user: "user",
  steering: "steering",
  request_begin: "request_begin",
  request_end: "request_end",
  thinking: "thinking",
  text: "text",
  tool_call: "tool_call",
  tool_result: "tool_call_output",
  approval_decision: "approval_decision",
  token_usage: "token_usage",
  assistant: "assistant",
  compact_boundary: "compact_boundary",
  goal: "goal",
};

function ms(v: number) {
  if (v <= 0) return "0ms";
  if (v < 1000) return `${v}ms`;
  return `${(v / 1000).toFixed(v < 10000 ? 2 : 1)}s`;
}

function money(v: number | null) {
  if (v === null || v === undefined) return "—";
  return v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
}

export function TracesPage() {
  const [params, setParams] = useSearchParams();
  const sessionId = params.get("sessionId") || "";
  const [sessions, setSessions] = useState<any[]>([]);
  const [search, setSearch] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void api.fetchSessions().then(({ data }) => setSessions(data?.sessions || []));
  }, []);

  useEffect(() => {
    if (!sessionId) { setAnalysis(null); setError(""); return; }
    setLoading(true);
    void api.fetchTraceAnalysis(sessionId).then(({ ok, status, data }) => {
      setLoading(false);
      if (ok) { setAnalysis(data as Analysis); setError(""); return; }
      setAnalysis(null);
      setError(status === 404
        ? "这个会话还没有写入磁盘轨迹（新建但未发消息，或已被删除）。"
        : `读取轨迹失败（HTTP ${status}）`);
    });
  }, [sessionId]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter((s) =>
      (s.name || "").toLowerCase().includes(q) || (s.session_id || "").toLowerCase().includes(q));
  }, [sessions, search]);

  return (
    <div className="trace-page">
      <aside className="trace-rail">
        <div className="side-head">
          <div className="brand"><Logo /><span>轨迹观测</span></div>
        </div>
        <label className="trace-search">
          <IconSearch />
          <input placeholder="搜索会话" value={search} onChange={(e) => setSearch(e.target.value)} />
          {search && <button className="search-clear" onClick={() => setSearch("")}><IconClose /></button>}
        </label>
        <div className="trace-rail-label">会话 {filtered.length ? `(${filtered.length})` : ""}</div>
        <div className="trace-rail-list">
          {!filtered.length && <div className="s-empty">没有会话</div>}
          {filtered.map((s) => (
            <button
              key={s.session_id}
              className={"trace-rail-item" + (s.session_id === sessionId ? " active" : "")}
              onClick={() => setParams({ sessionId: s.session_id })}
            >
              <span className="tri-title">{s.name || s.session_id}</span>
              <span className="tri-meta">
                <span>{s.user_count || 0} 轮</span>
                <span>{fmtFileSize(s.size_bytes || 0)}</span>
              </span>
            </button>
          ))}
        </div>
        <Link className="trace-rail-back" to="/chat"><IconChat /><span>返回对话</span></Link>
      </aside>

      <div className="trace-main">
        {!sessionId && (
          <div className="trace-empty">
            <h2>选择左侧会话查看执行轨迹</h2>
            <p>同一工作目录下的 CLI 会话也会出现在这里，Web 与 CLI 共用同一份 session 日志。</p>
          </div>
        )}
        {sessionId && loading && <div className="trace-empty"><p>加载中…</p></div>}
        {sessionId && !loading && error && <div className="trace-empty"><p>{error}</p></div>}
        {analysis && <TraceDetail a={analysis} />}
      </div>
    </div>
  );
}

function TraceDetail({ a }: { a: Analysis }) {
  const t = a.totals;
  const tokens = t.tokens;
  return (
    <>
      <div className="trace-head">
        <div className="trace-head-l">
          <h1>{a.file.name || a.session_id}</h1>
          <div className="trace-head-sub">
            <span>{new Date(a.file.modifiedAt).toLocaleString()}</span>
            <span>{fmtFileSize(a.file.sizeBytes)}</span>
            {a.meta.model && <span>{a.meta.model}</span>}
            {a.meta.gitBranch && <span>git:{a.meta.gitBranch}</span>}
            {a.meta.cwd && <span title={a.meta.cwd}>{shortenPath(a.meta.cwd)}</span>}
          </div>
        </div>
        <div className="trace-head-r">
          <span className="trace-badge">{a.session_id}</span>
          <button className="dp-btn" onClick={() => { void navigator.clipboard.writeText(a.file.path); showToast("已复制轨迹文件路径"); }}>
            <IconCopy /> 复制路径
          </button>
        </div>
      </div>

      <section className="trace-card">
        <div className="trace-card-title">全局统计</div>
        <div className="stat-grid">
          <Stat label="轮次" value={String(t.rounds)} />
          <Stat label="输入 tokens" value={fmtN(tokens.input)} hint={tokens.cacheRead ? `缓存命中 ${fmtN(tokens.cacheRead)}` : undefined} />
          <Stat label="成本" value={money(t.costUsd)} hint={a.meta.model ? undefined : "缺少模型信息"} />
          <Stat label="工具调用" value={String(t.toolCalls)} hint={t.toolErrors ? `失败 ${t.toolErrors}` : `成功 ${t.toolOk}`} />
          <Stat label="等待时间" value={ms(t.waitMs)} />
          <Stat label="耗时" value={ms(t.elapsedMs)} hint={`模型 ${ms(t.llmMs)}`} />
          <Stat label="响应次数" value={String(t.requests)} hint={a.reconnectCount ? `重试 ${a.reconnectCount}` : undefined} />
          <Stat label="输出 tokens" value={fmtN(tokens.output)} />
          <Stat label="输出 TPS" value={`${t.tps.toFixed(1)} tok/s`} />
        </div>
        {!a.hasTimeline && (
          <p className="trace-note">
            这个会话记录于轨迹事件上线之前，只能列出消息，不做时间线推测。
          </p>
        )}
        {t.compactions > 0 && (
          <p className="trace-note">上下文被压缩 {t.compactions} 次，压缩轮次单独成卡。</p>
        )}
      </section>

      {a.rounds.map((r, i) => (
        <RoundCard key={r.taskIndex} round={r} ordinal={i + 1} analysis={a} />
      ))}
    </>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-hint">{hint || ""}</div>
    </div>
  );
}

function RoundCard({ round, ordinal, analysis }: { round: Round; ordinal: number; analysis: Analysis }) {
  const [open, setOpen] = useState(ordinal === 1);
  const label = round.compaction ? "上下文压缩" : `第 ${ordinal} 轮`;
  return (
    <section className={"trace-card round" + (round.compaction ? " compaction" : "")}>
      <button className="round-head" onClick={() => setOpen(!open)}>
        <span className={"round-caret" + (open ? " open" : "")}>▸</span>
        <span className="round-title">{label}</span>
        <span className="round-subject" title={round.title}>{round.title}</span>
        <span className="round-stats">
          <span title="请求次数">⇅ {round.requests}</span>
          <span title="工具调用 / 失败">🔧 {round.toolCalls}{round.toolErrors ? ` (${round.toolErrors} 失败)` : ""}</span>
          <span title="输出 tokens">↑ {fmtN(round.tokens.output)}</span>
          <span title="成本">{money(round.costUsd)}</span>
          <span title="耗时">{ms(round.durationMs)}</span>
          <span title="输出 TPS">{round.tps.toFixed(1)} tok/s</span>
        </span>
      </button>
      {open && (
        <div className="round-body">
          <PhaseBar round={round} />
          <Lanes round={round} analysis={analysis} />
          <EntryList entries={round.entries} />
        </div>
      )}
    </section>
  );
}

function PhaseBar({ round }: { round: Round }) {
  const total = PHASES.reduce((sum, p) => sum + (round.phases[p.key] || 0), 0);
  if (total <= 0) return null;
  return (
    <div className="phase-block">
      <div className="phase-title">执行阶段</div>
      <div className="phase-bar">
        {PHASES.map((p) => {
          const v = round.phases[p.key] || 0;
          if (v <= 0) return null;
          return (
            <span
              key={p.key}
              className={"phase-seg " + p.cls}
              style={{ width: `${(v / total) * 100}%` }}
              title={`${p.label} ${ms(v)}`}
            />
          );
        })}
      </div>
      <div className="phase-legend">
        {PHASES.map((p) => {
          const v = round.phases[p.key] || 0;
          if (v <= 0) return null;
          return (
            <span key={p.key} className="phase-key">
              <i className={"phase-dot " + p.cls} />
              {p.label} <b>{ms(v)}</b>
            </span>
          );
        })}
      </div>
    </div>
  );
}

/** Serial model lane + one lane per tool call, positioned on the round's clock. */
function Lanes({ round, analysis }: { round: Round; analysis: Analysis }) {
  const start = Date.parse(round.startTs) || 0;
  const end = Date.parse(round.endTs) || start + 1;
  const span = Math.max(end - start, 1);
  const segs = analysis.modelSegments.filter((s) => s.taskIndex === round.taskIndex);
  const spans = analysis.toolSpans.filter((s) => s.taskIndex === round.taskIndex);
  if (!segs.length && !spans.length) return null;

  const pos = (from: string, to: string) => {
    const a = Date.parse(from) || start;
    const b = Date.parse(to || from) || a;
    return { left: `${((a - start) / span) * 100}%`, width: `${Math.max(((b - a) / span) * 100, 0.5)}%` };
  };
  const phaseFor = (kind: string) =>
    kind === "thinking" ? "ph-thinking" : kind === "text" ? "ph-text" : "ph-args";

  return (
    <div className="lane-block">
      <div className="lane-row">
        <span className="lane-name">模型</span>
        <div className="lane">
          {segs.map((s) => (
            <span key={s.key} className={"lane-seg " + phaseFor(s.kind)} style={pos(s.startTs, s.endTs)}
                  title={`${s.kind}${s.name ? " · " + s.name : ""} ${ms(Date.parse(s.endTs) - Date.parse(s.startTs))}`} />
          ))}
        </div>
      </div>
      {spans.map((sp) => (
        <div className="lane-row" key={sp.toolCallId}>
          <span className="lane-name" title={sp.name}>{sp.name || "tool"}</span>
          <div className="lane">
            {sp.approvalTs && (
              <span className="lane-seg ph-wait" style={pos(sp.callTs, sp.approvalTs)} title={`审批等待 ${ms(Date.parse(sp.approvalTs) - Date.parse(sp.callTs))}`} />
            )}
            {sp.outputTs && (
              <span className="lane-seg ph-exec" style={pos(sp.approvalTs || sp.callTs, sp.outputTs)}
                    title={`执行 ${ms(Date.parse(sp.outputTs) - Date.parse(sp.approvalTs || sp.callTs))}`} />
            )}
            {!sp.outputTs && <span className="lane-seg ph-exec pending" style={pos(sp.callTs, sp.callTs)} title="没有结果（被中断或仍在运行）" />}
          </div>
        </div>
      ))}
      <div className="lane-axis"><span>0ms</span><span>{ms(span)}</span></div>
    </div>
  );
}

function EntryList({ entries }: { entries: Entry[] }) {
  const [openAll, setOpenAll] = useState(false);
  return (
    <div className="entry-block">
      <div className="entry-head">
        <span>摘要 ({entries.length})</span>
        <button className="entry-expand" onClick={() => setOpenAll(!openAll)}>
          {openAll ? "全部收起" : "全部展开"}
        </button>
      </div>
      <div className="entry-list">
        {entries.map((e) => <EntryRow key={`${e.index}-${e.kind}`} e={e} forceOpen={openAll} />)}
      </div>
    </div>
  );
}

function EntryRow({ e, forceOpen }: { e: Entry; forceOpen: boolean }) {
  const [open, setOpen] = useState(false);
  const expanded = forceOpen || open;
  const expandable = !!e.detail;
  return (
    <div className={"entry" + (e.isError ? " is-error" : "")}>
      <button
        className={"entry-row" + (expandable ? " expandable" : "")}
        onClick={() => expandable && setOpen(!open)}
      >
        <span className="entry-ts">{e.ts.slice(11, 19)}</span>
        <span className="entry-caret">{expandable ? (expanded ? "▾" : "▸") : ""}</span>
        <span className={"entry-kind k-" + e.kind}>{KIND_LABEL[e.kind] || e.kind}</span>
        <span className="entry-summary">{e.summary}</span>
        {e.durationMs !== undefined && <span className="entry-dur">{ms(e.durationMs)}</span>}
      </button>
      {expanded && expandable && (
        <div className="entry-detail">
          <button className="entry-copy" onClick={() => { void navigator.clipboard.writeText(e.detail); showToast("已复制"); }}>
            <IconCopy />
          </button>
          <pre>{e.detail}</pre>
        </div>
      )}
    </div>
  );
}
