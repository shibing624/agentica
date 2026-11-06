import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Navigate, useNavigate, useSearchParams } from "react-router";
import * as api from "../api";
import { useStrings, type Strings } from "../i18n";
import { fmtCost, fmtDurationMs, fmtFileSize, fmtN, fmtTps } from "../lib/format";
import { IconCopy, IconSearch } from "../icons";
import { showToast } from "../store";
import { switchTo } from "../sessions";
import { TaskCard } from "../components/TaskCard";
import type { TaskResult } from "../lib/workActivity";

type Tokens = {
  input: number;
  cacheRead: number;
  cacheWrite: number;
  output: number;
  prompt: number;
  cacheHitPercent: number | null;
};

/** One row of the call list (see `_round_calls` in agentica/memory/trace.py).
 *  The body is not in the page payload; it is fetched per call. */
type Call = {
  kind: string;
  index: number;
  ts: string;
  summary: string;
  startTs?: string;
  endTs?: string;
  durationMs?: number | null;
  waitMs?: number;
  status?: string;
  tokens?: Tokens | null;
  toolCalls?: number;
  toolName?: string;
  toolCallId?: string;
  isError?: boolean;
  decision?: string;
  resultSummary?: string;
};

type TaskFields = Partial<TaskResult> & { description?: string; subagentType?: string };

type Body = {
  args?: string;
  result?: string;
  approval?: string;
  thinking?: string;
  text?: string;
  taskCall?: TaskFields;
  taskResult?: TaskFields;
};

type Round = {
  taskIndex: number;
  title: string;
  /** Opened by an automatic report (background job, subagent, peer), not by the user. */
  automated?: boolean;
  compaction: boolean;
  startTs: string;
  durationMs: number;
  llmMs: number;
  requests: number;
  waitMs: number;
  toolCalls: number;
  toolErrors: number;
  tokens: Tokens;
  costUsd: number | null;
  tps: number;
  calls: Call[];
};

type Analysis = {
  session_id: string;
  hasTimeline: boolean;
  reconnectCount: number;
  meta: { model: string | null; provider: string | null; cwd: string | null; gitBranch: string | null };
  totals: {
    rounds: number;
    requests: number;
    toolCalls: number;
    toolErrors: number;
    tokens: Tokens;
    elapsedMs: number;
    llmMs: number;
    waitMs: number;
    costUsd: number | null;
    tps: number;
  };
  rounds: Round[];
  file: { path: string; sizeBytes: number; modifiedAt: string; name: string };
};

type Tab = "duration" | "rounds" | "calls";

function money(v: number | null | undefined) {
  return v === null || v === undefined ? "—" : fmtCost(v);
}

function roundLabel(r: Round, ordinal: number, S: Strings) {
  if (r.compaction) return S.traces.compaction;
  return r.automated ? `${S.traces.round(ordinal)} · ${S.traces.kind.automated}` : S.traces.round(ordinal);
}

/** Tools whose run is the user answering, not work being done. */
const HUMAN_TOOLS = new Set(["ask_user_question"]);

/** Wall clock of rounds split three ways: input (waiting on the user —
 *  approvals and questions), model (its requests, approval waits excluded),
 *  and tools (the rest: tools running between requests). */
function split(rounds: Round[]) {
  let input = 0;
  let model = 0;
  let wall = 0;
  for (const r of rounds) {
    wall += r.durationMs;
    model += r.llmMs;
    input += r.waitMs;
    for (const c of r.calls) {
      if (c.kind === "tool" && HUMAN_TOOLS.has(c.toolName || "")) input += c.durationMs || 0;
    }
  }
  const tool = Math.max(0, wall - model - input);
  return { input, model, tool, total: input + model + tool };
}

export function TracesPage() {
  const S = useStrings();
  const nav = useNavigate();
  const [params] = useSearchParams();
  const sessionId = params.get("sessionId") || "";
  // `call`: a tool call opened from a chat row; the Calls tab opens on it.
  const focusCall = params.get("call") || "";
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  // Kept as the status code, not a sentence, so a language switch re-renders it.
  const [errStatus, setErrStatus] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sessionId) { setAnalysis(null); setErrStatus(0); return; }
    setLoading(true);
    void api.fetchTraceAnalysis(sessionId).then(({ ok, status, data }) => {
      setLoading(false);
      if (ok) { setAnalysis(data as Analysis); setErrStatus(0); return; }
      setAnalysis(null);
      setErrStatus(status || 500);
    });
  }, [sessionId]);

  if (!sessionId) return <Navigate to="/chat" replace />;

  const backToChat = () => {
    switchTo(sessionId);
    nav("/chat");
  };

  return (
    <div className="trace-main">
      <button className="trace-back" onClick={backToChat}>{S.traces.backToChat}</button>
      {loading && <div className="trace-empty"><p>{S.common.loading}</p></div>}
      {!loading && !!errStatus && (
        <div className="trace-empty">
          <p>{errStatus === 404 ? S.traces.missing : S.traces.loadFailed(errStatus)}</p>
        </div>
      )}
      {analysis && <TraceDetail key={sessionId} a={analysis} focusCall={focusCall} />}
    </div>
  );
}

function TraceDetail({ a, focusCall }: { a: Analysis; focusCall: string }) {
  const S = useStrings();
  const t = a.totals;
  const focusInit = useMemo(() => {
    if (!focusCall) return null;
    for (const r of a.rounds) for (const c of r.calls) if (c.toolCallId === focusCall) return c.index;
    return null;
  }, [a, focusCall]);
  const [tab, setTab] = useState<Tab>(focusInit != null ? "calls" : "duration");
  // -1 = all rounds. Shared by the Duration charts and the Calls list.
  const [scope, setScope] = useState(-1);
  const [focus, setFocus] = useState<number | null>(focusInit);
  const [query, setQuery] = useState("");
  const hits = useTraceSearch(a.session_id, query);

  const openCall = (index: number) => { setScope(-1); setFocus(index); setTab("calls"); };
  const openRound = (i: number) => { setScope(i); setFocus(null); setTab("calls"); };

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
            {a.meta.cwd && <span className="trace-cwd"><code>{a.meta.cwd}</code></span>}
          </div>
        </div>
        <div className="trace-head-r">
          <span className="trace-badge" title={a.session_id}>{a.session_id}</span>
          <button className="trace-cwd-copy" title={S.traces.copyPath} aria-label={S.traces.copyPath}
                  onClick={() => { void navigator.clipboard.writeText(a.file.path); showToast(S.traces.copiedPath); }}>
            <IconCopy />
          </button>
        </div>
      </div>

      <div className="tr-stats">
        <Stat label={S.traces.statElapsed} value={fmtDurationMs(t.elapsedMs)} />
        <Stat label={S.traces.statRounds} value={String(t.rounds)} />
        <Stat label={S.traces.statRequests} value={String(t.requests)}
              hint={a.reconnectCount ? S.traces.retried(a.reconnectCount) : undefined} />
        <Stat label={S.traces.statToolCalls} value={String(t.toolCalls)}
              hint={t.toolErrors ? S.traces.failed(t.toolErrors) : undefined} />
        <Stat label={S.traces.statInput} value={fmtN(t.tokens.prompt)}
              hint={t.tokens.cacheHitPercent != null ? `cache ${t.tokens.cacheHitPercent.toFixed(1)}%` : undefined} />
        <Stat label={S.traces.statOutput} value={fmtN(t.tokens.output)} hint={fmtTps(t.tps)} />
        <Stat label={S.traces.statCost} value={money(t.costUsd)} />
      </div>

      <div className="tr-tabs" role="tablist">
        {(["duration", "rounds", "calls"] as Tab[]).map((k) => (
          <button key={k} type="button" role="tab" aria-selected={tab === k}
                  className={"tr-tab" + (tab === k ? " on" : "")} onClick={() => setTab(k)}>
            {k === "duration" ? S.traces.tabDuration : k === "rounds" ? S.traces.tabRounds : S.traces.tabCalls}
          </button>
        ))}
        <label className="tr-search">
          <IconSearch />
          <input type="search" value={query} placeholder={S.traces.search}
                 onChange={(e) => { setQuery(e.target.value); if (e.target.value.trim()) { setTab("calls"); setScope(-1); } }}
                 onKeyDown={(e) => { if (e.key === "Escape") setQuery(""); }} />
          {query.trim() && (
            <span className="tr-search-n">
              {hits.loading ? S.traces.searching : hits.indices.size ? S.traces.searchHits(hits.indices.size) : S.traces.searchNone}
            </span>
          )}
        </label>
      </div>

      {!a.hasTimeline && tab === "duration" && <p className="trace-note">{S.traces.noTimeline}</p>}
      {tab === "duration" && <DurationView a={a} scope={scope} setScope={setScope} onOpenCall={openCall} />}
      {tab === "rounds" && <RoundsView a={a} onOpenRound={openRound} />}
      {tab === "calls" && (
        <CallsView a={a} roundFilter={scope} setRoundFilter={setScope} focus={focus}
                   hits={query.trim() && !hits.loading ? hits.indices : null} />
      )}
    </>
  );
}

/** Server-side search over the whole log (bodies included), debounced. */
function useTraceSearch(sessionId: string, query: string) {
  const [state, setState] = useState<{ q: string; indices: Set<number>; loading: boolean }>(
    { q: "", indices: new Set(), loading: false },
  );
  useEffect(() => {
    const q = query.trim();
    if (!q) { setState({ q: "", indices: new Set(), loading: false }); return; }
    setState((s) => ({ ...s, loading: true }));
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => {
      void api.searchTraceCalls(sessionId, q, ctrl.signal).then(({ ok, data }) => {
        if (ctrl.signal.aborted) return;
        setState({ q, indices: new Set(ok && data ? data.indices : []), loading: false });
      });
    }, 250);
    return () => { window.clearTimeout(timer); ctrl.abort(); };
  }, [sessionId, query]);
  return state;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: ReactNode }) {
  return (
    <div className="tr-stat">
      <div className="tr-stat-label">{label}</div>
      <div className="tr-stat-value">{value}</div>
      {hint ? <div className="tr-stat-hint">{hint}</div> : null}
    </div>
  );
}

function SplitBar({ parts, scale = 1 }: { parts: ReturnType<typeof split>; scale?: number }) {
  const S = useStrings();
  const pct = (v: number) => (parts.total ? (v / parts.total) * 100 * scale : 0);
  return (
    <div className="tr-bar">
      {parts.input > 0 && <span className="tr-seg ph-input" style={{ width: pct(parts.input) + "%" }}
                                title={`${S.traces.phaseInput} ${fmtDurationMs(parts.input)}`} />}
      {parts.model > 0 && <span className="tr-seg ph-model" style={{ width: pct(parts.model) + "%" }}
                                title={`${S.traces.phaseModel} ${fmtDurationMs(parts.model)}`} />}
      {parts.tool > 0 && <span className="tr-seg ph-tool" style={{ width: pct(parts.tool) + "%" }}
                               title={`${S.traces.phaseTool} ${fmtDurationMs(parts.tool)}`} />}
    </div>
  );
}

const at = (ts?: string) => (ts ? Date.parse(ts) || 0 : 0);

type Span = { lane: 0 | 1 | 2; start: number; end: number; call: Call };

/** Every timed stretch of the scope, on three lanes: input (waiting on the
 *  user's approval), model (one request), tool (running after approval). */
function spansOf(rounds: Round[]): Span[] {
  const out: Span[] = [];
  for (const r of rounds) {
    for (const c of r.calls) {
      const s0 = at(c.startTs);
      const e0 = at(c.endTs);
      if (!s0 || !e0 || e0 < s0) continue;
      if (c.kind === "model") out.push({ lane: 1, start: s0, end: e0, call: c });
      else if (c.kind === "tool") {
        if (HUMAN_TOOLS.has(c.toolName || "")) {
          out.push({ lane: 0, start: s0, end: e0, call: c });
          continue;
        }
        const wait = c.waitMs || 0;
        if (wait > 0) out.push({ lane: 0, start: s0, end: Math.min(e0, s0 + wait), call: c });
        out.push({ lane: 2, start: Math.min(e0, s0 + wait), end: e0, call: c });
      }
    }
  }
  return out;
}

/** Cut the stretches where nothing ran (the user away between turns, a
 *  session resumed next day), so the axis is time spent, not time elapsed. */
function compress(spans: Span[], rounds: Round[]) {
  const sorted = [...spans].sort((x, y) => x.start - y.start || x.end - y.end);
  const shift = new Map<Span, number>();
  let removed = 0;
  let covered: number | null = null;
  const cuts: Array<{ at: number; removed: number }> = [];
  for (const sp of sorted) {
    if (covered !== null && sp.start > covered) {
      removed += sp.start - covered;
      cuts.push({ at: sp.start, removed });
    }
    shift.set(sp, removed);
    covered = covered === null ? sp.end : Math.max(covered, sp.end);
  }
  const origin = sorted.length ? sorted[0].start : 0;
  const proj = (time: number) => {
    let off = 0;
    for (const c of cuts) { if (c.at <= time) off = c.removed; else break; }
    return time - off - origin;
  };
  const placed = sorted.map((sp) => ({ ...sp, start: sp.start - shift.get(sp)! - origin, end: sp.end - shift.get(sp)! - origin }));
  const total = placed.reduce((m, sp) => Math.max(m, sp.end), 0);
  const marks = rounds
    .map((r) => r.calls.find((c) => c.kind === "model" && at(c.startTs)))
    .filter((c): c is Call => !!c)
    .map((c) => proj(at(c.startTs)))
    .filter((v) => v >= 0 && v <= total);
  return { spans: placed, total, marks };
}

const LANE_CLS = ["ph-input", "ph-model", "ph-tool"] as const;

function Timeline({ rounds, onOpenCall }: { rounds: Round[]; onOpenCall: (index: number) => void }) {
  const S = useStrings();
  const { spans, total, marks } = useMemo(() => compress(spansOf(rounds), rounds), [rounds]);
  if (!spans.length || total <= 0) return <p className="trace-note">{S.traces.noTiming}</p>;
  const lanes = [S.traces.phaseInput, S.traces.phaseModel, S.traces.phaseTool];
  const pos = (sp: { start: number; end: number }) => ({
    left: (sp.start / total) * 100 + "%",
    width: Math.max(((sp.end - sp.start) / total) * 100, 0.15) + "%",
  });
  const label = (sp: Span) => {
    const c = sp.call;
    const head = c.kind === "tool" ? c.toolName || "tool" : S.traces.kind.model;
    const what = sp.lane === 0 ? S.traces.phaseInput : "";
    return `${head}${what ? " · " + what : ""} · ${fmtDurationMs(sp.end - sp.start)}\n${c.kind === "tool" ? toolArgsSummary(c) : c.summary}`;
  };
  return (
    <div className="tr-tl">
      {lanes.map((name, lane) => (
        <div key={lane} className="tr-tl-row">
          <span className="tr-tl-name">{name}</span>
          <div className="tr-tl-lane">
            {marks.map((m, i) => <i key={i} className="tr-tl-mark" style={{ left: (m / total) * 100 + "%" }} />)}
            {spans.filter((sp) => sp.lane === lane).map((sp, i) => (
              <span key={i} className={"tr-tl-seg " + LANE_CLS[lane] + (sp.call.isError ? " err" : "")}
                    style={pos(sp)} title={label(sp)} onClick={() => onOpenCall(sp.call.index)} />
            ))}
          </div>
        </div>
      ))}
      <div className="tr-tl-row tr-tl-axis">
        <span className="tr-tl-name" />
        <div className="tr-tl-ticks">
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <span key={f} style={{ left: f * 100 + "%" }}>{fmtDurationMs(Math.round(f * total))}</span>
          ))}
        </div>
      </div>
      <p className="trace-note">{S.traces.timelineNote}</p>
    </div>
  );
}

function ToolTable({ rounds }: { rounds: Round[] }) {
  const S = useStrings();
  const rows = useMemo(() => {
    const by = new Map<string, { name: string; n: number; total: number; max: number; err: number }>();
    for (const r of rounds) for (const c of r.calls) {
      if (c.kind !== "tool") continue;
      const name = c.toolName || "tool";
      const row = by.get(name) || { name, n: 0, total: 0, max: 0, err: 0 };
      const ms = c.durationMs || 0;
      row.n++; row.total += ms; row.max = Math.max(row.max, ms); if (c.isError) row.err++;
      by.set(name, row);
    }
    return [...by.values()].sort((x, y) => y.total - x.total);
  }, [rounds]);
  if (!rows.length) return <p className="trace-note">{S.traces.noTiming}</p>;
  const top = rows[0].total || 1;
  return (
    <table className="tr-table">
      <thead><tr><th>{S.traces.phaseTool}</th><th>{S.traces.colCount}</th><th>{S.traces.colTotal}</th>
        <th>{S.traces.colAvg}</th><th>{S.traces.colMax}</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.name}>
            <td>
              <span className="tr-table-name">{r.name}</span>
              <span className="tr-table-bar"><i className="ph-tool" style={{ width: (r.total / top) * 100 + "%" }} /></span>
            </td>
            <td>{r.n}{r.err ? <span className="err"> ({r.err})</span> : null}</td>
            <td>{fmtDurationMs(r.total)}</td>
            <td>{fmtDurationMs(r.total / r.n)}</td>
            <td>{fmtDurationMs(r.max)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SlowestCalls({ rounds, onOpenCall }: { rounds: Round[]; onOpenCall: (index: number) => void }) {
  const S = useStrings();
  const calls = useMemo(() => rounds
    .flatMap((r) => r.calls)
    .filter((c) => (c.kind === "tool" || c.kind === "model") && (c.durationMs || 0) > 0)
    .sort((x, y) => (y.durationMs || 0) - (x.durationMs || 0))
    .slice(0, 10), [rounds]);
  if (!calls.length) return <p className="trace-note">{S.traces.noTiming}</p>;
  const top = calls[0].durationMs || 1;
  return (
    <div className="tr-slow">
      {calls.map((c) => (
        <button key={c.index} type="button" className="tr-slow-row" onClick={() => onOpenCall(c.index)}>
          <span className={"tr-tag " + (c.kind === "tool" ? "k-tool" : "k-model") + (c.isError ? " err" : "")}>
            {c.kind === "tool" ? c.toolName || "tool" : S.traces.kind.model}
          </span>
          <span className="tr-slow-sum">{c.kind === "tool" ? toolArgsSummary(c) : c.summary}</span>
          <span className="tr-slow-bar"><i className={c.kind === "tool" ? "ph-tool" : "ph-model"}
                                           style={{ width: ((c.durationMs || 0) / top) * 100 + "%" }} /></span>
          <span className="tr-slow-ms">{fmtDurationMs(c.durationMs || 0)}</span>
        </button>
      ))}
    </div>
  );
}

function DurationView({ a, scope, setScope, onOpenCall }: {
  a: Analysis; scope: number; setScope: (i: number) => void; onOpenCall: (index: number) => void;
}) {
  const S = useStrings();
  const scoped = useMemo(() => (scope >= 0 ? [a.rounds[scope]] : a.rounds), [a, scope]);
  // The round list is at the bottom; the charts it scopes are at the top.
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => { scrollRef.current?.scrollTo({ top: 0, behavior: "smooth" }); }, [scope]);
  const total = scope >= 0
    ? split([a.rounds[scope]])
    : split(a.rounds);
  const longest = Math.max(1, ...a.rounds.map((r) => split([r]).total));
  const legend: Array<[string, string, number, string?]> = [
    ["ph-input", S.traces.phaseInput, total.input, S.traces.phaseInputTip],
    ["ph-model", S.traces.phaseModel, total.model],
    ["ph-tool", S.traces.phaseTool, total.tool],
  ];
  return (
    <div className="tr-scroll" ref={scrollRef}>
      <section className="tr-card">
        <div className="tr-card-head">
          <span className="tr-card-title">
            {scope >= 0 ? roundLabel(a.rounds[scope], ordinalOf(a.rounds, scope), S) : S.traces.allRounds}
          </span>
          {scope >= 0 && <button type="button" className="tr-chip" onClick={() => setScope(-1)}>{S.traces.allRounds} ×</button>}
        </div>
        <SplitBar parts={total} />
        <div className="tr-legend">
          {legend.map(([cls, label, ms, tip]) => (
            <span key={cls} className="tr-legend-item" title={tip}>
              <i className={"tr-dot " + cls} />{label}
              <b>{fmtDurationMs(ms)}</b>
              <em>{total.total ? Math.round((ms / total.total) * 100) : 0}%</em>
            </span>
          ))}
        </div>
      </section>
      <section className="tr-card">
        <div className="tr-card-title">{S.traces.timeline}</div>
        <Timeline rounds={scoped} onOpenCall={onOpenCall} />
      </section>
      <div className="tr-grid2">
        <section className="tr-card">
          <div className="tr-card-title">{S.traces.byTool}</div>
          <ToolTable rounds={scoped} />
        </section>
        <section className="tr-card">
          <div className="tr-card-title">{S.traces.slowest}</div>
          <SlowestCalls rounds={scoped} onOpenCall={onOpenCall} />
        </section>
      </div>
      <section className="tr-card tr-rounds-dur">
        <div className="tr-card-title" title={S.traces.perRoundTip}>{S.traces.perRound}</div>
        {a.rounds.map((r, i) => {
          const parts = split([r]);
          return (
            <button key={r.taskIndex} type="button" className={"tr-dur-row" + (scope === i ? " on" : "")}
                    onClick={() => setScope(scope === i ? -1 : i)}>
              <span className="tr-dur-name">{roundLabel(r, ordinalOf(a.rounds, i), S)}</span>
              <span className="tr-dur-title" title={r.title}>{r.title}</span>
              <span className="tr-dur-track"><SplitBar parts={parts} scale={parts.total / longest} /></span>
              <span className="tr-dur-ms">{fmtDurationMs(r.durationMs)}</span>
            </button>
          );
        })}
      </section>
    </div>
  );
}

/** Rounds are numbered by user turn; compaction cards do not take a number. */
function ordinalOf(rounds: Round[], i: number) {
  let n = 0;
  for (let k = 0; k <= i; k++) if (!rounds[k].compaction) n++;
  return n;
}

function RoundsView({ a, onOpenRound }: { a: Analysis; onOpenRound: (i: number) => void }) {
  const S = useStrings();
  return (
    <div className="tr-scroll">
      <div className="tr-card tr-round-list">
        {a.rounds.map((r, i) => (
          <button key={r.taskIndex} type="button" className={"tr-round" + (r.compaction ? " compaction" : "")}
                  onClick={() => onOpenRound(i)}>
            <div className="tr-round-top">
              <span className="tr-round-name">{roundLabel(r, ordinalOf(a.rounds, i), S)}</span>
              <span className="tr-round-title" title={r.title}>{r.title}</span>
              <span className="tr-round-ms">{fmtDurationMs(r.durationMs)}</span>
            </div>
            <div className="tr-round-meta">
              <span>{S.traces.roundMeta(r.requests, r.toolCalls)}</span>
              {r.toolErrors > 0 && <span className="err">{S.traces.failed(r.toolErrors)}</span>}
              <span>↓ {fmtN(r.tokens.prompt)}{r.tokens.cacheHitPercent != null ? ` (${r.tokens.cacheHitPercent.toFixed(0)}%)` : ""}</span>
              <span>↑ {fmtN(r.tokens.output)}</span>
              <span>{money(r.costUsd)}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

type Row = { type: "head"; round: number } | { type: "call"; round: number; call: Call };

const ROW_H = 30;
const OVERSCAN = 12;

function CallsView({ a, roundFilter, setRoundFilter, focus, hits }: {
  a: Analysis; roundFilter: number; setRoundFilter: (i: number) => void;
  focus: number | null; hits: Set<number> | null;
}) {
  const S = useStrings();
  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    a.rounds.forEach((r, i) => {
      if (roundFilter >= 0 && roundFilter !== i) return;
      const calls = hits ? r.calls.filter((c) => hits.has(c.index)) : r.calls;
      if (!calls.length) return;
      if (roundFilter < 0) out.push({ type: "head", round: i });
      for (const c of calls) out.push({ type: "call", round: i, call: c });
    });
    return out;
  }, [a, roundFilter, hits]);

  const initial = useMemo(() => {
    const target = focus != null ? rows.findIndex((r) => r.type === "call" && r.call.index === focus) : -1;
    return target >= 0 ? target : rows.findIndex((r) => r.type === "call");
  }, [rows, focus]);
  const [sel, setSel] = useState(initial);
  useEffect(() => { setSel(initial); }, [initial]);
  const picked = sel >= 0 && rows[sel]?.type === "call" ? (rows[sel] as Extract<Row, { type: "call" }>) : null;

  const move = (delta: number) => {
    let i = sel;
    do { i += delta; } while (i >= 0 && i < rows.length && rows[i].type !== "call");
    if (i >= 0 && i < rows.length) setSel(i);
  };

  return (
    <div className="tr-split">
      <div className="tr-list-pane">
        <div className="tr-list-head">
          <select value={roundFilter} onChange={(e) => setRoundFilter(Number(e.target.value))}>
            <option value={-1}>{S.traces.allRounds}</option>
            {a.rounds.map((r, i) => (
              <option key={r.taskIndex} value={i}>
                {roundLabel(r, ordinalOf(a.rounds, i), S)} · {r.title.slice(0, 40)}
              </option>
            ))}
          </select>
          <span className="tr-list-count">{S.traces.callsN(rows.filter((r) => r.type === "call").length)}</span>
        </div>
        {rows.length ? (
          <VirtualList
            rows={rows}
            selected={sel}
            onSelect={setSel}
            onMove={move}
            render={(row) => row.type === "head"
              ? <RoundHead r={a.rounds[row.round]} label={roundLabel(a.rounds[row.round], ordinalOf(a.rounds, row.round), S)} />
              : <CallRow c={row.call} />}
          />
        ) : <div className="trace-empty"><p>{S.traces.searchNone}</p></div>}
      </div>
      <div className="tr-detail-pane">
        {picked
          ? <CallDetail key={picked.call.index} sessionId={a.session_id} c={picked.call}
                        roundName={roundLabel(a.rounds[picked.round], ordinalOf(a.rounds, picked.round), S)} />
          : <div className="trace-empty"><p>{S.traces.pickCall}</p></div>}
      </div>
    </div>
  );
}

/** Fixed-height rows, only the visible window in the DOM: a long session has
 *  thousands of calls, and drawing them all is what made the page crawl. */
function VirtualList({ rows, selected, onSelect, onMove, render }: {
  rows: Row[];
  selected: number;
  onSelect: (i: number) => void;
  onMove: (delta: number) => void;
  render: (row: Row) => ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [top, setTop] = useState(0);
  const [height, setHeight] = useState(600);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el);
    setHeight(el.clientHeight);
    return () => ro.disconnect();
  }, []);
  // Keep the selection in view (a focus call from the chat, or arrow keys).
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || selected < 0) return;
    const y = selected * ROW_H;
    if (y < el.scrollTop) el.scrollTop = Math.max(0, y - ROW_H * 3);
    else if (y + ROW_H > el.scrollTop + el.clientHeight) el.scrollTop = y - el.clientHeight + ROW_H * 4;
  }, [selected]);
  const from = Math.max(0, Math.floor(top / ROW_H) - OVERSCAN);
  const to = Math.min(rows.length, Math.ceil((top + height) / ROW_H) + OVERSCAN);
  const items: ReactNode[] = [];
  for (let i = from; i < to; i++) {
    const row = rows[i];
    items.push(
      <div key={i} className={"tr-row" + (row.type === "head" ? " head" : "") + (i === selected ? " sel" : "")}
           style={{ top: i * ROW_H, height: ROW_H }}
           onClick={() => row.type === "call" && onSelect(i)}>
        {render(row)}
      </div>,
    );
  }
  return (
    <div ref={ref} className="tr-list" tabIndex={0}
         onScroll={(e) => setTop(e.currentTarget.scrollTop)}
         onKeyDown={(e) => {
           if (e.key === "ArrowDown") { e.preventDefault(); onMove(1); }
           if (e.key === "ArrowUp") { e.preventDefault(); onMove(-1); }
         }}>
      <div style={{ height: rows.length * ROW_H, position: "relative" }}>{items}</div>
    </div>
  );
}

function RoundHead({ r, label }: { r: Round; label: string }) {
  return (
    <>
      <span className="tr-row-round">{label}</span>
      <span className="tr-row-sum" title={r.title}>{r.title}</span>
      <span className="tr-row-ms">{fmtDurationMs(r.durationMs)}</span>
    </>
  );
}

function toolArgsSummary(c: Call) {
  const name = c.toolName || "";
  return name && c.summary.startsWith(name + " ") ? c.summary.slice(name.length + 1) : c.summary;
}

function CallRow({ c }: { c: Call }) {
  const S = useStrings();
  if (c.kind === "model") {
    return (
      <>
        <span className="tr-tag k-model">{S.traces.kind.model}</span>
        <span className="tr-row-sum">{c.summary}</span>
        {c.tokens && <span className="tr-row-tok">{fmtN(c.tokens.prompt)} → {fmtN(c.tokens.output)}</span>}
        <span className="tr-row-ms">{c.durationMs != null ? fmtDurationMs(c.durationMs) : ""}</span>
      </>
    );
  }
  if (c.kind === "tool") {
    return (
      <>
        <span className={"tr-tag k-tool" + (c.isError ? " err" : "")} title={c.toolName}>{c.toolName || "tool"}</span>
        <span className={"tr-row-sum" + (c.isError ? " err" : "")}>{toolArgsSummary(c)}</span>
        {!!c.waitMs && <span className="tr-row-tok">⏸ {fmtDurationMs(c.waitMs)}</span>}
        <span className="tr-row-ms">{c.durationMs != null ? fmtDurationMs(c.durationMs) : ""}</span>
      </>
    );
  }
  return (
    <>
      <span className={"tr-tag k-" + c.kind}>{S.traces.kind[c.kind] || c.kind}</span>
      <span className="tr-row-sum">{c.summary}</span>
    </>
  );
}

const bodyCache = new Map<string, Body>();

function CallDetail({ sessionId, c, roundName }: { sessionId: string; c: Call; roundName: string }) {
  const S = useStrings();
  const key = `${sessionId}:${c.index}`;
  const [body, setBody] = useState<Body | null>(bodyCache.get(key) || null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (bodyCache.has(key)) return;
    let live = true;
    void api.fetchTraceCallBody(sessionId, c.index).then(({ ok, data }) => {
      if (!live) return;
      if (!ok || !data) { setFailed(true); return; }
      bodyCache.set(key, data as Body);
      setBody(data as Body);
    });
    return () => { live = false; };
  }, [key, sessionId, c.index]);

  const tabs: Array<[string, string]> = c.kind === "tool"
    ? [["result", S.traces.tabResult], ["args", S.traces.tabArgs], ...(body?.approval ? [["approval", S.traces.tabApproval] as [string, string]] : [])]
    : c.kind === "model"
      ? [["text", S.traces.tabReply], ["thinking", S.traces.tabThinking]]
      : [["text", S.traces.tabContent]];
  const [tab, setTab] = useState(tabs[0][0]);

  const title = c.kind === "tool" ? c.toolName || "tool" : S.traces.kind[c.kind] || c.kind;
  return (
    <div className="tr-detail">
      <div className="tr-detail-head">
        <span className={"tr-tag k-" + c.kind + (c.isError ? " err" : "")}>{title}</span>
        <span className="tr-detail-round">{roundName}</span>
      </div>
      <Facts c={c} />
      <div className="tr-dtabs" role="tablist">
        {tabs.map(([k, label]) => (
          <button key={k} type="button" role="tab" aria-selected={tab === k}
                  className={"tr-dtab" + (tab === k ? " on" : "")} onClick={() => setTab(k)}>{label}</button>
        ))}
      </div>
      <div className="tr-dbody">
        {failed ? <p className="trace-note">{S.traces.bodyFailed}</p>
          : !body ? <p className="trace-note">{S.traces.loadingBody}</p>
          : <BodyView c={c} body={body} tab={tab} />}
      </div>
    </div>
  );
}

function BodyView({ c, body, tab }: { c: Call; body: Body; tab: string }) {
  const S = useStrings();
  if (c.kind === "tool" && c.toolName === "task") {
    if (tab === "result" && body.taskResult) {
      return <TaskCard brief={body.taskCall?.description || ""} subagentType={body.taskCall?.subagentType}
                       result={taskResultOf(body.taskResult)} />;
    }
  }
  const text = tab === "args" ? body.args
    : tab === "result" ? body.result
    : tab === "approval" ? body.approval
    : tab === "thinking" ? body.thinking
    : body.text;
  if (!text) return <p className="trace-note">{S.traces.empty}</p>;
  return (
    <div className="tr-pre-wrap">
      <button className="entry-copy" title={S.common.copy}
              onClick={() => { void navigator.clipboard.writeText(text).then(() => showToast(S.common.copied)); }}>
        <IconCopy />
      </button>
      <pre className={"tr-pre" + (tab === "result" && c.isError ? " err" : "")}>{text}</pre>
    </div>
  );
}

function Facts({ c }: { c: Call }) {
  const S = useStrings();
  const rows: Array<[string, string]> = [];
  if (c.ts) rows.push([S.traces.dStarted, new Date(c.ts).toLocaleTimeString()]);
  if (c.durationMs != null) rows.push([S.traces.dDuration, fmtDurationMs(c.durationMs)]);
  if (c.waitMs) rows.push([S.traces.dWait, fmtDurationMs(c.waitMs)]);
  if (c.kind === "tool" && c.decision) rows.push([S.traces.dDecision, c.decision]);
  if (c.kind === "model") {
    if (c.status) rows.push([S.traces.dStatus, c.status]);
    if (c.toolCalls) rows.push([S.traces.dToolCalls, String(c.toolCalls)]);
  }
  const tok = c.kind === "model" ? c.tokens : null;
  const active = (c.durationMs || 0) - (c.waitMs || 0);
  return (
    <>
      <dl className="tr-facts">
        {rows.map(([k, v]) => <Fact key={k} k={k} v={v} />)}
      </dl>
      {c.kind === "model" && (tok ? (
        <dl className="tr-facts tr-usage">
          <Fact k={S.traces.dPrompt} v={fmtN(tok.prompt)} />
          <Fact k={S.traces.dFresh} v={fmtN(tok.input)} />
          <Fact k={S.traces.dCache}
                v={`${fmtN(tok.cacheRead)}${tok.cacheHitPercent != null ? ` / ${tok.cacheHitPercent.toFixed(1)}%` : ""}`} />
          {!!tok.cacheWrite && <Fact k={S.traces.dCacheWrite} v={fmtN(tok.cacheWrite)} />}
          <Fact k={S.traces.dOutput} v={fmtN(tok.output)} />
          {active > 0 && <Fact k={S.traces.dTps} v={fmtTps(tok.output / (active / 1000))} />}
        </dl>
      ) : <p className="trace-note">{S.traces.noTokens}</p>)}
    </>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return <div className="tr-fact"><dt>{k}</dt><dd>{v}</dd></div>;
}

function taskResultOf(t: TaskFields): TaskResult {
  return {
    success: !!t.success,
    status: t.status || (t.success ? "completed" : "error"),
    subagentName: t.subagentName || "",
    subagentType: t.subagentType || "",
    executionMs: t.executionMs,
    toolCount: t.toolCount || 0,
    steps: t.steps || [],
    stepsTruncated: t.stepsTruncated,
    result: t.result || "",
    error: t.error || "",
    runId: t.runId,
  };
}
