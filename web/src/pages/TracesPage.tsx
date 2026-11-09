import { lazy, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, Navigate, useSearchParams } from "react-router";
import * as api from "../api";
import { useStrings, type Strings } from "../i18n";
import { fmtCost, fmtDurationMs, fmtFileSize, fmtN, fmtTps } from "../lib/format";
import { IconChat, IconCopy, IconSearch } from "../icons";
import { showToast, useAppState } from "../store";
import { switchTo } from "../sessions";
import { TaskCard } from "../components/TaskCard";
import type { TaskResult } from "../lib/workActivity";

const ChatMarkdown = lazy(() => import("../components/ChatMarkdown").then((m) => ({ default: m.ChatMarkdown })));

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
  toolModel?: string;
  toolModelTier?: string;
  isError?: boolean;
  decision?: string;
  resultSummary?: string;
};

type TaskFields = Partial<TaskResult> & { description?: string; subagentType?: string; model?: string; modelTier?: string };

type ModelToolCall = {
  index: number;
  toolCallId?: string;
  toolName: string;
  summary: string;
  args?: string;
  resultSummary?: string;
  toolModel?: string;
  toolModelTier?: string;
  isError?: boolean;
  durationMs?: number | null;
};

type Body = {
  args?: string;
  result?: string;
  approval?: string;
  thinking?: string;
  text?: string;
  toolCalls?: ModelToolCall[];
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
  const s = useAppState();
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
    // Leaving for another session before this one answers must not let the late
    // reply overwrite the page (or clear its loading state) for the new one.
    let current = true;
    void api.fetchTraceAnalysis(sessionId).then(({ ok, status, data }) => {
      if (!current) return;
      setLoading(false);
      if (ok) { setAnalysis(data as Analysis); setErrStatus(0); return; }
      setAnalysis(null);
      setErrStatus(status || 500);
    });
    return () => { current = false; };
  }, [sessionId]);

  if (!sessionId) return <Navigate to="/chat" replace />;

  const sessTitle = s.sessions[sessionId]?.title || analysis?.file.name || sessionId;

  return (
    <div className="main">
      <div className="topbar chat-topbar">
        <div className="topbar-row">
          <div className="tb-l">
            <div className="tb-chat-info">
              <span className="tb-chat-icon"><IconChat /></span>
              <span className="tb-chat-title">{sessTitle}</span>
            </div>
          </div>
          <div className="tb-r" />
        </div>
        <div className="topbar-tabs" role="tablist">
          <Link className="tb-tab" role="tab" aria-selected="false" to="/chat" onClick={() => switchTo(sessionId)}>
            {S.chat.chatTab}
          </Link>
          <span className="tb-tab active" role="tab" aria-selected="true">
            {S.chat.viewTrace}
          </span>
        </div>
      </div>
      <div className="trace-main">
        {loading && <div className="trace-empty"><p>{S.common.loading}</p></div>}
        {!loading && !!errStatus && (
          <div className="trace-empty">
            <p>{errStatus === 404 ? S.traces.missing : S.traces.loadFailed(errStatus)}</p>
          </div>
        )}
        {analysis && <TraceDetail key={sessionId} a={analysis} focusCall={focusCall} />}
      </div>
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
  const [showDuration, setShowDuration] = useState(a.hasTimeline);
  // -1 = all rounds. Scopes both the top duration timeline and the call table.
  const [scope, setScope] = useState(-1);
  const [focus, setFocus] = useState<number | null>(focusInit);
  const [query, setQuery] = useState("");
  const hits = useTraceSearch(a.session_id, query);
  const [collapsedRounds, setCollapsedRounds] = useState<Set<number>>(() => new Set());
  const [collapsedAssistants, setCollapsedAssistants] = useState<Set<number>>(() => new Set());

  const allRoundIndices = useMemo(() => a.rounds.map((_, i) => i), [a.rounds]);
  const collapsibleAssistantIndices = useMemo(() => {
    const ids: number[] = [];
    for (const r of a.rounds) {
      for (const c of r.calls) {
        if (c.kind === "model" && (c.toolCalls || 0) > 0) ids.push(c.index);
      }
    }
    return ids;
  }, [a.rounds]);

  const allRoundsCollapsed = allRoundIndices.length > 0 && allRoundIndices.every((i) => collapsedRounds.has(i));
  const allToolsCollapsed = collapsibleAssistantIndices.length > 0
    && collapsibleAssistantIndices.every((idx) => collapsedAssistants.has(idx));

  const toggleAllRounds = () => {
    setCollapsedRounds(allRoundsCollapsed ? new Set() : new Set(allRoundIndices));
  };
  const toggleRound = (rIdx: number) => {
    setCollapsedRounds((prev) => {
      const next = new Set(prev);
      if (next.has(rIdx)) next.delete(rIdx); else next.add(rIdx);
      return next;
    });
  };
  const toggleAllTools = () => {
    setCollapsedAssistants(allToolsCollapsed ? new Set() : new Set(collapsibleAssistantIndices));
  };
  const toggleAssistantTools = (modelIdx: number) => {
    setCollapsedAssistants((prev) => {
      const next = new Set(prev);
      if (next.has(modelIdx)) next.delete(modelIdx); else next.add(modelIdx);
      return next;
    });
  };

  const openCall = (index: number) => {
    a.rounds.forEach((r, rIdx) => {
      let curModelIdx: number | null = null;
      for (const c of r.calls) {
        if (c.kind === "model") curModelIdx = c.index;
        if (c.index === index) {
          setCollapsedRounds((prev) => {
            if (!prev.has(rIdx)) return prev;
            const next = new Set(prev); next.delete(rIdx); return next;
          });
          if (c.kind === "tool" && curModelIdx != null) {
            const mIdx = curModelIdx;
            setCollapsedAssistants((prev) => {
              if (!prev.has(mIdx)) return prev;
              const next = new Set(prev); next.delete(mIdx); return next;
            });
          }
        }
      }
    });
    setFocus(index);
  };

  return (
    <>
      <div className="trace-meta-bar">
        {a.meta.model && <span className="trace-meta-item">{a.meta.model}</span>}
        {a.meta.gitBranch && <span className="trace-meta-item">git:{a.meta.gitBranch}</span>}
        <span className="trace-meta-item">{fmtFileSize(a.file.sizeBytes)}</span>
        <span className="trace-meta-path" title={a.file.path}>
          <code>{a.file.path}</code>
          <button type="button" className="trace-cwd-copy" title={S.traces.copyPath} aria-label={S.traces.copyPath}
                  onClick={() => { void navigator.clipboard.writeText(a.file.path); showToast(S.traces.copiedPath); }}>
            <IconCopy />
          </button>
        </span>
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
        <div className="tr-toolbar-actions">
          <button
            type="button"
            className={"tr-tbtn" + (showDuration ? " on" : "")}
            aria-pressed={showDuration}
            onClick={() => setShowDuration((v) => !v)}
          >
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
              <circle cx="8" cy="8" r="5.5" />
              <path d="M8 4.8V8l2.2 1.5" strokeLinecap="round" />
            </svg>
            <span>{S.traces.tabDuration}</span>
          </button>
          <button
            type="button"
            className={"tr-tbtn" + (!allRoundsCollapsed ? " on" : "")}
            aria-pressed={!allRoundsCollapsed}
            title={allRoundsCollapsed ? S.traces.expandTurns : S.traces.collapseTurns}
            onClick={toggleAllRounds}
          >
            <span className="tr-tbtn-ico" aria-hidden="true">{allRoundsCollapsed ? "⊞" : "⊟"}</span>
            <span>{S.traces.tabRounds}</span>
          </button>
          <button
            type="button"
            className={"tr-tbtn" + (!allToolsCollapsed ? " on" : "")}
            aria-pressed={!allToolsCollapsed}
            title={allToolsCollapsed ? S.traces.expandTools : S.traces.collapseTools}
            onClick={toggleAllTools}
          >
            <span className="tr-tbtn-ico" aria-hidden="true">{allToolsCollapsed ? "⊞" : "⊟"}</span>
            <span>{S.traces.tabCalls}</span>
          </button>
          <select className="tr-scope-select" value={scope} onChange={(e) => setScope(Number(e.target.value))}>
            <option value={-1}>{S.traces.allRounds}</option>
            {a.rounds.map((r, i) => (
              <option key={r.taskIndex} value={i}>
                {roundLabel(r, ordinalOf(a.rounds, i), S)} · {r.title.slice(0, 40)}
              </option>
            ))}
          </select>
        </div>
        <label className="tr-search">
          <IconSearch />
          <input type="search" value={query} placeholder={S.traces.search}
                 onChange={(e) => { setQuery(e.target.value); if (e.target.value.trim()) setScope(-1); }}
                 onKeyDown={(e) => { if (e.key === "Escape") setQuery(""); }} />
          {query.trim() && (
            <span className="tr-search-n">
              {hits.loading ? S.traces.searching : hits.indices.size ? S.traces.searchHits(hits.indices.size) : S.traces.searchNone}
            </span>
          )}
        </label>
      </div>

      {showDuration && (
        a.hasTimeline
          ? <DurationBanner a={a} scope={scope} setScope={setScope} onOpenCall={openCall} />
          : <p className="trace-note">{S.traces.noTimeline}</p>
      )}
      <CallsView
        a={a}
        roundFilter={scope}
        focus={focus}
        onOpenCall={openCall}
        hits={query.trim() && !hits.loading ? hits.indices : null}
        collapsedRounds={collapsedRounds}
        onToggleRound={toggleRound}
        collapsedAssistants={collapsedAssistants}
        onToggleAssistant={toggleAssistantTools}
      />
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
    const sum = c.kind === "tool"
      ? (c.resultSummary ? `${toolArgsSummary(c)} → ${c.resultSummary}` : toolArgsSummary(c))
      : c.summary;
    return `${head}${what ? " · " + what : ""} · ${fmtDurationMs(sp.end - sp.start)}\n${sum}`;
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
    </div>
  );
}

function DurationBanner({ a, scope, setScope, onOpenCall }: {
  a: Analysis; scope: number; setScope: (i: number) => void; onOpenCall: (index: number) => void;
}) {
  const S = useStrings();
  const scoped = useMemo(() => (scope >= 0 ? [a.rounds[scope]] : a.rounds), [a, scope]);
  const total = scope >= 0
    ? split([a.rounds[scope]])
    : split(a.rounds);
  const legend: Array<[string, string, number, string?]> = [
    ["ph-input", S.traces.phaseInput, total.input, S.traces.phaseInputTip],
    ["ph-model", S.traces.phaseModel, total.model],
    ["ph-tool", S.traces.phaseTool, total.tool],
  ];
  return (
    <section className="tr-card tr-duration-banner">
      <div className="tr-banner-top">
        <span className="tr-card-title">
          {scope >= 0 ? roundLabel(a.rounds[scope], ordinalOf(a.rounds, scope), S) : S.traces.allRounds}
        </span>
        {scope >= 0 && <button type="button" className="tr-chip" onClick={() => setScope(-1)}>{S.traces.allRounds} ×</button>}
        <div className="tr-legend">
          {legend.map(([cls, label, ms, tip]) => (
            <span key={cls} className="tr-legend-item" title={tip}>
              <i className={"tr-dot " + cls} />{label}
              <b>{fmtDurationMs(ms)}</b>
              <em>{total.total ? Math.round((ms / total.total) * 100) : 0}%</em>
            </span>
          ))}
        </div>
      </div>
      <Timeline rounds={scoped} onOpenCall={onOpenCall} />
    </section>
  );
}

/** Rounds are numbered by user turn; compaction cards do not take a number. */
function ordinalOf(rounds: Round[], i: number) {
  let n = 0;
  for (let k = 0; k <= i; k++) if (!rounds[k].compaction) n++;
  return n;
}

type Row =
  | { type: "head"; round: number; collapsed: boolean }
  | { type: "call"; round: number; call: Call; toolsCollapsed?: boolean };

const ROW_H = 30;
const OVERSCAN = 12;

function CallsView({
  a, roundFilter, focus, onOpenCall, hits,
  collapsedRounds, onToggleRound, collapsedAssistants, onToggleAssistant,
}: {
  a: Analysis; roundFilter: number;
  focus: number | null; onOpenCall: (index: number) => void; hits: Set<number> | null;
  collapsedRounds: Set<number>; onToggleRound: (round: number) => void;
  collapsedAssistants: Set<number>; onToggleAssistant: (modelIndex: number) => void;
}) {
  const S = useStrings();
  const longestRound = useMemo(() => Math.max(1, ...a.rounds.map((r) => split([r]).total)), [a.rounds]);
  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    a.rounds.forEach((r, i) => {
      if (roundFilter >= 0 && roundFilter !== i) return;
      const calls = hits ? r.calls.filter((c) => hits.has(c.index)) : r.calls;
      if (!calls.length) return;
      const roundFolded = !hits && collapsedRounds.has(i);
      out.push({ type: "head", round: i, collapsed: roundFolded });
      if (roundFolded) return;
      let activeModelCollapsed = false;
      for (const c of calls) {
        if (c.kind === "model") {
          activeModelCollapsed = !hits && collapsedAssistants.has(c.index);
          out.push({ type: "call", round: i, call: c, toolsCollapsed: activeModelCollapsed });
          continue;
        }
        if (c.kind === "tool" && activeModelCollapsed) continue;
        out.push({ type: "call", round: i, call: c });
      }
    });
    return out;
  }, [a, roundFilter, hits, collapsedRounds, collapsedAssistants]);

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
        {rows.length ? (
          <VirtualList
            rows={rows}
            selected={sel}
            onSelect={setSel}
            onToggleRound={onToggleRound}
            onMove={move}
            render={(row) => row.type === "head"
              ? (
                <RoundHead
                  r={a.rounds[row.round]}
                  label={roundLabel(a.rounds[row.round], ordinalOf(a.rounds, row.round), S)}
                  collapsed={row.collapsed}
                  longest={longestRound}
                />
              )
              : (
                <CallRow
                  c={row.call}
                  toolsCollapsed={row.toolsCollapsed}
                  onToggleTools={() => onToggleAssistant(row.call.index)}
                />
              )}
          />
        ) : <div className="trace-empty"><p>{S.traces.searchNone}</p></div>}
      </div>
      <div className="tr-detail-pane">
        {picked
          ? <CallDetail key={picked.call.index} sessionId={a.session_id} c={picked.call}
                        round={a.rounds[picked.round]}
                        roundName={roundLabel(a.rounds[picked.round], ordinalOf(a.rounds, picked.round), S)}
                        onOpenCall={onOpenCall} />
          : <div className="trace-empty"><p>{S.traces.pickCall}</p></div>}
      </div>
    </div>
  );
}

/** Fixed-height rows, only the visible window in the DOM: a long session has
 *  thousands of calls, and drawing them all is what made the page crawl. */
function VirtualList({ rows, selected, onSelect, onToggleRound, onMove, render }: {
  rows: Row[];
  selected: number;
  onSelect: (i: number) => void;
  onToggleRound: (round: number) => void;
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
           onClick={() => row.type === "head" ? onToggleRound(row.round) : onSelect(i)}>
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

function RoundHead({ r, label, collapsed, longest }: { r: Round; label: string; collapsed: boolean; longest: number }) {
  const S = useStrings();
  const parts = split([r]);
  return (
    <>
      <span className="tr-fold-chev" aria-hidden="true">{collapsed ? "▸" : "▾"}</span>
      <span className="tr-row-round">{label}</span>
      <span className="tr-row-sum" title={r.title}>{r.title}</span>
      <span className="tr-round-inline-meta">{S.traces.roundMeta(r.requests, r.toolCalls)}</span>
      <span className="tr-round-inline-bar"><SplitBar parts={parts} scale={parts.total / longest} /></span>
      <span className="tr-row-ms">{fmtDurationMs(r.durationMs)}</span>
    </>
  );
}

function toolArgsSummary(c: Call) {
  const name = c.toolName || "";
  return name && c.summary.startsWith(name + " ") ? c.summary.slice(name.length + 1) : c.summary;
}

/**
 * Tool call row summary (deepseek-harness style): `args → result`. Both halves
 * keep their full single-line text in the DOM and clip via CSS ellipsis, so
 * selecting and copying a row copies the elided command tail and output too.
 */
function ToolCallSummary({ c, className = "tr-row-sum" }: { c: Call; className?: string }) {
  const args = toolArgsSummary(c);
  const res = c.resultSummary;
  if (!res) {
    return <span className={className + (c.isError ? " err" : "")} title={args}>{args}</span>;
  }
  return (
    <span className={className + " has-result"} title={args ? `${args} → ${res}` : res}>
      <span className="tr-call-req">{args}</span>
      <span className={"tr-call-res" + (c.isError ? " err" : "")}>
        <span className="tr-call-arr">{" → "}</span>
        <span className="tr-call-res-text">{res}</span>
      </span>
    </span>
  );
}

function formatToolModelBadge(model?: string, tier?: string) {
  if (!model && !tier) return "";
  const short = model && model.includes("/") ? model.slice(model.indexOf("/") + 1) : (model || "");
  const tierLabel = tier === "main" ? "main" : tier === "auxiliary" ? "aux" : "";
  if (tierLabel && short) return `${tierLabel} · ${short}`;
  return short || tierLabel;
}

function CallRow({ c, toolsCollapsed, onToggleTools }: { c: Call; toolsCollapsed?: boolean; onToggleTools?: () => void }) {
  const S = useStrings();
  if (c.kind === "model") {
    const text = c.summary || ((c.toolCalls || 0) > 0 ? S.traces.toolCallOnly : "");
    return (
      <>
        <span className="tr-tag k-model">{S.traces.kind.model}</span>
        <span className={"tr-row-sum" + (!c.summary ? " muted" : "")} title={text}>{text}</span>
        {!!c.toolCalls && (
          <button
            type="button"
            className={"tr-tool-badge" + (toolsCollapsed ? " folded" : "")}
            title={toolsCollapsed ? S.traces.expandTools : S.traces.collapseTools}
            onClick={(e) => { e.stopPropagation(); onToggleTools?.(); }}
          >
            🔧 {c.toolCalls}
          </button>
        )}
        {c.tokens && <span className="tr-row-tok">{fmtN(c.tokens.prompt)} → {fmtN(c.tokens.output)}</span>}
        <span className="tr-row-ms">{c.durationMs != null ? fmtDurationMs(c.durationMs) : ""}</span>
      </>
    );
  }
  if (c.kind === "tool") {
    const modelBadge = formatToolModelBadge(c.toolModel, c.toolModelTier);
    return (
      <>
        <span className={"tr-tag k-tool" + (c.isError ? " err" : "")} title={c.toolName}>{c.toolName || "tool"}</span>
        {modelBadge && (
          <span className={"tr-tool-model" + (c.toolModelTier === "main" ? " main" : "")} title={c.toolModel || c.toolModelTier}>
            {modelBadge}
          </span>
        )}
        <ToolCallSummary c={c} className="tr-row-sum" />
        {!!c.waitMs && <span className="tr-row-tok">⏸ {fmtDurationMs(c.waitMs)}</span>}
        <span className="tr-row-ms">{c.durationMs != null ? fmtDurationMs(c.durationMs) : ""}</span>
      </>
    );
  }
  return (
    <>
      <span className={"tr-tag k-" + c.kind}>{S.traces.kind[c.kind] || c.kind}</span>
      <span className="tr-row-sum" title={c.summary}>{c.summary}</span>
    </>
  );
}

const bodyCache = new Map<string, Body>();

function deriveModelToolCalls(round: Round, modelCall: Call, fromBody?: ModelToolCall[]): ModelToolCall[] {
  if (fromBody && fromBody.length > 0) return fromBody;
  const out: ModelToolCall[] = [];
  let collect = false;
  for (const item of round.calls) {
    if (item.index === modelCall.index) { collect = true; continue; }
    if (!collect) continue;
    if (item.kind === "model") break;
    if (item.kind === "tool") {
      out.push({
        index: item.index,
        toolCallId: item.toolCallId,
        toolName: item.toolName || "tool",
        summary: item.summary,
        resultSummary: item.resultSummary,
        toolModel: item.toolModel,
        toolModelTier: item.toolModelTier,
        isError: item.isError,
        durationMs: item.durationMs,
      });
    }
  }
  return out;
}

function CallDetail({ sessionId, c, round, roundName, onOpenCall }: {
  sessionId: string; c: Call; round: Round; roundName: string; onOpenCall: (index: number) => void;
}) {
  const S = useStrings();
  const key = `${sessionId}:${c.index}`;
  const [body, setBody] = useState<Body | null>(bodyCache.get(key) || null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (bodyCache.has(key)) return;
    // Stepping through the list with the arrow keys moves the selection faster
    // than a body arrives. Wait for it to settle, and drop a request the user
    // has already moved past, so the server only reads the ones that are looked at.
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => {
      void api.fetchTraceCallBody(sessionId, c.index, ctrl.signal).then(({ ok, data }) => {
        if (ctrl.signal.aborted) return;
        if (!ok || !data) { setFailed(true); return; }
        bodyCache.set(key, data as Body);
        setBody(data as Body);
      });
    }, 120);
    return () => { window.clearTimeout(timer); ctrl.abort(); };
  }, [key, sessionId, c.index]);

  const tabs: Array<[string, string]> = c.kind === "tool"
    ? [["result", S.traces.tabResult], ["args", S.traces.tabArgs], ...(body?.approval ? [["approval", S.traces.tabApproval] as [string, string]] : [])]
    : c.kind === "model"
      ? [["overview", S.traces.tabOverview], ["text", S.traces.tabReply], ["thinking", S.traces.tabThinking]]
      : [["text", S.traces.tabContent]];
  const [tab, setTab] = useState(tabs[0][0]);
  const modelTools = useMemo(
    () => (c.kind === "model" ? deriveModelToolCalls(round, c, body?.toolCalls) : []),
    [c, round, body?.toolCalls],
  );

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
          : <BodyView c={c} body={body} tab={tab} modelTools={modelTools} onOpenCall={onOpenCall} />}
      </div>
    </div>
  );
}

function AssistantOverview({ body, modelTools, onOpenCall }: {
  body: Body; modelTools: ModelToolCall[]; onOpenCall: (index: number) => void;
}) {
  const S = useStrings();
  const [thinkOpen, setThinkOpen] = useState(!body.text && modelTools.length === 0);
  const hasAny = Boolean(body.thinking?.trim() || body.text?.trim() || modelTools.length);
  if (!hasAny) return <p className="trace-note">{S.traces.empty}</p>;
  return (
    <div className="tr-assistant-overview">
      {body.thinking?.trim() && (
        <div className={"tr-think-quote" + (thinkOpen ? " open" : "")}>
          <button type="button" className="tr-think-toggle" aria-expanded={thinkOpen} onClick={() => setThinkOpen((v) => !v)}>
            <span>{S.traces.tabThinking}</span>
            <span className="tr-think-chev" aria-hidden="true">{thinkOpen ? "▾" : "▸"}</span>
          </button>
          {thinkOpen && (
            <div className="tr-think-body">
              <Suspense fallback={<pre className="tr-pre">{body.thinking}</pre>}>
                <ChatMarkdown text={body.thinking} streaming={false} />
              </Suspense>
            </div>
          )}
        </div>
      )}
      {body.text?.trim() && (
        <div className="tr-assistant-reply">
          <Suspense fallback={<pre className="tr-pre">{body.text}</pre>}>
            <ChatMarkdown text={body.text} streaming={false} />
          </Suspense>
        </div>
      )}
      {modelTools.length > 0 && (
        <ul className="tr-assistant-tools">
          {modelTools.map((tc) => {
            const args = tc.toolName && tc.summary.startsWith(tc.toolName + " ")
              ? tc.summary.slice(tc.toolName.length + 1)
              : tc.summary;
            const modelBadge = formatToolModelBadge(tc.toolModel, tc.toolModelTier);
            return (
              <li key={tc.toolCallId || tc.index}>
                <button
                  type="button"
                  className={"tr-assistant-tool-btn" + (tc.isError ? " err" : "")}
                  title={tc.resultSummary ? `${tc.toolName} ${args} → ${tc.resultSummary}` : `${tc.toolName} ${args}`}
                  onClick={() => onOpenCall(tc.index)}
                >
                  <svg className="tr-assistant-tool-ico" width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94z"
                          stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span className="tr-assistant-tool-name">{tc.toolName}</span>
                  {modelBadge && (
                    <span className={"tr-tool-model" + (tc.toolModelTier === "main" ? " main" : "")} title={tc.toolModel || tc.toolModelTier}>
                      {modelBadge}
                    </span>
                  )}
                  <span className="tr-assistant-tool-args">{args}</span>
                  {tc.resultSummary && (
                    <span className="tr-assistant-tool-res">
                      <span className="tr-call-arr">{" → "}</span>
                      <span>{tc.resultSummary}</span>
                    </span>
                  )}
                  {tc.durationMs != null && <span className="tr-assistant-tool-ms">{fmtDurationMs(tc.durationMs)}</span>}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function BodyView({ c, body, tab, modelTools, onOpenCall }: {
  c: Call; body: Body; tab: string; modelTools: ModelToolCall[]; onOpenCall: (index: number) => void;
}) {
  const S = useStrings();
  if (c.kind === "model" && tab === "overview") {
    return <AssistantOverview body={body} modelTools={modelTools} onOpenCall={onOpenCall} />;
  }
  if (c.kind === "tool" && c.toolName === "task") {
    if (tab === "result" && body.taskResult) {
      return <TaskCard brief={body.taskCall?.description || ""} subagentType={body.taskCall?.subagentType}
                       model={body.taskResult.model || body.taskCall?.model || c.toolModel}
                       modelTier={body.taskResult.modelTier || body.taskCall?.modelTier || c.toolModelTier}
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
  if (c.kind === "tool" && (c.toolModel || c.toolModelTier)) {
    rows.push([S.traces.dModel, c.toolModelTier && c.toolModel ? `${c.toolModel} (${c.toolModelTier})` : (c.toolModel || c.toolModelTier!)]);
  }
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
    model: t.model,
    modelTier: t.modelTier,
    executionMs: t.executionMs,
    toolCount: t.toolCount || 0,
    steps: t.steps || [],
    stepsTruncated: t.stepsTruncated,
    result: t.result || "",
    error: t.error || "",
    runId: t.runId,
  };
}
