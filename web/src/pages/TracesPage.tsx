import { Fragment, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import * as api from "../api";
import { AppShell } from "../components/AppShell";
import { SessionTree } from "../components/SessionTree";
import { getStrings, useStrings, type Strings } from "../i18n";
import { fmtFileSize, fmtN, shortenPath } from "../lib/format";
import { IconClose, IconCopy, IconFolder, IconSearch } from "../icons";
import {
  getState, projectIdForDir, projectNameForDir, showToast, useAppState,
} from "../store";

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

const PHASES: Array<{ key: string; label: (S: Strings) => string; cls: string }> = [
  { key: "thinking", label: (S) => S.traces.phaseThinking, cls: "ph-thinking" },
  { key: "text", label: (S) => S.traces.phaseText, cls: "ph-text" },
  { key: "toolArgs", label: (S) => S.traces.phaseArgs, cls: "ph-args" },
  { key: "toolWait", label: (S) => S.traces.phaseWait, cls: "ph-wait" },
  { key: "toolExec", label: (S) => S.traces.phaseExec, cls: "ph-exec" },
  { key: "other", label: (S) => S.traces.phaseOther, cls: "ph-other" },
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
  const S = useStrings();
  // Subscribed, not just read: the rail groups by the working directory the
  // client store knows for each session, and that arrives with `loadSessions`.
  const state = useAppState();
  const [params, setParams] = useSearchParams();
  const sessionId = params.get("sessionId") || "";
  const [sessions, setSessions] = useState<any[]>([]);
  const [search, setSearch] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  // The failure is kept as its status code, not as a sentence: a message
  // rendered once would stay in the language it was written in when the user
  // switches languages.
  const [errStatus, setErrStatus] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void api.fetchSessions().then(({ data }) => setSessions(data?.sessions || []));
  }, []);

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

  // Grouped by working directory, the same grouping the chat sidebar uses: a
  // trace is read against the project it ran in, and the folder is what tells
  // two same-named conversations apart.
  const groups = useMemo(() => {
    const q = search.trim().toLowerCase();
    const known = getState().sessions;
    const fallbackDir = getState().serverDir || "";
    const by = new Map<string, { name: string; dir: string; items: any[] }>();
    for (const s of sessions) {
      if (q && !(s.name || "").toLowerCase().includes(q)
          && !(s.session_id || "").toLowerCase().includes(q)) continue;
      const dir = known[s.session_id]?.dir || fallbackDir;
      const id = projectIdForDir(dir);
      let group = by.get(id);
      if (!group) by.set(id, group = { name: projectNameForDir(dir), dir, items: [] });
      group.items.push(s);
    }
    return [...by.values()];
  }, [sessions, search, state.rev]);
  const shown = groups.reduce((n, g) => n + g.items.length, 0);

  return (
    <AppShell active="traces" list={<><div className="project-list-label">{S.chat.projects}</div><SessionTree /></>}>
      {/* The middle column. Traces are read by walking a list and comparing, so
          the picker stays on screen next to what it opened — and the left
          sidebar keeps the conversation tree, because leaving a trace for the
          conversation it describes is the most common next move. */}
      <aside className="trace-rail">
        <label className="trace-search">
          <IconSearch />
          <input placeholder={S.traces.searchSessions} value={search} onChange={(e) => setSearch(e.target.value)} />
          {search && <button className="search-clear" onClick={() => setSearch("")}><IconClose /></button>}
        </label>
        <div className="trace-rail-label">{S.traces.sessions} {shown ? `(${shown})` : ""}</div>
        <div className="trace-rail-list">
          {!shown && <div className="s-empty">{S.traces.noSessions}</div>}
          {groups.map((g) => (
            <div className="trace-rail-group" key={g.dir || g.name}>
              <div className="trace-rail-project" title={g.dir}>
                <IconFolder />
                <span className="trp-name">{g.name}</span>
                <span className="trp-count">{g.items.length}</span>
              </div>
              {g.items.map((s) => (
                <button
                  key={s.session_id}
                  className={"trace-rail-item" + (s.session_id === sessionId ? " active" : "")}
                  onClick={() => setParams({ sessionId: s.session_id })}
                >
                  <span className="tri-title">{s.name || s.session_id}</span>
                  <span className="tri-meta">
                    <span>{S.traces.rounds(s.user_count || 0)}</span>
                    <span>{fmtFileSize(s.size_bytes || 0)}</span>
                  </span>
                </button>
              ))}
            </div>
          ))}
        </div>
      </aside>

      <div className="trace-main">
        {!sessionId && (
          <div className="trace-empty">
            <h2>{S.traces.pickTitle}</h2>
            <p>{S.traces.pickDesc}</p>
          </div>
        )}
        {sessionId && loading && <div className="trace-empty"><p>{S.common.loading}</p></div>}
        {sessionId && !loading && !!errStatus && (
          <div className="trace-empty">
            <p>{errStatus === 404 ? S.traces.missing : S.traces.loadFailed(errStatus)}</p>
          </div>
        )}
        {analysis && <TraceDetail a={analysis} />}
      </div>
    </AppShell>
  );
}

function TraceDetail({ a }: { a: Analysis }) {
  const S = useStrings();
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
          <button className="dp-btn" onClick={() => { void navigator.clipboard.writeText(a.file.path); showToast(S.traces.copiedPath); }}>
            <IconCopy /> {S.traces.copyPath}
          </button>
        </div>
      </div>

      <section className="trace-card">
        <div className="trace-card-title">{S.traces.totals}</div>
        <div className="stat-grid">
          <Stat label={S.traces.statRounds} value={String(t.rounds)} />
          <Stat label={S.traces.statInput} value={fmtN(tokens.input)} hint={tokens.cacheRead ? S.traces.cacheHit(fmtN(tokens.cacheRead)) : undefined} />
          <Stat label={S.traces.statCost} value={money(t.costUsd)} hint={a.meta.model ? undefined : S.traces.noModelInfo} />
          <Stat label={S.traces.statToolCalls} value={String(t.toolCalls)} hint={t.toolErrors ? S.traces.failed(t.toolErrors) : S.traces.succeeded(t.toolOk)} />
          <Stat label={S.traces.statWait} value={ms(t.waitMs)} />
          <Stat label={S.traces.statElapsed} value={ms(t.elapsedMs)} hint={S.traces.modelTime(ms(t.llmMs))} />
          <Stat label={S.traces.statRequests} value={String(t.requests)} hint={a.reconnectCount ? S.traces.retried(a.reconnectCount) : undefined} />
          <Stat label={S.traces.statOutput} value={fmtN(tokens.output)} />
          <Stat label={S.traces.statTps} value={`${t.tps.toFixed(1)} tok/s`} />
        </div>
        {!a.hasTimeline && <p className="trace-note">{S.traces.noTimeline}</p>}
        {t.compactions > 0 && (
          <p className="trace-note">{S.traces.compactions(t.compactions)}</p>
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
  const S = useStrings();
  const [open, setOpen] = useState(ordinal === 1);
  const label = round.compaction ? S.traces.compaction : S.traces.round(ordinal);
  return (
    <section className={"trace-card round" + (round.compaction ? " compaction" : "")}>
      <button className="round-head" onClick={() => setOpen(!open)}>
        <span className={"round-caret" + (open ? " open" : "")}>▸</span>
        <span className="round-title">{label}</span>
        <span className="round-subject" title={round.title}>{round.title}</span>
        <span className="round-stats">
          <span title={S.traces.hRequests}>⇅ {round.requests}</span>
          <span title={S.traces.hToolCalls}>🔧 {round.toolCalls}{round.toolErrors ? ` (${S.traces.failed(round.toolErrors)})` : ""}</span>
          <span title={S.traces.hOutput}>↑ {fmtN(round.tokens.output)}</span>
          <span title={S.traces.hCost}>{money(round.costUsd)}</span>
          <span title={S.traces.hElapsed}>{ms(round.durationMs)}</span>
          <span title={S.traces.hTps}>{round.tps.toFixed(1)} tok/s</span>
        </span>
      </button>
      {open && (
        <div className="round-body">
          <Timeline round={round} analysis={analysis} />
          <EntryList entries={round.entries} />
        </div>
      )}
    </section>
  );
}

const TICKS = [0, 0.25, 0.5, 0.75, 1];

type ToolSpan = Analysis["toolSpans"][number];

/**
 * One lane per tool name, so a round reads as "what did `execute` do" rather
 * than as one row per call — twenty parallel `read_file`s used to push the
 * model lane off the top of the card.
 *
 * A name gets a second lane when two of its calls overlap in time, which is
 * exactly what a parallel batch does: drawn in one lane they would cover each
 * other and the picture would claim the batch was serial.
 */
function toolLanes(spans: ToolSpan[]): Array<{ key: string; name: string; spans: ToolSpan[] }> {
  const at = (ts?: string) => (ts ? Date.parse(ts) || 0 : 0);
  const byName = new Map<string, ToolSpan[]>();
  for (const sp of spans) {
    const name = sp.name || "tool";
    (byName.get(name) || byName.set(name, []).get(name)!).push(sp);
  }
  const out: Array<{ key: string; name: string; spans: ToolSpan[] }> = [];
  for (const [name, group] of byName) {
    const rows: Array<{ spans: ToolSpan[]; end: number }> = [];
    for (const sp of group) {
      const start = at(sp.callTs);
      const end = at(sp.outputTs) || at(sp.approvalTs) || start;
      const row = rows.find((r) => r.end <= start);
      if (row) {
        row.spans.push(sp);
        row.end = end;
      } else {
        rows.push({ spans: [sp], end });
      }
    }
    rows.forEach((row, i) => out.push({ key: `${name}#${i}`, name, spans: row.spans }));
  }
  return out;
}

/**
 * The round on its own clock: a serial model lane, then one lane per tool name,
 * all sharing the round's time axis.
 *
 * This replaced a stacked bar of per-phase totals. The totals answered "where
 * did the time go" and nothing else — two tools running side by side and the
 * same two run back to back drew the identical picture, which is the one
 * question worth asking of a trace. They are still on the round header as
 * numbers, where a number is all they were.
 */
function Timeline({ round, analysis }: { round: Round; analysis: Analysis }) {
  const S = useStrings();
  const segs = analysis.modelSegments.filter((s) => s.taskIndex === round.taskIndex);
  const spans = useMemo(
    () => analysis.toolSpans
      .filter((s) => s.taskIndex === round.taskIndex)
      .sort((a, b) => (Date.parse(a.callTs) || 0) - (Date.parse(b.callTs) || 0)),
    [analysis.toolSpans, round.taskIndex],
  );
  const lanes = useMemo(() => toolLanes(spans), [spans]);
  if (!segs.length && !spans.length) return null;

  // The axis spans everything drawn, not `startTs..endTs`: a tool whose output
  // lands after the round's last model event would otherwise run off the end
  // and be clamped to a flat edge at 100%.
  const stamps: number[] = [];
  const at = (ts?: string) => (ts ? Date.parse(ts) || 0 : 0);
  for (const s of segs) stamps.push(at(s.startTs), at(s.endTs));
  for (const sp of spans) stamps.push(at(sp.callTs), at(sp.approvalTs), at(sp.outputTs));
  const marks = stamps.filter((v) => v > 0);
  const t0 = Math.min(...marks, at(round.startTs) || Infinity);
  const t1 = Math.max(...marks, at(round.endTs) || 0);
  const span = Math.max(t1 - t0, 1);

  const pos = (from: string, to?: string) => {
    const a = at(from) || t0;
    const b = at(to) || a;
    const left = Math.max(0, Math.min(100, ((a - t0) / span) * 100));
    // A floor, so a sub-millisecond call is still a visible tick rather than a
    // lane that reads as "never ran".
    const width = Math.max(((b - a) / span) * 100, 0.6);
    return { left: `${left}%`, width: `${Math.min(width, 100 - left)}%` };
  };
  const phaseFor = (kind: string) =>
    kind === "thinking" ? "ph-thinking" : kind === "text" ? "ph-text" : "ph-args";

  return (
    <div className="lane-block">
      <div className="lane-title">{S.traces.timeline}<span className="lane-total">{ms(span)}</span></div>
      <div className="lane-row">
        <span className="lane-name">{S.traces.laneModel}</span>
        <div className="lane">
          {segs.map((s) => (
            <span key={s.key} className={"lane-seg " + phaseFor(s.kind)} style={pos(s.startTs, s.endTs)}
                  title={`${s.kind}${s.name ? " · " + s.name : ""} ${ms(at(s.endTs) - at(s.startTs))}`} />
          ))}
        </div>
      </div>
      {lanes.map((lane) => (
        <div className="lane-row" key={lane.key}>
          <span className="lane-name" title={lane.name}>
            {lane.name}
            {lane.spans.length > 1 && <span className="lane-count">×{lane.spans.length}</span>}
          </span>
          <div className="lane is-tool">
            {lane.spans.map((sp) => (
              <Fragment key={sp.toolCallId}>
                {sp.approvalTs && (
                  <span className="lane-seg ph-wait" style={pos(sp.callTs, sp.approvalTs)}
                        title={S.traces.approvalWait(ms(at(sp.approvalTs) - at(sp.callTs)))} />
                )}
                {sp.outputTs ? (
                  <span className="lane-seg ph-exec" style={pos(sp.approvalTs || sp.callTs, sp.outputTs)}
                        title={S.traces.exec(ms(at(sp.outputTs) - at(sp.approvalTs || sp.callTs)))} />
                ) : (
                  <span className="lane-seg ph-exec pending" style={pos(sp.callTs)}
                        title={S.traces.noResult} />
                )}
              </Fragment>
            ))}
          </div>
        </div>
      ))}
      <div className="lane-axis">
        <span className="lane-name" />
        <div className="lane-ticks">
          {TICKS.map((f) => (
            <span key={f} className="lane-tick" style={{ left: `${f * 100}%` }}>{ms(Math.round(f * span))}</span>
          ))}
        </div>
      </div>
      {/* Colours only, and all six of them every time. The per-phase totals
          used to be printed here and are a trap next to a wall-clock axis: they
          are sums across lanes, so a round with four parallel tools reads "tool
          execution 9.4s" under an axis that ends at 6.8s. Each bar carries its
          own duration on hover, and the round header carries the totals. The
          list is not filtered to the phases present either — a legend that
          changes shape per round is one the reader has to re-read, and "no
          approval wait in this round" is worth seeing. */}
      <div className="phase-legend">
        {PHASES.map((p) => (
          <span key={p.key} className="phase-key">
            <i className={"phase-dot " + p.cls} />{p.label(S)}
          </span>
        ))}
      </div>
    </div>
  );
}

function EntryList({ entries }: { entries: Entry[] }) {
  const S = useStrings();
  const [openAll, setOpenAll] = useState(false);
  return (
    <div className="entry-block">
      <div className="entry-head">
        <span>{S.traces.summary(entries.length)}</span>
        <button className="entry-expand" onClick={() => setOpenAll(!openAll)}>
          {openAll ? S.traces.collapseAll : S.traces.expandAll}
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
          <button className="entry-copy" onClick={() => { void navigator.clipboard.writeText(e.detail); showToast(getStrings().common.copied); }}>
            <IconCopy />
          </button>
          <pre>{e.detail}</pre>
        </div>
      )}
    </div>
  );
}
