import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router";
import * as api from "../api";
import type { Automation, AutomationRun, AutomationTarget } from "../api";
import { loadCronJobs } from "../data";
import { getStrings, useStrings, type Strings } from "../i18n";
import {
  IconClock, IconClose, IconMore, IconPause, IconPlay, IconPlus, IconSearch, IconSparkle,
  IconSpinner, IconTrash, IconChevronDown, IconChat,
} from "../icons";
import {
  FREQUENCIES, defaultDraft, draftFromSpec, relativeParts, scheduleFromDraft, systemZone,
  type Frequency, type TimingDraft,
} from "../lib/automationTiming";
import { loadSessions, switchTo } from "../sessions";
import { askConfirm, getState, normDir, setState, showToast, useAppState, workspaceName } from "../store";

/**
 * Automations, after deepseek-harness's task manager: a filtered, searchable
 * list on the left, and the selected task's detail on the right — its rule
 * (name, instruction, timing, where runs land) and its run records.
 *
 * Every run of an automation is a turn in a conversation, so the detail links
 * to that conversation and each record opens the conversation it ran in.
 */

type Filter = "all" | "active" | "paused" | "ended";
type Selection = { kind: "task"; id: string } | { kind: "create" } | null;

/** Form state shared by the create view and a task's Rule tab. */
type RuleDraft = {
  name: string;
  prompt: string;
  timing: TimingDraft;
  target: AutomationTarget;
  workDir: string;
};

function statusOf(j: Automation): Exclude<Filter, "all"> {
  if (j.status === "active") return "active";
  if (j.status === "paused") return "paused";
  return "ended";
}

function draftOf(j: Automation): RuleDraft {
  return {
    name: j.name, prompt: j.prompt,
    timing: draftFromSpec(j.schedule_spec),
    target: j.target, workDir: j.work_dir || "",
  };
}

function sameDraft(a: RuleDraft, b: RuleDraft) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function locale(lang: string) {
  return lang === "zh" ? "zh-CN" : "en-US";
}

/** "Nov 12 09:00 (in 3h)", the next-run line dsh shows. */
function fmtNext(ms: number, now: number, lang: string, S: Strings) {
  const abs = new Date(ms).toLocaleString(locale(lang), {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
  const r = relativeParts(ms, now);
  return S.automation.withRelative(abs, S.automation.relative(r.value, r.unit, r.future));
}

/** The repeat rule as one short phrase, from the draft the editor would open. */
function frequencyText(j: Automation, S: Strings): string {
  const A = S.automation;
  const d = draftFromSpec(j.schedule_spec);
  switch (d.frequency) {
    case "once": return A.once;
    case "minutely": return A.everyMinutesN(d.interval);
    case "hourly": return A.everyHoursN(d.interval);
    case "daily": return `${A.daily} ${d.time}`;
    case "weekdays": return `${A.weekdays} ${d.time}`;
    case "weekly": return `${A.weekly} ${d.weekdays.map((w) => A.weekdayNames[w]).join("、")} ${d.time}`;
    case "monthly": return A.monthlyOn(d.monthDay, d.time);
    default: return j.schedule;
  }
}

export function AutomationsPage() {
  const s = useAppState();
  const S = useStrings();
  const A = S.automation;
  const nav = useNavigate();
  const [params, setParams] = useSearchParams();
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [scheduler, setScheduler] = useState<{ enabled: boolean } | null>(null);
  const [moreOpen, setMoreOpen] = useState(false);
  const [now, setNow] = useState(Date.now());

  const selectedId = params.get("task");
  const creating = params.get("new") === "1";
  const selection: Selection = creating ? { kind: "create" } : selectedId ? { kind: "task", id: selectedId } : null;
  const select = (next: Selection) => {
    const p = new URLSearchParams();
    if (next?.kind === "task") p.set("task", next.id);
    if (next?.kind === "create") p.set("new", "1");
    setParams(p, { replace: true });
  };

  const refresh = async () => {
    const [jobs, st] = await Promise.all([api.fetchCronJobs(), api.fetchSchedulerStatus()]);
    setFailed(!jobs.ok);
    if (jobs.ok && jobs.data) setState({ cronJobs: jobs.data.jobs });
    if (st.ok && st.data) setScheduler({ enabled: st.data.enabled });
    setLoading(false);
  };

  useEffect(() => {
    void refresh();
    // Runs start and finish on the gateway's clock; keep "running" and the
    // next-run line current without a manual refresh.
    const poll = window.setInterval(() => { if (document.visibilityState === "visible") void refresh(); }, 5000);
    const tick = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => { window.clearInterval(poll); window.clearInterval(tick); };
  }, []);

  const jobs = s.cronJobs;
  const counts = useMemo(() => ({
    all: jobs.length,
    active: jobs.filter((j) => statusOf(j) === "active").length,
    paused: jobs.filter((j) => statusOf(j) === "paused").length,
    ended: jobs.filter((j) => statusOf(j) === "ended").length,
  }), [jobs]);
  const q = query.trim().toLowerCase();
  const shown = jobs
    .filter((j) => filter === "all" || statusOf(j) === filter)
    .filter((j) => !q || `${j.name} ${j.prompt}`.toLowerCase().includes(q))
    .sort((a, b) => {
      const order = { active: 0, paused: 1, ended: 2 };
      const d = order[statusOf(a)] - order[statusOf(b)];
      if (d) return d;
      return (a.next_run_at_ms || Infinity) - (b.next_run_at_ms || Infinity) || b.created_at_ms - a.created_at_ms;
    });
  const selected = selection?.kind === "task" ? jobs.find((j) => j.id === selection.id) || null : null;

  const toggleScheduler = async (enabled: boolean) => {
    const { ok, data } = await api.setSchedulerEnabledApi(enabled);
    if (!ok || !data) { showToast(A.schedulerToggleFailed, 2500); return; }
    setScheduler({ enabled: data.enabled });
  };

  const clearEnded = () => {
    setMoreOpen(false);
    if (!counts.ended) { showToast(A.clearedNone); return; }
    askConfirm({
      title: A.clearEnded, msg: A.clearEndedMsg(counts.ended), okLabel: S.common.delete,
      onOk: async () => {
        const { ok, data } = await api.clearEndedCronJobsApi();
        if (!ok) { showToast(A.actionFailed, 2500); return; }
        showToast(A.cleared(data?.deleted || 0));
        if (selected && statusOf(selected) === "ended") select(null);
        await refresh();
      },
    });
  };

  const emptyText = q ? A.noMatches
    : filter === "active" ? A.emptyActive : filter === "paused" ? A.emptyPaused
      : filter === "ended" ? A.emptyEnded : A.empty;

  return (
    <div className="main auto-page">
      <div className="auto-list-pane">
        <div className="auto-scroll">
          <div className="auto-content">
            <div className="auto-heading">
              <h1>{A.title}</h1>
              <div className="auto-heading-actions">
                <button type="button" className="auto-cta" onClick={() => select({ kind: "create" })}>
                  <IconPlus /><span>{A.newAction}</span>
                </button>
              </div>
            </div>

            {scheduler && !scheduler.enabled && (
              <div className="auto-banner">
                <IconClock /><span>{A.schedulerOff}</span>
                <button type="button" className="auto-link" onClick={() => void toggleScheduler(true)}>{A.schedulerTurnOn}</button>
              </div>
            )}

            <div className="auto-filters" role="tablist" aria-label={A.title}>
              {(["all", "active", "paused", "ended"] as const).map((f) => (
                <button key={f} type="button" role="tab" aria-selected={filter === f}
                        className={"auto-filter" + (filter === f ? " active" : "")} onClick={() => setFilter(f)}>
                  {{ all: A.filterAll, active: A.filterActive, paused: A.filterPaused, ended: A.filterEnded }[f]}
                  {counts[f] ? <span className="auto-filter-n">{counts[f]}</span> : null}
                </button>
              ))}
            </div>

            <div className="auto-search-tools">
              <label className="auto-search">
                <IconSearch />
                <input type="search" placeholder={A.searchPlaceholder} value={query}
                       onChange={(e) => setQuery(e.target.value)}
                       onKeyDown={(e) => { if (e.key === "Escape") setQuery(""); }} />
                {query && (
                  <button type="button" className="auto-search-clear" aria-label={A.clearSearch} onClick={() => setQuery("")}>
                    <IconClose />
                  </button>
                )}
              </label>
              <div className="auto-more-wrap">
                <button type="button" className={"ib auto-more" + (moreOpen ? " on" : "")} aria-label={A.more}
                        aria-expanded={moreOpen} onClick={() => setMoreOpen((v) => !v)}>
                  <IconMore />
                </button>
                {moreOpen && (
                  <>
                    <div className="auto-menu-scrim" onClick={() => setMoreOpen(false)} />
                    <div className="auto-menu" role="menu">
                      {scheduler && (
                        <button type="button" role="menuitem" onClick={() => { setMoreOpen(false); void toggleScheduler(!scheduler.enabled); }}>
                          {scheduler.enabled ? <IconPause /> : <IconPlay />}
                          <span>{scheduler.enabled ? `${A.schedulerOn} · ${S.common.pause}` : A.schedulerTurnOn}</span>
                        </button>
                      )}
                      <button type="button" role="menuitem" className="danger" onClick={clearEnded}>
                        <IconTrash /><span>{A.clearEnded}</span>
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>

            {loading ? (
              <div className="auto-empty"><IconSpinner /><p>{A.loading}</p></div>
            ) : failed && !jobs.length ? (
              <div className="auto-empty">
                <p>{A.loadError}</p>
                <button type="button" className="auto-btn" onClick={() => void refresh()}>{A.retry}</button>
              </div>
            ) : !shown.length ? (
              <div className="auto-empty">
                <EmptyArt />
                <p>{emptyText}</p>
                {!q && filter === "all" && (
                  <button type="button" className="auto-btn" onClick={() => select({ kind: "create" })}>{A.emptyAction}</button>
                )}
              </div>
            ) : (
              <ul className="auto-rows">
                {shown.map((j) => {
                  const st = statusOf(j);
                  return (
                    <li key={j.id}>
                      <button type="button"
                              className={"auto-row" + (selection?.kind === "task" && selection.id === j.id ? " selected" : "")}
                              onClick={() => select({ kind: "task", id: j.id })}>
                        <span className={"auto-row-glyph " + st}>
                          {j.running ? <IconSpinner /> : <IconClock />}
                        </span>
                        <span className="auto-row-main">
                          <span className="auto-row-title">{j.name}</span>
                          <span className="auto-row-summary">
                            {j.running ? <span className="auto-tag running">{A.running}</span>
                              : st !== "active" ? <span className={"auto-tag " + st}>{st === "paused" ? A.statusPaused : A.statusEnded}</span>
                                : null}
                            <span>{frequencyText(j, S)}</span>
                            {st === "active" && j.next_run_at_ms > 0 && (
                              <span className="auto-row-next">
                                {A.nextPrefix}{fmtNext(j.next_run_at_ms, now, s.lang, S)}
                              </span>
                            )}
                          </span>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      </div>

      {selection?.kind === "create" && (
        <CreatePane onClose={() => select(null)} onCreated={async (job, run) => {
          await refresh();
          if (run?.status === "queued" && run.session_id) {
            await openConversation(run.session_id, nav);
            return;
          }
          select({ kind: "task", id: job.id });
        }} />
      )}
      {selection?.kind === "task" && (
        selected
          ? <DetailPane key={selected.id} job={selected} now={now} onClose={() => select(null)} onChanged={refresh}
                        onDeleted={async () => { select(null); await refresh(); }} />
          : !loading && (
            <aside className="auto-detail">
              <div className="auto-detail-bar"><span /><button type="button" className="ib" onClick={() => select(null)}><IconClose /></button></div>
              <div className="auto-empty"><p>{A.automationGone}</p></div>
            </aside>
          )
      )}
    </div>
  );
}

/** Make sure the sidebar knows the conversation, open it, and go to chat. */
async function openConversation(sessionId: string, nav: (to: string) => void) {
  if (!getState().sessions[sessionId]) await loadSessions();
  if (getState().sessions[sessionId]) switchTo(sessionId);
  else showToast(getStrings().automation.sessionMissing, 2500);
  nav("/chat");
}

// ---------------------------------------------------------------------------
// Detail

function DetailPane({ job, now, onClose, onChanged, onDeleted }: {
  job: Automation;
  now: number;
  onClose: () => void;
  onChanged: () => Promise<void>;
  onDeleted: () => Promise<void>;
}) {
  const S = useStrings();
  const A = S.automation;
  const s = useAppState();
  const nav = useNavigate();
  const [tab, setTab] = useState<"rule" | "records">("rule");
  const [menu, setMenu] = useState(false);
  const [busy, setBusy] = useState("");
  const saved = useMemo(() => draftOf(job), [job]);
  const [draft, setDraft] = useState<RuleDraft>(saved);
  const [error, setError] = useState("");
  const savedKey = JSON.stringify(saved);
  const lastSaved = useRef(savedKey);
  // A poll that brings back the same job must not throw away edits; a save
  // (or a change made elsewhere) with nothing pending here is taken in.
  useEffect(() => {
    if (lastSaved.current === savedKey) return;
    const prev = JSON.parse(lastSaved.current) as RuleDraft;
    lastSaved.current = savedKey;
    setDraft((d) => (sameDraft(d, prev) ? saved : d));
  }, [savedKey, saved]);

  const dirty = !sameDraft(draft, saved);
  const st = statusOf(job);

  const save = async () => {
    const issue = validate(draft);
    if (issue) { setError(A.errors[issue] || A.errors.invalidRule); return; }
    const timing = scheduleFromDraft(draft.timing, Date.now());
    if ("error" in timing) { setError(A.errors[timing.error]); return; }
    setBusy("save");
    const body: api.AutomationWrite = {};
    if (draft.name.trim() !== saved.name) body.name = draft.name.trim();
    if (draft.prompt.trim() !== saved.prompt) body.prompt = draft.prompt.trim();
    if (JSON.stringify(draft.timing) !== JSON.stringify(saved.timing)) {
      body.schedule = timing.schedule;
      body.timezone = timing.timezone;
    }
    if (draft.target !== saved.target) body.target = draft.target;
    if (draft.workDir !== saved.workDir) body.work_dir = draft.workDir;
    const { ok, data } = await api.updateCronJobApi(job.id, body);
    setBusy("");
    if (!ok) { setError((data as { detail?: string } | null)?.detail || A.actionFailed); return; }
    setError("");
    showToast(A.saved);
    await onChanged();
  };

  const act = async (kind: "pause" | "resume" | "run") => {
    setMenu(false);
    setBusy(kind);
    if (kind === "run") {
      const { ok, status, data } = await api.triggerCronJobApi(job.id);
      setBusy("");
      if (status === 409) { showToast(A.runBusy, 2500); return; }
      if (!ok || !data) { showToast((data as { detail?: string } | null)?.detail || A.actionFailed, 2500); return; }
      await onChanged();
      if (data.run.session_id) {
        showToast(A.runQueued);
        await openConversation(data.run.session_id, nav);
      }
      return;
    }
    const res = kind === "pause" ? await api.pauseCronJobApi(job.id) : await api.resumeCronJobApi(job.id);
    setBusy("");
    if (!res.ok) { showToast((res.data as { detail?: string } | null)?.detail || A.actionFailed, 2500); return; }
    await onChanged();
  };

  const remove = () => {
    setMenu(false);
    askConfirm({
      title: A.deleteTitle, msg: A.deleteMsg, okLabel: S.common.delete,
      onOk: async () => {
        const { ok } = await api.deleteCronJobApi(job.id);
        if (!ok) { showToast(A.actionFailed, 2500); return; }
        showToast(A.deleted);
        await onDeleted();
      },
    });
  };

  const session = job.session_id ? s.sessions[job.session_id] : undefined;

  return (
    <aside className="auto-detail" aria-label={job.name}>
      <div className="auto-detail-bar">
        <div className="auto-tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === "rule"} className="auto-tab" onClick={() => setTab("rule")}>
            {A.ruleTitle}
          </button>
          <button type="button" role="tab" aria-selected={tab === "records"} className="auto-tab" onClick={() => setTab("records")}>
            {A.records}{job.run_count ? <span className="auto-tab-n">{job.run_count}</span> : null}
          </button>
        </div>
        <div className="auto-detail-actions">
          {st !== "ended" && (
            <button type="button" className="ib" title={A.runNow} aria-label={A.runNow}
                    disabled={!!busy || job.running} onClick={() => void act("run")}>
              {busy === "run" || job.running ? <IconSpinner /> : <IconPlay />}
            </button>
          )}
          <div className="auto-more-wrap">
            <button type="button" className={"ib" + (menu ? " on" : "")} aria-label={A.more} aria-expanded={menu}
                    onClick={() => setMenu((v) => !v)}>
              <IconMore />
            </button>
            {menu && (
              <>
                <div className="auto-menu-scrim" onClick={() => setMenu(false)} />
                <div className="auto-menu" role="menu">
                  {st === "active" && (
                    <button type="button" role="menuitem" onClick={() => void act("pause")}><IconPause /><span>{A.pause}</span></button>
                  )}
                  {st !== "active" && (
                    <button type="button" role="menuitem" onClick={() => void act("resume")}><IconPlay /><span>{A.resume}</span></button>
                  )}
                  <button type="button" role="menuitem" className="danger" onClick={remove}><IconTrash /><span>{A.delete}</span></button>
                </div>
              </>
            )}
          </div>
          <button type="button" className="ib" aria-label={A.close} title={A.close} onClick={onClose}><IconClose /></button>
        </div>
      </div>

      {tab === "rule" ? (
        <>
          <div className="auto-detail-scroll">
            <div className="auto-detail-head">
              <span className={"auto-status " + (job.running ? "running" : st)}>
                {job.running ? A.running : st === "active" ? A.statusActive : st === "paused" ? A.statusPaused : A.statusEnded}
              </span>
              {st === "active" && job.next_run_at_ms > 0 && (
                <span className="auto-next">
                  {A.nextRun} {fmtNext(job.next_run_at_ms, now, s.lang, S)}
                </span>
              )}
            </div>
            {job.paused_reason && <div className="auto-banner warn">{A.pausedReason(job.paused_reason)}</div>}
            <RuleEditor draft={draft} onChange={(d) => { setDraft(d); setError(""); }} />
          </div>
          {dirty ? (
            <div className="auto-detail-foot">
              <span className={"auto-foot-msg" + (error ? " err" : "")}>{error || A.unsaved}</span>
              <div className="auto-foot-actions">
                <button type="button" className="auto-btn" onClick={() => { setDraft(saved); setError(""); }}>{A.discard}</button>
                <button type="button" className="auto-btn primary" disabled={busy === "save"} onClick={() => void save()}>
                  {busy === "save" ? A.saving : A.save}
                </button>
              </div>
            </div>
          ) : (
            <div className="auto-detail-foot session">
              <span className="auto-foot-label">{A.session}</span>
              {job.session_id ? (
                <button type="button" className="auto-session-link" title={A.openSession}
                        onClick={() => void openConversation(job.session_id, nav)}>
                  <IconChat /><span>{session?.title || job.session_id}</span>
                </button>
              ) : (
                <span className="auto-foot-muted">{job.target === "new" ? A.targetNewHelp : A.sessionPending}</span>
              )}
            </div>
          )}
        </>
      ) : (
        <RunRecords job={job} />
      )}
    </aside>
  );
}

function validate(d: RuleDraft): string {
  const name = d.name.trim();
  if (!name || name.length > 120) return "invalidTitle";
  if (!d.prompt.trim()) return "instructionRequired";
  return "";
}

// ---------------------------------------------------------------------------
// Rule editor: name, instruction, then the rule rows box (dsh's `.ruleRows`)

function RuleEditor({ draft, onChange, creating }: {
  draft: RuleDraft;
  onChange: (d: RuleDraft) => void;
  creating?: boolean;
}) {
  const S = useStrings();
  const A = S.automation;
  const s = useAppState();
  const [polishing, setPolishing] = useState(false);
  const t = draft.timing;
  const setT = (p: Partial<TimingDraft>) => onChange({ ...draft, timing: { ...t, ...p } });

  const polish = async () => {
    if (!draft.prompt.trim() || polishing) return;
    setPolishing(true);
    const { ok, data } = await api.polishPromptApi(draft.prompt.trim());
    setPolishing(false);
    if (!ok || !data?.prompt) { showToast(A.polishFailed, 2500); return; }
    onChange({ ...draft, prompt: data.prompt });
  };

  const freqLabel: Record<Frequency, string> = {
    once: A.once, minutely: A.everyMinutes, hourly: A.everyHours, daily: A.daily,
    weekdays: A.weekdays, weekly: A.weekly, monthly: A.monthly, custom: A.custom,
  };
  const dirs = useMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const d of [draft.workDir, s.serverDir, ...s.workspaces.map((w) => w.dir),
      ...Object.values(s.sessions).map((x) => x.dir)]) {
      const k = normDir(d || "");
      if (k && !seen.has(k)) { seen.add(k); out.push(k); }
    }
    return out;
  }, [draft.workDir, s.serverDir, s.workspaces, s.sessions]);

  const targetOptions: AutomationTarget[] = draft.target === "session" ? ["session", "new_once", "new"] : ["new_once", "new"];

  return (
    <div className="auto-editor">
      <label className="auto-field">
        <span className="auto-field-label">{A.nameLabel}</span>
        <input className="auto-input" value={draft.name} maxLength={120} placeholder={A.namePlaceholder}
               onChange={(e) => onChange({ ...draft, name: e.target.value })} />
      </label>
      <label className="auto-field">
        <span className="auto-field-label">
          {A.instruction}
          <button type="button" className={"auto-polish" + (polishing ? " spinning" : "")} title={A.polish}
                  aria-label={A.polish} disabled={!draft.prompt.trim() || polishing}
                  onClick={(e) => { e.preventDefault(); void polish(); }}>
            {polishing ? <IconSpinner /> : <IconSparkle />}
          </button>
        </span>
        <textarea className="auto-input auto-instruction" rows={creating ? 6 : 5} value={draft.prompt}
                  placeholder={A.instructionPlaceholder}
                  onChange={(e) => onChange({ ...draft, prompt: e.target.value })} />
      </label>

      <div className="auto-field">
        <span className="auto-field-label">{A.ruleTitle}</span>
        <div className="auto-rule-rows">
          <RuleSelect label={A.repeat} value={t.frequency}
                      options={FREQUENCIES.map((f) => ({ value: f, label: freqLabel[f] }))}
                      onChange={(v) => setT({ frequency: v as Frequency })} />
          {(t.frequency === "minutely" || t.frequency === "hourly") && (
            <div className="auto-rule-row">
              <span className="auto-rule-label">{A.interval}</span>
              <span className="auto-rule-control">
                <input className="auto-stepper" type="number" min={1} max={t.frequency === "minutely" ? 1440 : 720} step={1}
                       value={t.interval}
                       onChange={(e) => {
                         const n = Number(e.target.value);
                         if (e.target.value !== "" && Number.isSafeInteger(n)) setT({ interval: n });
                       }} />
                <span className="auto-unit">{t.frequency === "minutely" ? A.unitMinutes : A.unitHours}</span>
              </span>
            </div>
          )}
          {t.frequency === "once" && (
            <div className="auto-rule-row">
              <span className="auto-rule-label">{A.date}</span>
              <input className="auto-rule-input" type="date" value={t.date} onChange={(e) => setT({ date: e.target.value })} />
            </div>
          )}
          {t.frequency === "weekly" && (
            <div className="auto-rule-row">
              <span className="auto-rule-label">{A.weekday}</span>
              <span className="auto-pills" role="group" aria-label={A.weekday}>
                {[1, 2, 3, 4, 5, 6, 0].map((w) => {
                  const on = t.weekdays.includes(w);
                  return (
                    <button key={w} type="button" aria-pressed={on} className={"auto-pill" + (on ? " on" : "")}
                            onClick={() => setT({ weekdays: on ? t.weekdays.filter((x) => x !== w) : [...t.weekdays, w].sort() })}>
                      {A.weekdayNames[w]}
                    </button>
                  );
                })}
              </span>
            </div>
          )}
          {t.frequency === "monthly" && (
            <div className="auto-rule-row">
              <span className="auto-rule-label">{A.monthDay}</span>
              <input className="auto-stepper" type="number" min={1} max={31} step={1} value={t.monthDay}
                     onChange={(e) => {
                       const n = Number(e.target.value);
                       if (e.target.value !== "" && Number.isSafeInteger(n)) setT({ monthDay: Math.min(31, Math.max(1, n)) });
                     }} />
            </div>
          )}
          {["once", "daily", "weekdays", "weekly", "monthly"].includes(t.frequency) && (
            <div className="auto-rule-row">
              <span className="auto-rule-label">{A.time}</span>
              <input className="auto-rule-input" type="time" value={t.time} onChange={(e) => setT({ time: e.target.value })} />
            </div>
          )}
          {t.frequency === "custom" && (
            <div className="auto-rule-row stacked">
              <span className="auto-rule-label">{A.customRule}</span>
              <input className="auto-input mono" value={t.custom} placeholder="0 9 * * 1-5"
                     onChange={(e) => setT({ custom: e.target.value })} />
              <span className="auto-rule-hint">{A.customHint}</span>
            </div>
          )}
          {t.frequency !== "minutely" && t.frequency !== "hourly" && (
            <RuleSelect label={A.zone} value={t.timezone || systemZone()}
                        options={zoneOptions(t.timezone, A.zoneSystem)}
                        onChange={(v) => setT({ timezone: v })} />
          )}
        </div>
      </div>

      <div className="auto-field">
        <span className="auto-field-label">{A.target}</span>
        <div className="auto-rule-rows">
          <RuleSelect label={A.target} value={draft.target}
                      options={targetOptions.map((v) => ({
                        value: v,
                        label: v === "session" ? A.targetSession : v === "new_once" ? A.targetNewOnce : A.targetNew,
                        help: v === "new_once" ? A.targetNewOnceHelp : v === "new" ? A.targetNewHelp : undefined,
                      }))}
                      onChange={(v) => onChange({ ...draft, target: v as AutomationTarget })} />
          {draft.target !== "session" && (
            <RuleSelect label={A.workspace} value={draft.workDir || s.serverDir}
                        options={dirs.map((d) => ({ value: d, label: workspaceName(d), help: d }))}
                        onChange={(v) => onChange({ ...draft, workDir: v })} />
          )}
        </div>
      </div>
    </div>
  );
}

function zoneOptions(current: string, systemLabel: (z: string) => string) {
  const sys = systemZone();
  const common = ["Asia/Shanghai", "Asia/Tokyo", "Asia/Singapore", "Europe/London", "Europe/Berlin",
    "America/New_York", "America/Los_Angeles", "UTC"];
  let all: string[] = common;
  try {
    const intl = Intl as unknown as { supportedValuesOf?: (k: string) => string[] };
    if (intl.supportedValuesOf) all = intl.supportedValuesOf("timeZone");
  } catch { /* older engines: the common list */ }
  const head = [sys, current].filter(Boolean) as string[];
  return [...new Set([...head, ...common, ...all])].map((z) => ({ value: z, label: z === sys ? systemLabel(z) : z }));
}

/** dsh's `RuleSelectRow`: label on the left, the value as a quiet pill with a chevron, a menu below. */
function RuleSelect({ label, value, options, onChange }: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string; help?: string }>;
  onChange: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const current = options.find((o) => o.value === value);
  const searchable = options.length > 12;
  const shown = searchable && filter
    ? options.filter((o) => `${o.value} ${o.label}`.toLowerCase().includes(filter.toLowerCase()))
    : options;
  return (
    <div className="auto-rule-row menu-row">
      <button type="button" className="auto-rule-value" aria-haspopup="menu" aria-expanded={open}
              onClick={() => setOpen((v) => !v)}>
        <span className="auto-rule-label">{label}</span>
        <span className="auto-rule-face"><span className="auto-rule-current">{current?.label || value}</span><IconChevronDown /></span>
      </button>
      {open && (
        <>
          <div className="auto-menu-scrim" onClick={() => { setOpen(false); setFilter(""); }} />
          <div className={"auto-menu rule" + (searchable ? " tall" : "")} role="menu">
            {searchable && (
              <input className="auto-menu-search" autoFocus type="search" value={filter}
                     onChange={(e) => setFilter(e.target.value)} placeholder="UTC / Asia/Shanghai" />
            )}
            <div className="auto-menu-list">
              {shown.map((o) => (
                <button key={o.value} type="button" role="menuitemradio" aria-checked={o.value === value}
                        className={o.value === value ? "on" : ""}
                        onClick={() => { setOpen(false); setFilter(""); onChange(o.value); }}>
                  <span className="auto-menu-item">
                    <span>{o.label}</span>
                    {o.help && <span className="auto-menu-help">{o.help}</span>}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Run records

function RunRecords({ job }: { job: Automation }) {
  const S = useStrings();
  const A = S.automation;
  const s = useAppState();
  const nav = useNavigate();
  const [runs, setRuns] = useState<AutomationRun[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState<Record<number, boolean>>({});

  const load = async () => {
    const { ok, data } = await api.fetchCronRuns(job.id, 100);
    if (!ok || !data) { setFailed(true); return; }
    setFailed(false);
    setRuns(data.runs);
  };
  // Reload when a run finishes (the job's run count moves).
  useEffect(() => { void load(); }, [job.id, job.run_count, job.running]);

  if (failed) {
    return (
      <div className="auto-empty">
        <p>{A.recordsError}</p>
        <button type="button" className="auto-btn" onClick={() => void load()}>{A.retry}</button>
      </div>
    );
  }
  if (!runs) return <div className="auto-empty"><IconSpinner /><p>{A.recordsLoading}</p></div>;
  if (!runs.length && !job.running) return <div className="auto-empty"><p>{A.noRecords}</p></div>;
  return (
    <div className="auto-detail-scroll records">
      {job.running && (
        <div className="auto-record running">
          <div className="auto-record-head">
            <span className="auto-record-status running"><IconSpinner />{A.running}</span>
          </div>
        </div>
      )}
      {runs.map((r) => {
        const body = r.result_full || r.error || "";
        const long = body.length > 280;
        const expanded = !!open[r.started_at_ms];
        const status = r.status === "ok" ? "ok" : r.status === "timeout" ? "timeout" : "failed";
        return (
          <section key={`${r.started_at_ms}-${r.status}`} className="auto-record">
            <div className="auto-record-head">
              <span className={"auto-record-status " + status}>
                {status === "ok" ? A.runOk : status === "timeout" ? A.runTimeout : A.runFailed}
              </span>
              <span className="auto-record-time">{new Date(r.started_at_ms).toLocaleString(locale(s.lang), { hour12: false })}</span>
              {r.duration_ms > 0 && <span className="auto-record-time">· {(r.duration_ms / 1000).toFixed(1)}s</span>}
              {r.session_id && (
                <button type="button" className="auto-link" onClick={() => void openConversation(r.session_id, nav)}>
                  {A.viewInSession}
                </button>
              )}
            </div>
            {body && (
              <div className={"auto-record-body" + (expanded || !long ? "" : " clamped") + (status !== "ok" ? " err" : "")}>
                {body}
              </div>
            )}
            {long && (
              <button type="button" className="auto-link" onClick={() => setOpen({ ...open, [r.started_at_ms]: !expanded })}>
                {expanded ? A.collapse : A.expand}
              </button>
            )}
          </section>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Create

function CreatePane({ onClose, onCreated }: {
  onClose: () => void;
  onCreated: (job: Automation, run?: api.AutomationRunStart) => Promise<void>;
}) {
  const S = useStrings();
  const A = S.automation;
  const s = useAppState();
  const [params] = useSearchParams();
  // "Create from a chat" passes the conversation: the job then runs there.
  const fromSession = params.get("session") || "";
  const fromSess = fromSession ? s.sessions[fromSession] : undefined;
  const [draft, setDraft] = useState<RuleDraft>(() => ({
    name: "", prompt: "", timing: { ...defaultDraft(), frequency: "daily" as Frequency },
    target: fromSession ? "session" : "new_once",
    workDir: fromSess?.dir || getState().serverDir,
  }));
  const [runNow, setRunNow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const create = async () => {
    const issue = validate(draft);
    if (issue) { setError(A.errors[issue]); return; }
    const timing = scheduleFromDraft(draft.timing, Date.now());
    if ("error" in timing) { setError(A.errors[timing.error]); return; }
    setBusy(true);
    const { ok, data } = await api.createCronJobApi({
      name: draft.name.trim(), prompt: draft.prompt.trim(),
      schedule: timing.schedule, timezone: timing.timezone,
      target: draft.target,
      // The page may have opened before the gateway's default directory was
      // known; the default is the same one the picker shows.
      ...(draft.target === "session" ? { session_id: fromSession } : { work_dir: draft.workDir || s.serverDir }),
      validate_run: runNow,
    });
    setBusy(false);
    if (!ok || !data?.job) { setError(data?.detail || A.actionFailed); return; }
    showToast(A.created);
    await onCreated(data.job, data.validation_run);
  };

  return (
    <aside className="auto-detail" aria-label={A.emptyAction}>
      <div className="auto-detail-bar">
        <h2 className="auto-create-title">{A.emptyAction}</h2>
        <div className="auto-detail-actions">
          <button type="button" className="ib" aria-label={A.close} title={A.close} onClick={onClose}><IconClose /></button>
        </div>
      </div>
      <div className="auto-detail-scroll">
        {fromSession && (
          <div className="auto-banner">
            <IconChat /><span>{A.targetSession}：{fromSess?.title || fromSession}</span>
          </div>
        )}
        <RuleEditor draft={draft} creating onChange={(d) => { setDraft(d); setError(""); }} />
        <label className="auto-check">
          <input type="checkbox" checked={runNow} onChange={(e) => setRunNow(e.target.checked)} />
          <span>{A.createAndRun}</span>
        </label>
      </div>
      <div className="auto-detail-foot">
        <span className={"auto-foot-msg" + (error ? " err" : "")}>{error}</span>
        <div className="auto-foot-actions">
          <button type="button" className="auto-btn" onClick={onClose}>{S.common.cancel}</button>
          <button type="button" className="auto-btn primary" disabled={busy} onClick={() => void create()}>
            {busy ? A.creating : A.create}
          </button>
        </div>
      </div>
    </aside>
  );
}

/** dsh's document-and-clock empty artwork, on this theme's greys. */
function EmptyArt(): ReactNode {
  return (
    <svg className="auto-empty-art" width="110" height="110" viewBox="0 0 170 170" fill="none" aria-hidden="true">
      <path d="M163.8 97.9c0 28.6-19.3 37.4-48 37.4-28.6 0-55.7-8.8-55.7-37.4 0-28.6 40.9-51.8 69.5-51.8 28.6 0 34.2 23.2 34.2 51.8z" fill="var(--bg3)" />
      <path d="M9.5 88.7c11.3 22.5 30.1 21.8 52.7 10.4 22.6-11.4 40.6-29 29.3-51.6-11.3-22.5-52.8-24.6-75.5-13.2-22.6 11.4-17.8 31.9-6.5 54.4z" fill="var(--bg3)" />
      <rect x="40" y="16.7" width="86.2" height="98.3" rx="8.4" fill="var(--bg)" stroke="var(--border)" />
      <path d="M72 45h38v5.4H72zM72 63.3h38v5.4H72zM72 81.6h19v5.3H72zM56 43.5h8v8h-8zM56 62h8v8h-8zM56 80.3h8v8h-8z" fill="var(--fg3)" opacity=".55" />
      <circle cx="124" cy="98.9" r="25" fill="var(--fg3)" opacity=".55" />
      <circle cx="124" cy="98.9" r="20.1" fill="var(--bg)" />
      <path d="M125.1 86.7v12.6l9.8 4.8-2.1 4.3-10.1-5c-1.4-.7-2.4-2.2-2.4-3.8V86.7z" fill="var(--fg3)" />
    </svg>
  );
}
