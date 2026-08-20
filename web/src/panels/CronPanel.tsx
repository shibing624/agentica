import * as api from "../api";
import { loadCronJobs } from "../data";
import { getStrings, useStrings } from "../i18n";
import { IconClose } from "../icons";
import {
  askConfirm, emptyCronForm, getState, setState, showToast, useAppState,
  type CronForm,
} from "../store";

function whenStr(msValue: number) {
  if (!msValue) return "—";
  const d = new Date(msValue);
  return d.toLocaleString();
}

function patch(p: Partial<CronForm>) {
  const f = getState().cronForm;
  if (f) setState({ cronForm: { ...f, ...p } });
}

function formFromJob(job: any): CronForm {
  return {
    id: job.id,
    name: job.name || "",
    prompt: job.prompt || "",
    // The list's `schedule` is human text and does not parse back.
    schedule: job.schedule_expr || "",
    timeout_seconds: job.timeout_seconds ? String(job.timeout_seconds) : "",
    max_retries: String(job.max_retries ?? 0),
    validate_run: false,
  };
}

async function save() {
  const f = getState().cronForm;
  if (!f) return;
  const S = getStrings();
  if (!f.prompt.trim() || !f.schedule.trim()) {
    showToast(S.cron.required, 2500);
    return;
  }
  const body: Record<string, unknown> = {
    prompt: f.prompt.trim(),
    schedule: f.schedule.trim(),
    name: f.name.trim() || undefined,
    timeout_seconds: parseFloat(f.timeout_seconds) || 0,
    max_retries: parseInt(f.max_retries, 10) || 0,
  };
  if (!f.id) body.validate_run = f.validate_run;
  const res = f.id
    ? await api.updateCronJobApi(f.id, body)
    : await api.createCronJobApi(body);
  if (!res.ok) { showToast((res.data as any)?.detail || S.common.saveFailed, 3000); return; }
  const validation = (res.data as any)?.validation_run;
  setState({ cronForm: null });
  await loadCronJobs();
  if (validation) {
    showToast(validation.status === "ok"
      ? S.cron.createdValidated
      : S.cron.createdValidationFailed(validation.error || validation.status), 4000);
  } else {
    showToast(f.id ? S.cron.jobUpdated : S.cron.jobCreated);
  }
}

async function polish() {
  const f = getState().cronForm;
  const S = getStrings();
  if (!f || !f.prompt.trim()) { showToast(S.cron.polishNeedsDraft); return; }
  setState({ cronBusy: "polish" });
  const { ok, data } = await api.polishPromptApi(f.prompt.trim());
  setState({ cronBusy: "" });
  if (!ok || !(data as any)?.prompt) { showToast((data as any)?.detail || S.cron.polishFailed, 3000); return; }
  patch({ prompt: (data as any).prompt });
  showToast(S.cron.polished);
}

async function act(id: string, kind: "pause" | "resume" | "trigger") {
  const S = getStrings();
  setState({ cronBusy: id + kind });
  const call = kind === "pause" ? api.pauseCronJobApi : kind === "resume" ? api.resumeCronJobApi : api.triggerCronJobApi;
  const { ok, data } = await call(id);
  setState({ cronBusy: "" });
  if (!ok) { showToast((data as any)?.detail || S.cron.actionFailed, 3000); return; }
  if (kind === "trigger") {
    const run = (data as any)?.run;
    showToast(run?.status === "ok" ? S.cron.runOk
      : S.cron.runFailed(run?.error || run?.status || "unknown"), 4000);
    await openRuns(id, true);
  } else {
    showToast(kind === "pause" ? S.cron.paused : S.cron.resumed);
  }
  await loadCronJobs();
}

function remove(job: any) {
  const S = getStrings();
  askConfirm({
    title: S.cron.removeJob,
    msg: S.cron.removeJobMsg(job.name || job.id),
    onOk: async () => {
      const { ok, data } = await api.deleteCronJobApi(job.id);
      if (!ok) { showToast((data as any)?.detail || S.common.deleteFailed, 3000); return; }
      await loadCronJobs();
      showToast(S.cron.jobDeleted);
    },
  });
}

async function openRuns(id: string, force = false) {
  const st = getState();
  const isOpen = st.cronRunsOpen.includes(id);
  if (isOpen && !force) {
    setState({ cronRunsOpen: st.cronRunsOpen.filter((x) => x !== id) });
    return;
  }
  const { ok, data } = await api.fetchCronRuns(id);
  setState({
    cronRuns: { ...getState().cronRuns, [id]: ok ? ((data as any)?.runs || []) : [] },
    cronRunsOpen: isOpen ? st.cronRunsOpen : [...st.cronRunsOpen, id],
  });
}

export function CronPanel() {
  const s = useAppState();
  const S = useStrings();
  return (
    <div className="cron-panel">
      <div className="panel-bar">
        <span className="panel-count">{S.cron.jobCount(s.cronJobs.length)}</span>
        <div className="panel-bar-actions">
          <button className="dp-btn" onClick={() => void loadCronJobs()}>{S.common.refresh}</button>
          <button className="dp-btn primary" onClick={() => setState({ cronForm: emptyCronForm() })}>{S.cron.newJob}</button>
        </div>
      </div>

      {s.cronForm && <CronFormView />}

      <div className="settings-list">
        {!s.cronJobs.length && <div className="settings-empty">{S.cron.noJobs}</div>}
        {s.cronJobs.map((j: any) => {
          const paused = j.status === "paused" || j.enabled === false;
          const runsOpen = s.cronRunsOpen.includes(j.id);
          return (
            <div className="cron-item" key={j.id}>
              <div className="cron-item-head">
                <span className="cron-name">{j.name || j.id}</span>
                <span className={"cron-status " + (paused ? "paused" : "active")}>{j.status}</span>
                <div className="cron-actions">
                  <button className="cron-act" disabled={!!s.cronBusy} onClick={() => void act(j.id, "trigger")}>
                    {s.cronBusy === j.id + "trigger" ? S.cron.running : S.cron.runNow}
                  </button>
                  <button className="cron-act" disabled={!!s.cronBusy} onClick={() => void act(j.id, paused ? "resume" : "pause")}>
                    {paused ? S.common.resume : S.common.pause}
                  </button>
                  <button className="cron-act" onClick={() => setState({ cronForm: formFromJob(j) })}>{S.common.edit}</button>
                  <button className="cron-act danger" onClick={() => remove(j)}>{S.common.delete}</button>
                </div>
              </div>
              <div className="cron-prompt" title={j.prompt}>{j.prompt}</div>
              <div className="cron-meta">
                <span>{S.cron.schedule(j.schedule)}</span>
                <span>{S.cron.next(whenStr(j.next_run_at_ms))}</span>
                <span>{S.cron.last(whenStr(j.last_run_at_ms))}{j.last_status ? ` (${j.last_status})` : ""}</span>
                <span>{S.cron.runCount(j.run_count || 0)}</span>
                {j.timeout_seconds ? <span>{S.cron.timeout(j.timeout_seconds)}</span> : null}
                {j.max_retries ? <span>{S.cron.retries(j.max_retries)}</span> : null}
              </div>
              <button className="cron-runs-toggle" onClick={() => void openRuns(j.id)}>
                {runsOpen ? "▾" : "▸"} {S.cron.history}
              </button>
              {runsOpen && (
                <div className="cron-runs">
                  {!(s.cronRuns[j.id] || []).length && <div className="cron-run-empty">{S.cron.noRuns}</div>}
                  {(s.cronRuns[j.id] || []).map((r: any, i: number) => (
                    <div className={"cron-run " + (r.status === "ok" ? "ok" : "bad")} key={i}>
                      <span className="cron-run-status">{r.status}</span>
                      <span className="cron-run-time">{whenStr(r.started_at_ms)}</span>
                      <span className="cron-run-text" title={r.result_full || r.error || ""}>
                        {r.error || r.result_preview || ""}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CronFormView() {
  const s = useAppState();
  const S = useStrings();
  const f = s.cronForm!;
  return (
    <div className="settings-form cron-form">
      <h4>{f.id ? S.cron.editJob : S.cron.newJobTitle}</h4>
      <input className="pf-input" placeholder={S.cron.namePlaceholder} value={f.name} onChange={(e) => patch({ name: e.target.value })} />
      <div className="cron-prompt-wrap">
        <textarea
          className="pf-input pf-textarea"
          rows={4}
          placeholder={S.cron.promptPlaceholder}
          value={f.prompt}
          onChange={(e) => patch({ prompt: e.target.value })}
        />
        <button className="cron-act" disabled={s.cronBusy === "polish"} onClick={() => void polish()}>
          {s.cronBusy === "polish" ? S.cron.polishing : S.cron.polish}
        </button>
      </div>
      <input
        className="pf-input"
        placeholder={S.cron.schedulePlaceholder}
        value={f.schedule}
        onChange={(e) => patch({ schedule: e.target.value })}
      />
      <div className="pf-row">
        <input className="pf-input" type="number" placeholder={S.cron.timeoutPlaceholder} value={f.timeout_seconds} onChange={(e) => patch({ timeout_seconds: e.target.value })} />
        <input className="pf-input" type="number" placeholder={S.cron.retriesPlaceholder} value={f.max_retries} onChange={(e) => patch({ max_retries: e.target.value })} />
      </div>
      {!f.id && (
        <label className="pf-check">
          <input type="checkbox" checked={f.validate_run} onChange={(e) => patch({ validate_run: e.target.checked })} />
          {S.cron.validateRun}
        </label>
      )}
      <div className="pf-actions">
        <button className="dp-btn" onClick={() => setState({ cronForm: null })}>{S.common.cancel}</button>
        <button className="dp-btn primary" onClick={() => void save()}>{f.id ? S.common.save : S.common.create}</button>
      </div>
      <button className="pf-close" onClick={() => setState({ cronForm: null })}><IconClose /></button>
    </div>
  );
}
