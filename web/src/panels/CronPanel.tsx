import * as api from "../api";
import { loadCronJobs } from "../data";
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
  if (!f.prompt.trim() || !f.schedule.trim()) {
    showToast("prompt 和 schedule 是必填项", 2500);
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
  if (!res.ok) { showToast((res.data as any)?.detail || "保存失败", 3000); return; }
  const validation = (res.data as any)?.validation_run;
  setState({ cronForm: null });
  await loadCronJobs();
  if (validation) {
    showToast(validation.status === "ok"
      ? "已创建，试跑成功"
      : `已创建，但试跑失败：${validation.error || validation.status}`, 4000);
  } else {
    showToast(f.id ? "任务已更新" : "任务已创建");
  }
}

async function polish() {
  const f = getState().cronForm;
  if (!f || !f.prompt.trim()) { showToast("先写点 prompt 草稿"); return; }
  setState({ cronBusy: "polish" });
  const { ok, data } = await api.polishPromptApi(f.prompt.trim());
  setState({ cronBusy: "" });
  if (!ok || !(data as any)?.prompt) { showToast((data as any)?.detail || "润色失败", 3000); return; }
  patch({ prompt: (data as any).prompt });
  showToast("prompt 已润色");
}

async function act(id: string, kind: "pause" | "resume" | "trigger") {
  setState({ cronBusy: id + kind });
  const call = kind === "pause" ? api.pauseCronJobApi : kind === "resume" ? api.resumeCronJobApi : api.triggerCronJobApi;
  const { ok, data } = await call(id);
  setState({ cronBusy: "" });
  if (!ok) { showToast((data as any)?.detail || "操作失败", 3000); return; }
  if (kind === "trigger") {
    const run = (data as any)?.run;
    showToast(run?.status === "ok" ? "运行成功" : `运行失败：${run?.error || run?.status || "unknown"}`, 4000);
    await openRuns(id, true);
  } else {
    showToast(kind === "pause" ? "已暂停" : "已恢复");
  }
  await loadCronJobs();
}

function remove(job: any) {
  askConfirm({
    title: "删除定时任务",
    msg: `“${job.name || job.id}” 将被永久删除，已有运行记录一并移除。`,
    onOk: async () => {
      const { ok, data } = await api.deleteCronJobApi(job.id);
      if (!ok) { showToast((data as any)?.detail || "删除失败", 3000); return; }
      await loadCronJobs();
      showToast("任务已删除");
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
  return (
    <div className="cron-panel">
      <div className="panel-bar">
        <span className="panel-count">{s.cronJobs.length} 个任务</span>
        <div className="panel-bar-actions">
          <button className="dp-btn" onClick={() => void loadCronJobs()}>刷新</button>
          <button className="dp-btn primary" onClick={() => setState({ cronForm: emptyCronForm() })}>+ 新建任务</button>
        </div>
      </div>

      {s.cronForm && <CronFormView />}

      <div className="settings-list">
        {!s.cronJobs.length && <div className="settings-empty">还没有定时任务</div>}
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
                    {s.cronBusy === j.id + "trigger" ? "运行中…" : "立即运行"}
                  </button>
                  <button className="cron-act" disabled={!!s.cronBusy} onClick={() => void act(j.id, paused ? "resume" : "pause")}>
                    {paused ? "恢复" : "暂停"}
                  </button>
                  <button className="cron-act" onClick={() => setState({ cronForm: formFromJob(j) })}>编辑</button>
                  <button className="cron-act danger" onClick={() => remove(j)}>删除</button>
                </div>
              </div>
              <div className="cron-prompt" title={j.prompt}>{j.prompt}</div>
              <div className="cron-meta">
                <span>计划：{j.schedule}</span>
                <span>下次：{whenStr(j.next_run_at_ms)}</span>
                <span>上次：{whenStr(j.last_run_at_ms)}{j.last_status ? ` (${j.last_status})` : ""}</span>
                <span>已运行 {j.run_count || 0} 次</span>
                {j.timeout_seconds ? <span>超时 {j.timeout_seconds}s</span> : null}
                {j.max_retries ? <span>重试 {j.max_retries}</span> : null}
              </div>
              <button className="cron-runs-toggle" onClick={() => void openRuns(j.id)}>
                {runsOpen ? "▾" : "▸"} 运行历史
              </button>
              {runsOpen && (
                <div className="cron-runs">
                  {!(s.cronRuns[j.id] || []).length && <div className="cron-run-empty">没有运行记录</div>}
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
  const f = s.cronForm!;
  return (
    <div className="settings-form cron-form">
      <h4>{f.id ? "编辑任务" : "新建任务"}</h4>
      <input className="pf-input" placeholder="任务名（可选）" value={f.name} onChange={(e) => patch({ name: e.target.value })} />
      <div className="cron-prompt-wrap">
        <textarea
          className="pf-input pf-textarea"
          rows={4}
          placeholder="要让 agent 做什么（prompt，必填）"
          value={f.prompt}
          onChange={(e) => patch({ prompt: e.target.value })}
        />
        <button className="cron-act" disabled={s.cronBusy === "polish"} onClick={() => void polish()}>
          {s.cronBusy === "polish" ? "润色中…" : "AI 润色"}
        </button>
      </div>
      <input
        className="pf-input"
        placeholder="计划：30 7 * * *  |  every 2h  |  2026-01-15T09:30:00"
        value={f.schedule}
        onChange={(e) => patch({ schedule: e.target.value })}
      />
      <div className="pf-row">
        <input className="pf-input" type="number" placeholder="超时秒数（0 = 不限）" value={f.timeout_seconds} onChange={(e) => patch({ timeout_seconds: e.target.value })} />
        <input className="pf-input" type="number" placeholder="最大重试次数" value={f.max_retries} onChange={(e) => patch({ max_retries: e.target.value })} />
      </div>
      {!f.id && (
        <label className="pf-check">
          <input type="checkbox" checked={f.validate_run} onChange={(e) => patch({ validate_run: e.target.checked })} />
          创建后立即试跑一次做校验
        </label>
      )}
      <div className="pf-actions">
        <button className="dp-btn" onClick={() => setState({ cronForm: null })}>取消</button>
        <button className="dp-btn primary" onClick={() => void save()}>{f.id ? "保存" : "创建"}</button>
      </div>
      <button className="pf-close" onClick={() => setState({ cronForm: null })}><IconClose /></button>
    </div>
  );
}
