import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as api from "../api";
import type { BackgroundJob } from "../api";
import { useStrings } from "../i18n";
import { IconChevronDown, IconLayers, IconSpinner, IconStop } from "../icons";
import { isLiveJob, jobDuration, orderJobs } from "../lib/backgroundJobs";
import { showToast } from "../store";
import { useDismiss } from "../workspace/PreviewActions";

const POLL_MS = 2000;
const STOP_ARM_MS = 3000;

/**
 * This chat's background commands and delegated workers, in the header: the
 * Web's `/ps` and `/stop`. Draws nothing until the chat has a job. Keyed by
 * session at the call site, so all local state belongs to one chat.
 *
 * A job can only start during a turn, so polling runs while one streams or
 * while a job is still live, and stops otherwise.
 */
export function BackgroundJobs({ sessionId, streaming }: { sessionId: string; streaming: boolean }) {
  const S = useStrings();
  const [jobs, setJobs] = useState<BackgroundJob[]>([]);
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [output, setOutput] = useState("");
  const [settledOpen, setSettledOpen] = useState<boolean | null>(null);
  const [cleared, setCleared] = useState<ReadonlySet<string>>(new Set());
  const [armed, setArmed] = useState<string | null>(null);
  const [stopping, setStopping] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const outputRef = useRef<HTMLPreElement>(null);
  // Follow the newest lines like a terminal, unless the reader scrolled up.
  const followRef = useRef(true);
  const close = useCallback(() => setOpen(false), []);
  const rootRef = useDismiss(open, close);

  const refresh = useCallback(async () => {
    const { ok, data } = await api.fetchBackgroundJobs(sessionId);
    if (ok && data) setJobs(data.jobs);
  }, [sessionId]);

  const rows = useMemo(() => orderJobs(jobs), [jobs]);
  const live = rows.filter(isLiveJob);
  const settled = rows.filter((j) => !isLiveJob(j) && !cleared.has(j.id));
  const hasLive = live.length > 0;

  useEffect(() => {
    void refresh();
    if (!streaming && !hasLive) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh, streaming, hasLive]);

  useEffect(() => {
    if (!open || !hasLive) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [open, hasLive]);

  // Output flows only while someone is looking at it. Depending on the status
  // reads the last lines once more when the job ends.
  const expandedStatus = open ? rows.find((j) => j.id === expanded)?.status : undefined;
  useEffect(() => {
    if (!expanded || !expandedStatus) return;
    const load = async () => {
      const { ok, data } = await api.fetchBackgroundJobOutput(sessionId, expanded);
      if (ok && data) setOutput(data.output);
    };
    void load();
    if (expandedStatus !== "running") return;
    const timer = window.setInterval(() => void load(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [sessionId, expanded, expandedStatus]);

  useLayoutEffect(() => {
    const el = outputRef.current;
    if (el && followRef.current) el.scrollTop = el.scrollHeight;
  }, [output, expanded, open]);

  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(null), STOP_ARM_MS);
    return () => window.clearTimeout(timer);
  }, [armed]);

  const pressStop = async (job: BackgroundJob) => {
    if (armed !== job.id) { setArmed(job.id); return; }
    setArmed(null);
    setStopping(job.id);
    const { ok, data } = await api.stopBackgroundJobApi(sessionId, job.id);
    setStopping(null);
    if (!ok) showToast((data as any)?.detail || S.chat.bgStopFailed, 3000);
    await refresh();
  };

  const count = live.length || settled.length;
  if (!live.length && !settled.length) return null;
  const showSettled = settledOpen ?? !hasLive;

  const row = (job: BackgroundJob) => {
    const isOpen = expanded === job.id;
    const detail = job.status === "failed" && job.exit_code != null
      ? S.chat.bgExitCode(job.exit_code)
      : S.chat.bgStatus[job.status];
    return (
      <div key={job.id}>
        <div className="bg-line">
          <button type="button" className="bg-row" aria-expanded={isOpen}
                  onClick={() => { setOutput(""); followRef.current = true; setExpanded(isOpen ? null : job.id); }}>
            <span className={"bg-dot " + job.status}>{job.status === "running" && <IconSpinner />}</span>
            <span className="bg-label" title={job.label}>{job.label}</span>
            <span className="bg-meta">
              {job.kind === "delegate" ? S.chat.bgDelegate : S.chat.bgCommand} · {isLiveJob(job) ? "" : `${detail} · `}{jobDuration(job, now)}
            </span>
            <span className={"bg-chev" + (isOpen ? " open" : "")}><IconChevronDown /></span>
          </button>
          {isLiveJob(job) && (
            <button type="button" className={"bg-stop" + (armed === job.id ? " armed" : "")}
                    disabled={stopping === job.id} title={S.chat.bgStop} aria-label={S.chat.bgStop}
                    onClick={() => void pressStop(job)}>
              <IconStop />
              {armed === job.id && <span>{S.chat.bgStopConfirm}</span>}
            </button>
          )}
        </div>
        {isOpen && (
          <pre className="bg-output" ref={outputRef}
               onScroll={(e) => {
                 const el = e.currentTarget;
                 followRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 8;
               }}>
            {output || S.chat.bgNoOutput}
          </pre>
        )}
      </div>
    );
  };

  return (
    <div className="bg-jobs" ref={rootRef}>
      <button type="button" className={"bg-trigger" + (open ? " on" : "")} aria-expanded={open}
              onClick={() => { setNow(Date.now()); followRef.current = true; setOpen((v) => !v); }}>
        {hasLive ? <IconSpinner /> : <IconLayers />}
        <span>{S.chat.bgCount(count)}</span>
      </button>
      {open && (
        <div className="bg-pop">
          {hasLive && <div className="bg-section">{S.chat.bgLive}</div>}
          {live.map(row)}
          {settled.length > 0 && (
            <div className="bg-section">
              <button type="button" className={"bg-section-toggle" + (showSettled ? " open" : "")} aria-expanded={showSettled}
                      onClick={() => setSettledOpen(!showSettled)}>
                <IconChevronDown />{S.chat.bgSettled(settled.length)}
              </button>
              <button type="button" className="bg-clear"
                      onClick={() => setCleared(new Set([...cleared, ...settled.map((j) => j.id)]))}>
                {S.chat.bgClear}
              </button>
            </div>
          )}
          {showSettled && settled.map(row)}
        </div>
      )}
    </div>
  );
}
