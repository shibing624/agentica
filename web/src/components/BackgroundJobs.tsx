import {
  useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { createPortal } from "react-dom";
import * as api from "../api";
import type { BackgroundJob } from "../api";
import { useStrings } from "../i18n";
import { IconChevronDown, IconCopy, IconExternal, IconLayers, IconSpinner, IconStop } from "../icons";
import { isLiveJob, jobDuration, orderJobs, visibleJobBadgeCount } from "../lib/backgroundJobs";
import { normalizeMentionedPath } from "../lib/filePath";
import { formatToolDisplay } from "../lib/toolDisplay";
import { bump, getState, showToast, useAppState } from "../store";

const POLL_MS = 2000;
const OUTPUT_POLL_MS = 1500;
const STOP_ARM_MS = 3000;
const HOVER_OPEN_MS = 150;
const HOVER_CLOSE_MS = 140;

type MenuMode = "closed" | "hover" | "pinned";

function publishSessionJobs(sessionId: string, list: BackgroundJob[]) {
  const st = getState();
  const prev = st.bgJobsBySession[sessionId];
  const sameLen = prev && prev.length === list.length;
  const unchanged = sameLen && prev.every((p, i) => (
    p.id === list[i].id
    && p.status === list[i].status
    && p.finished_at_ms === list[i].finished_at_ms
    && p.progress === list[i].progress
    && p.exit_code === list[i].exit_code
  ));
  const liveCount = list.filter(isLiveJob).length;
  const sess = st.sessions[sessionId];
  const countChanged = Boolean(sess && (sess.backgroundJobs || 0) !== liveCount);
  if (unchanged && !countChanged) return;
  st.bgJobsBySession = { ...st.bgJobsBySession, [sessionId]: list };
  if (sess) sess.backgroundJobs = liveCount;
  bump();
}

/**
 * Session-header background jobs control (modeled after deepseek-harness `JobListAction`):
 * - Syncs across Web, Desktop, IM, and CLI via durable per-session job records + pulse.
 * - Hover previews after 150ms; click pins; portaled above the right workspace dock.
 * - Live rows render as filled cards with human `description`, live `progress` line,
 *   live clock, two-press stop pill, and an expandable terminal log panel.
 */
export function BackgroundJobs({
  sessionId,
  streaming,
  onOpenFile,
}: {
  sessionId: string;
  streaming: boolean;
  onOpenFile?: (path: string) => void;
}) {
  const S = useStrings();
  const appState = useAppState();
  const sess = appState.sessions[sessionId];
  const remoteRunning = Boolean(sess?.running);
  const pulseSeq = appState.pulseSeq;

  const [jobs, setJobs] = useState<BackgroundJob[]>(() => getState().bgJobsBySession[sessionId] || []);
  const [mode, setMode] = useState<MenuMode>("closed");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [output, setOutput] = useState("");
  const [copied, setCopied] = useState(false);
  const [settledOpen, setSettledOpen] = useState<boolean | null>(null);
  const [cleared, setCleared] = useState<ReadonlySet<string>>(new Set());
  const [armed, setArmed] = useState<string | null>(null);
  const [stopping, setStopping] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [menuPos, setMenuPos] = useState<{ top: number; left: number }>({ top: 44, left: 16 });

  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const outputRef = useRef<HTMLPreElement>(null);
  const followRef = useRef(true);
  const hoverOpenTimer = useRef<number | null>(null);
  const hoverCloseTimer = useRef<number | null>(null);

  const open = mode !== "closed";

  const clearHoverTimers = useCallback(() => {
    if (hoverOpenTimer.current != null) window.clearTimeout(hoverOpenTimer.current);
    if (hoverCloseTimer.current != null) window.clearTimeout(hoverCloseTimer.current);
    hoverOpenTimer.current = null;
    hoverCloseTimer.current = null;
  }, []);

  useEffect(() => clearHoverTimers, [clearHoverTimers, sessionId]);

  const setMenuMode = useCallback((next: MenuMode) => {
    clearHoverTimers();
    if (next !== "closed") setNow(Date.now());
    setMode(next);
  }, [clearHoverTimers]);

  const refresh = useCallback(async () => {
    const { ok, data } = await api.fetchBackgroundJobs(sessionId);
    if (ok && data && Array.isArray(data.jobs)) {
      setJobs(data.jobs);
      publishSessionJobs(sessionId, data.jobs);
    }
  }, [sessionId]);

  const rows = useMemo(() => orderJobs(jobs), [jobs]);
  const live = useMemo(() => rows.filter(isLiveJob), [rows]);
  const settled = useMemo(() => rows.filter((j) => !isLiveJob(j) && !cleared.has(j.id)), [rows, cleared]);
  const hasLive = live.length > 0;
  const visibleCount = live.length + settled.length;

  // Refresh immediately on mount, session switch, and whenever remote pulse / catalog lands.
  useEffect(() => {
    void refresh();
  }, [refresh, pulseSeq, sess?.backgroundJobs, sess?.sizeBytes]);

  // Poll while local stream runs, remote IM/CLI run is active, a job is live, or popover is open.
  const shouldPoll = streaming || remoteRunning || hasLive || open;
  useEffect(() => {
    if (!shouldPoll) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh, shouldPoll]);

  // Refresh when returning to the window/tab.
  useEffect(() => {
    const onFocus = () => { if (!document.hidden) void refresh(); };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onFocus);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onFocus);
    };
  }, [refresh]);

  useEffect(() => {
    if (!hasLive) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [hasLive, open]);

  // Auto-expand the sole running job when opening the popover for quick inspection.
  useEffect(() => {
    if (open && expanded === null && live.length === 1) {
      followRef.current = true;
      setExpanded(live[0].id);
    }
  }, [open, expanded, live]);

  useEffect(() => {
    if (!visibleCount && open) setMenuMode("closed");
  }, [visibleCount, open, setMenuMode]);

  // Position the portaled popover anchored to the trigger and bounded inside the viewport.
  const updatePosition = useCallback(() => {
    const btn = triggerRef.current;
    if (!btn) return;
    const rect = btn.getBoundingClientRect();
    const popW = Math.min(500, window.innerWidth - 24);
    const left = Math.max(12, Math.min(rect.right - popW, window.innerWidth - popW - 12));
    const top = Math.max(12, rect.bottom + 6);
    setMenuPos({ top, left });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open, updatePosition, expanded, rows.length]);

  // Outside pointer & Escape dismissal across trigger + portaled menu.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (rootRef.current?.contains(target) || menuRef.current?.contains(target)) return;
      setMenuMode("closed");
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setMenuMode("closed");
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, setMenuMode]);

  // Output flows only while someone is looking at the expanded row.
  const expandedJob = open ? rows.find((j) => j.id === expanded) : undefined;
  const expandedStatus = expandedJob?.status;
  useEffect(() => {
    if (!expanded || !expandedStatus) return;
    let cancelled = false;
    const load = async () => {
      const { ok, data } = await api.fetchBackgroundJobOutput(sessionId, expanded);
      if (!cancelled && ok && data) setOutput(data.output);
    };
    void load();
    if (expandedStatus !== "running") return () => { cancelled = true; };
    const timer = window.setInterval(() => void load(), OUTPUT_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
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

  const hoverEnter = (e: ReactPointerEvent<HTMLElement>) => {
    if (e.pointerType === "touch") return;
    clearHoverTimers();
    if (open) return;
    hoverOpenTimer.current = window.setTimeout(() => {
      hoverOpenTimer.current = null;
      setNow(Date.now());
      setMode("hover");
    }, HOVER_OPEN_MS);
  };

  const hoverLeave = (e: ReactPointerEvent<HTMLElement>) => {
    if (e.pointerType === "touch") return;
    clearHoverTimers();
    if (!open || mode === "pinned") return;
    hoverCloseTimer.current = window.setTimeout(() => {
      hoverCloseTimer.current = null;
      setMode("closed");
    }, HOVER_CLOSE_MS);
  };

  const pressStop = async (job: BackgroundJob) => {
    setMenuMode("pinned");
    if (armed !== job.id) { setArmed(job.id); return; }
    setArmed(null);
    setStopping(job.id);
    const { ok, data } = await api.stopBackgroundJobApi(sessionId, job.id);
    setStopping(null);
    if (!ok) showToast((data as any)?.detail || S.chat.bgStopFailed, 3000);
    await refresh();
  };

  if (!visibleCount) return null;
  const badgeCount = visibleJobBadgeCount(live.length, settled.length);
  const showSettled = settledOpen ?? !hasLive;
  const cwd = sess?.dir || appState.serverDir || "";

  const renderItem = (job: BackgroundJob) => {
    const liveRow = isLiveJob(job);
    const isOpen = expanded === job.id;
    const dur = jobDuration(job, now);
    const kindText = job.kind === "delegate" ? S.chat.bgDelegate : S.chat.bgCommand;
    const statusDetail = job.status === "failed" && job.exit_code != null
      ? S.chat.bgExitCode(job.exit_code)
      : S.chat.bgStatus[job.status];
    const cleanCmd = job.command ? formatToolDisplay("execute", { command: job.command }, cwd) : "";
    const showCmdHeader = Boolean(cleanCmd && cleanCmd.trim() !== job.label.trim());
    const openableLog = job.log_path && onOpenFile
      ? normalizeMentionedPath(job.log_path, cwd || null)
      : null;

    return (
      <li key={job.id} className={"bg-item" + (liveRow ? " live" : " settled")}>
        <div className={"bg-line" + (liveRow ? " live" : "")}>
          <button
            type="button"
            className={"bg-row" + (liveRow ? " live" : " settled")}
            aria-expanded={isOpen}
            onClick={() => {
              setMenuMode("pinned");
              if (!isOpen) setOutput("");
              followRef.current = true;
              setExpanded(isOpen ? null : job.id);
            }}
          >
            <span className={"bg-dot " + job.status}>
              {liveRow && <IconSpinner />}
            </span>
            {liveRow ? (
              <span className="bg-main">
                <span className="bg-primary">
                  <span className="bg-label" title={job.label}>{job.label}</span>
                </span>
                <span className="bg-secondary">
                  <span className="bg-kind">{kindText}</span>
                  {job.progress ? (
                    <span className="bg-progress" title={job.progress}>{job.progress}</span>
                  ) : null}
                  <span className="bg-dur">{dur}</span>
                </span>
              </span>
            ) : (
              <>
                <span className="bg-kind">{kindText}</span>
                <span className="bg-label" title={job.label}>{job.label}</span>
                <span className={"bg-status " + job.status}>{statusDetail}</span>
                <span className="bg-dur">{dur}</span>
              </>
            )}
            <span className={"bg-chev-box" + (isOpen ? " open" : "")}>
              <IconChevronDown />
            </span>
          </button>
          {liveRow && (
            <button
              type="button"
              className={"bg-stop" + (armed === job.id ? " armed" : "")}
              disabled={stopping === job.id}
              title={S.chat.bgStop}
              aria-label={S.chat.bgStop}
              onClick={() => void pressStop(job)}
            >
              <IconStop />
              {armed === job.id && <span className="bg-stop-label">{S.chat.bgStopConfirm}</span>}
            </button>
          )}
        </div>
        {isOpen && (
          <div className="bg-panel">
            <div className="bg-panel-head">
              <span className="bg-panel-cmd" title={cleanCmd || job.label}>
                {showCmdHeader ? `$ ${cleanCmd}` : `#${job.num} ${job.id}${job.pid ? ` · PID ${job.pid}` : ""}`}
              </span>
              <span className="bg-panel-acts">
                {openableLog && onOpenFile && (
                  <button
                    type="button"
                    className="bg-panel-btn"
                    title={S.chat.bgOpenLogTip}
                    onClick={() => {
                      onOpenFile(openableLog);
                      if (mode === "hover") setMenuMode("closed");
                    }}
                  >
                    <IconExternal />
                    <span>{S.chat.bgOpenLog}</span>
                  </button>
                )}
                {(output || cleanCmd) && (
                  <button
                    type="button"
                    className="bg-panel-btn"
                    onClick={() => {
                      void navigator.clipboard.writeText(output || cleanCmd).then(() => {
                        setCopied(true);
                        window.setTimeout(() => setCopied(false), 1500);
                      });
                    }}
                  >
                    <IconCopy />
                    <span>{copied ? S.common.copied : S.common.copy}</span>
                  </button>
                )}
              </span>
            </div>
            <pre
              className="bg-output"
              ref={outputRef}
              onScroll={(e) => {
                const el = e.currentTarget;
                followRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 12;
              }}
            >
              {output || S.chat.bgNoOutput}
            </pre>
          </div>
        )}
      </li>
    );
  };

  return (
    <div
      className="bg-jobs"
      ref={rootRef}
      onPointerEnter={hoverEnter}
      onPointerLeave={hoverLeave}
    >
      <button
        ref={triggerRef}
        type="button"
        className={"bg-trigger" + (open ? " on" : "") + (hasLive ? " live" : "")}
        aria-expanded={open}
        onClick={() => setMenuMode(!open || mode !== "pinned" ? "pinned" : "closed")}
      >
        <span className="bg-trigger-icon">
          {hasLive ? <span className="bg-live-dot" aria-hidden /> : <IconLayers />}
        </span>
        <span>{S.chat.bgCount(badgeCount)}</span>
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className="bg-pop"
          style={{ top: menuPos.top, left: menuPos.left }}
          onPointerEnter={hoverEnter}
          onPointerLeave={hoverLeave}
          onPointerDownCapture={() => { if (mode === "hover") setMenuMode("pinned"); }}
        >
          <ul className="bg-list" aria-label={S.chat.bgCount(badgeCount)}>
            {hasLive && <li className="bg-section">{S.chat.bgLive}</li>}
            {live.map(renderItem)}
            {settled.length > 0 && (
              <li className="bg-section settled-head">
                <button
                  type="button"
                  className={"bg-section-toggle" + (showSettled ? " open" : "")}
                  aria-expanded={showSettled}
                  onClick={() => { setMenuMode("pinned"); setSettledOpen(!showSettled); }}
                >
                  <IconChevronDown />
                  <span>{S.chat.bgSettled(settled.length)}</span>
                </button>
                <button
                  type="button"
                  className="bg-clear"
                  onClick={() => {
                    const next = new Set(cleared);
                    for (const j of settled) next.add(j.id);
                    setCleared(next);
                    if (expanded && settled.some((j) => j.id === expanded)) setExpanded(null);
                  }}
                >
                  {S.chat.bgClear}
                </button>
              </li>
            )}
            {showSettled && settled.map(renderItem)}
          </ul>
        </div>,
        document.body,
      )}
    </div>
  );
}
