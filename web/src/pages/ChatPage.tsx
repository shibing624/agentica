import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router";
import * as api from "../api";
import { ChatNav } from "../components/ChatNav";
import { ComposerDir } from "../components/ComposerDir";
import { SteerChip, WorkGroup, ApprovalCard } from "../components/WorkGroup";
import { RenameInput } from "../components/SessionTree";
import { SlashMenu, SkillsPicker, filterSlashItems, invocableSkills, slashQuery, webSlashItems, type SlashItem } from "../components/SlashMenu";
import { FeedbackDialog, SkillTurn } from "../components/MessageExtras";
import { leadingSkillToken, parseSkillTurn } from "../lib/skillInvocation";
import { getStrings, useStrings } from "../i18n";
import { fmtCost, fmtDurationMs, fmtN, fmtTps, formatErrorStamp, parseTokenBudget, uid, UNLIMITED_TOKEN_BUDGET } from "../lib/format";
import { primeSettings, switchProfile } from "../panels/SettingsModal";
import { createSession, forkAt, hydrateSession, markSessionRead, pushPendingName, rateMessage, renameSession, syncSessionRoundStats } from "../sessions";
import { loadPlugins } from "../data";
import {
  Logo, IconPlus, IconClose, IconSidebar, IconChat, IconAsk, IconAuto,
  IconAllowAll, IconSend, IconStop, IconCopy, IconArrowDown, IconPencil, IconBook, IconDatabase,
  IconChevronDown, IconThumbUp, IconThumbDown, IconFork,
} from "../icons";
import {
  bump, dequeueApproval, enqueueApproval, getState, pushPrefs, saveSessions, setState,
  showToast, useAppState, type ApprovalDecision, type ApprovalRequest, type ChatMsg, type QueuedMessage,
} from "../store";
import { applySessionUsage, ContextUsageTip } from "../components/ContextUsageTip";
import { appendSteerPart, appendText, appendThink, appendTool, finishTool, groupParts, partsOf, settleWork, unfinishedToolCallId } from "../lib/msgParts";
import { approvalKeyAction } from "../lib/approvalKeys";
import { createStreamFollow, stickToBottom } from "../lib/streamFollow";
import { FilesPanel } from "../workspace/FilesPanel";
import { MessageFilesCard } from "../workspace/MessageFilesCard";
import { useFilesPanel } from "../workspace/useFilesPanel";

const ChatMarkdown = lazy(() => import("../components/ChatMarkdown").then((m) => ({ default: m.ChatMarkdown })));

function permIcon(mode: string) {
  if (mode === "ask") return <IconAsk />;
  if (mode === "allow-all") return <IconAllowAll />;
  return <IconAuto />;
}

export function ChatPage() {
  const s = useAppState();
  const S = useStrings();
  const taRef = useRef<HTMLTextAreaElement>(null);
  const goalBudgetRef = useRef<HTMLInputElement>(null);
  const hadGoalCompose = useRef(false);
  const goalBudgetSnapshot = useRef("");
  const msgsRef = useRef<HTMLDivElement>(null);
  const chatWrapRef = useRef<HTMLDivElement>(null);
  const inputAreaRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(createStreamFollow());
  const [showJump, setShowJump] = useState(false);
  const [slashActive, setSlashActive] = useState(0);
  const [skillsOpen, setSkillsOpen] = useState(false);
  const [ctxOpen, setCtxOpen] = useState(false);
  const [titleEditing, setTitleEditing] = useState(false);
  const [goalBudgetOpen, setGoalBudgetOpen] = useState(false);
  const modelWrapRef = useRef<HTMLDivElement>(null);
  const approvalWrapRef = useRef<HTMLDivElement>(null);
  const skillsWrapRef = useRef<HTMLDivElement>(null);
  const ctxWrapRef = useRef<HTMLDivElement>(null);
  const skillHlRef = useRef<HTMLDivElement>(null);
  const skillMarkRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (s.goalCompose && !hadGoalCompose.current) {
      requestAnimationFrame(() => taRef.current?.focus());
    }
    if (!s.goalCompose) setGoalBudgetOpen(false);
    hadGoalCompose.current = !!s.goalCompose;
  }, [s.goalCompose]);

  useEffect(() => {
    if (goalBudgetOpen) {
      goalBudgetRef.current?.focus();
      goalBudgetRef.current?.select();
    }
  }, [goalBudgetOpen]);

  const streaming = !!s.streams[s.curSess || ""];
  const goalRun = s.curSess ? s.goalRuns[s.curSess] : undefined;
  const compacting = !!(s.curSess && s.commandRuns[s.curSess]);
  const busy = streaming || !!goalRun || compacting;
  const showStop = busy && !compacting;
  const cur = s.curSess ? s.sessions[s.curSess] : null;
  const queued = s.messageQueue.filter((q) => q.sessionId === s.curSess);
  const pendingImages = s.pendingFiles.filter((f) => f.type.startsWith("image/"));
  const pendingApprovals = s.curSess ? (s.streams[s.curSess]?.pendingApprovals || []) : [];
  const pendingReq = pendingApprovals[0];
  const q = slashQuery(s.inputText);
  const slashItems = q != null && !pendingReq
    ? filterSlashItems(webSlashItems(s.pluginsData.skills || [], S.chat), q)
    : [];
  const slashOpen = q != null && !pendingReq;
  const pendingToolCallId = pendingReq?.toolCallId;
  const decidingApproval = !!(s.curSess && s.streams[s.curSess]?.decidingApproval);
  const goalBudgetText = s.goalCompose?.budgetText ?? "";
  const goalBudgetParsed = parseTokenBudget(goalBudgetText);
  const goalBudgetInvalid = goalBudgetParsed === null;
  const goalBudgetSummary =
    goalBudgetParsed === null || goalBudgetParsed === UNLIMITED_TOKEN_BUDGET
      ? S.chat.goalBudgetUnlimited
      : S.chat.goalBudgetValue(/[km]$/i.test(goalBudgetText.trim()) ? goalBudgetText.trim() : fmtN(goalBudgetParsed));
  const openGoalBudget = () => {
    goalBudgetSnapshot.current = goalBudgetText;
    setGoalBudgetOpen(true);
  };
  const saveGoalBudget = () => {
    if (goalBudgetInvalid) return;
    setGoalBudgetOpen(false);
    requestAnimationFrame(() => taRef.current?.focus());
  };
  const cancelGoalBudget = () => {
    if (s.goalCompose) setState({ goalCompose: { budgetText: goalBudgetSnapshot.current } });
    setGoalBudgetOpen(false);
    requestAnimationFrame(() => taRef.current?.focus());
  };

  useEffect(() => { void loadPlugins(); }, []);
  useEffect(() => {
    if (slashOpen) setSkillsOpen(false);
    if (pendingReq) {
      setSkillsOpen(false);
      setState({ approvalMenuOpen: false });
    }
  }, [slashOpen, pendingReq?.toolCallId]);
  useEffect(() => {
    if (!s.modelDDOpen && !s.approvalMenuOpen && !skillsOpen && !ctxOpen) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      const patch: { modelDDOpen?: false; approvalMenuOpen?: false } = {};
      if (!modelWrapRef.current?.contains(t)) patch.modelDDOpen = false;
      if (!approvalWrapRef.current?.contains(t)) patch.approvalMenuOpen = false;
      if (patch.modelDDOpen !== undefined || patch.approvalMenuOpen !== undefined) setState(patch);
      if (!skillsWrapRef.current?.contains(t)) setSkillsOpen(false);
      if (!ctxWrapRef.current?.contains(t)) setCtxOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      setSkillsOpen(false);
      setCtxOpen(false);
      setState({ modelDDOpen: false, approvalMenuOpen: false });
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [s.modelDDOpen, s.approvalMenuOpen, skillsOpen, ctxOpen]);
  useEffect(() => {
    if (!pendingReq) return;
    const onKey = (e: KeyboardEvent) => {
      if (s.modelDDOpen || s.approvalMenuOpen || skillsOpen) return;
      if (e.isComposing || e.keyCode === 229) return;
      const action = approvalKeyAction(e.key, { shiftKey: e.shiftKey });
      if (!action) return;
      e.preventDefault();
      e.stopPropagation();
      void decideCurrentApproval(action);
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [pendingReq?.toolCallId, s.modelDDOpen, s.approvalMenuOpen, skillsOpen]);
  useEffect(() => {
    const onHide = () => { pageUnloading = true; };
    const onShow = () => { pageUnloading = false; };
    window.addEventListener("pagehide", onHide);
    window.addEventListener("pageshow", onShow);
    return () => {
      window.removeEventListener("pagehide", onHide);
      window.removeEventListener("pageshow", onShow);
    };
  }, []);
  useEffect(() => {
    const id = s.pendingResume;
    if (!id) return;
    setState({ pendingResume: null });
    void resumeLiveStream(id);
  }, [s.pendingResume]);
  useEffect(() => {
    if (!s.curSess || streaming) return;
    const sess = s.sessions[s.curSess];
    if (sess) {
      let healed = false;
      for (const m of sess.msgs) {
        if (settleWork(m)) healed = true;
      }
      if (healed) saveSessions();
    }
    void syncSessionRoundStats(s.curSess);
  }, [s.curSess, streaming]);

  useLayoutEffect(() => {
    const el = msgsRef.current;
    if (!el || !followRef.current.stick) return;
    stickToBottom(el, followRef.current);
  });

  const syncJump = () => {
    const el = msgsRef.current;
    const follow = followRef.current;
    setShowJump(
      !follow.stick &&
        !!el &&
        el.scrollHeight - el.scrollTop - el.clientHeight > 1 &&
        !!(cur && cur.msgs.length),
    );
  };

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 220) + "px";
    if (skillHlRef.current) skillHlRef.current.scrollTop = ta.scrollTop;
  }, [s.inputText]);

  useEffect(() => { setSlashActive(0); }, [s.inputText]);

  useEffect(() => {
    const area = inputAreaRef.current;
    const wrap = chatWrapRef.current;
    if (!area || !wrap) return;
    const measure = () => wrap.style.setProperty("--composer-height", `${area.offsetHeight}px`);
    const observer = new ResizeObserver(measure);
    observer.observe(area);
    measure();
    return () => observer.disconnect();
  }, []);

  const windowSize = s.serverContextWindow || 128000;
  const occupancy = cur?.contextTokens || cur?.lastInputTokens || 0;
  const ctxPct = Math.min(100, Math.round((occupancy / windowSize) * 100));
  const files = useFilesPanel(s.curSess);
  const workspace = cur?.dir || s.pendingNewChatDir || s.serverDir || "";
  useEffect(() => { setCtxOpen(false); setTitleEditing(false); }, [s.curSess]);
  const skillToken = leadingSkillToken(s.inputText, invocableSkills(s.pluginsData.skills || []));
  // The token is drawn in an overlay under the textarea; hit-test its boxes so
  // a click on the name opens the skill while a click elsewhere just edits.
  const overSkillToken = (x: number, y: number) => {
    const mark = skillMarkRef.current;
    if (!skillToken || !mark) return false;
    return Array.from(mark.getClientRects()).some((r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom);
  };

  return (
    <>
      <div className={"main" + (!cur || !cur.msgs.length ? " chat-empty" : "")}>
        <div className="topbar">
          <div className="tb-l">
            <button className="ib sidebar-expand-btn" onClick={() => setState({ sidebarCollapsed: false })} title={S.nav.expand}><IconSidebar /></button>
            <div className="tb-chat-info">
              {cur && <span className="tb-chat-icon"><IconChat /></span>}
              {titleEditing && s.curSess && cur ? (
                <RenameInput className="tb-chat-rename-input" initial={cur.title}
                             onDone={(name) => {
                               setTitleEditing(false);
                               if (name !== null && s.curSess) void renameSession(s.curSess, name);
                             }} />
              ) : (
                <span className="tb-chat-title" title={cur ? S.chat.dblclickRename : undefined}
                      onDoubleClick={() => { if (cur) setTitleEditing(true); }}>
                  {cur?.title || S.chat.newConversation}
                </span>
              )}
              {s.curSess && <Link className="trace-link" to={`/traces?sessionId=${s.curSess}`}>{S.chat.viewTrace}</Link>}
            </div>
          </div>
          <div className="tb-r">
            <button type="button" className={"tb-workspace" + (files.open ? " on" : "")}
                    onClick={() => files.setOpen(!files.open)}>
              {S.chat.openWorkspace}
            </button>
          </div>
        </div>
        <div className="chat-wrap" ref={chatWrapRef}>
          <div className="chat" id="chatArea" ref={msgsRef}
            onWheel={(e) => { followRef.current.wheel(e.deltaY); syncJump(); }}
            onTouchStart={(e) => followRef.current.touchStart(e.touches[0]?.clientY ?? 0)}
            onTouchMove={(e) => { followRef.current.touchMove(e.touches[0]?.clientY ?? 0); syncJump(); }}
            onTouchEnd={() => followRef.current.touchEnd()}
            onScroll={() => {
              const el = msgsRef.current;
              if (!el) return;
              followRef.current.scrolled({
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
              });
              syncJump();
            }}>
            <div className="msgs">
              {!cur || !cur.msgs.length ? (
                <div className="welcome welcome-new">
                  <Logo className="w-icon-img" />
                  <h2>Agentica</h2>
                  <p>{S.chat.welcome}</p>
                </div>
              ) : cur.msgs.map((m, i) => (
                <MessageView key={i} idx={i} m={m} sessionId={s.curSess} workspace={workspace}
                  onOpenFile={files.browsePath} onOpenSkill={files.openSkill}
                  live={!!(s.curSess && s.streams[s.curSess]?.aiMsg === m)}
                  pendingToolCallId={pendingToolCallId} />
              ))}
            </div>
          </div>
          <ChatNav
            msgs={cur?.msgs || []}
            areaRef={msgsRef}
            onLeaveFollow={() => followRef.current.leave()}
          />
          {showJump && (
            <button type="button" className="scroll-bottom-btn visible" title={S.chat.jumpLatest}
                    onClick={() => {
                      const el = msgsRef.current;
                      followRef.current.resume();
                      if (el) stickToBottom(el, followRef.current);
                      setShowJump(false);
                    }}>
              <IconArrowDown />
            </button>
          )}
          <div className="input-area" ref={inputAreaRef}>
          {pendingReq && (
            <ApprovalCard
              req={pendingReq}
              queueIndex={0}
              queueTotal={pendingApprovals.length}
              busy={decidingApproval}
              onDecide={(d) => void decideCurrentApproval(d)}
            />
          )}
          {goalRun && (
            <div className="goal-bar" title={goalRun.progress}>
              <span className="goal-bar-status">{S.chat.goalBar(goalRun.status, goalRun.objective)}</span>
              {goalRun.progress && <span className="goal-bar-progress">{goalRun.progress}</span>}
            </div>
          )}
          {compacting && (
            <div className="goal-bar">
              <span className="goal-bar-status">{S.chat.compactingBar}</span>
            </div>
          )}
          {!!queued.length && (
            <div className="queue-bar">
              <span className="queue-label">{S.chat.queuedCount(queued.length)}</span>
              {queued.map((qItem) => (
                <span className="queue-chip" key={qItem.id} title={qItem.text}>
                  <button type="button" className="queue-chip-text" title={S.chat.editQueued}
                          onClick={() => editQueued(qItem)}>
                    {qItem.text.slice(0, 40) || S.chat.attachmentOnly}
                  </button>
                  <button type="button" title={S.chat.editQueued} onClick={() => editQueued(qItem)}><IconPencil /></button>
                  <button type="button" onClick={() => setState({ messageQueue: getState().messageQueue.filter((x) => x.id !== qItem.id) })}><IconClose /></button>
                </span>
              ))}
            </div>
          )}
          <div className={"input-box" + (pendingReq ? " approval-locked" : "")} onPaste={(e) => {
            if (pendingReq) return;
            const files = imageFilesFromClipboard(e.clipboardData);
            if (!files.length) return;
            e.preventDefault();
            setState({ pendingFiles: [...getState().pendingFiles, ...files] });
          }} onDragOver={(e) => {
            if (pendingReq) return;
            e.preventDefault();
            e.currentTarget.classList.add("dragover");
          }}
             onDragLeave={(e) => e.currentTarget.classList.remove("dragover")}
             onDrop={(e) => {
               e.preventDefault();
               e.currentTarget.classList.remove("dragover");
               if (pendingReq) return;
               const files = Array.from(e.dataTransfer.files || []);
               if (files.length) setState({ pendingFiles: [...getState().pendingFiles, ...files] });
             }}>
            {slashOpen && (
              <SlashMenu items={slashItems} active={slashActive} onPick={(it) => pickSlash(it, taRef)} />
            )}
            {s.goalCompose && (
              <div className="goal-chip-row" inert={pendingReq ? true : undefined}>
                <span className="goal-chip">
                  <span className="goal-chip-label">{S.chat.goalMode}</span>
                  <span className="goal-chip-sep" aria-hidden />
                  {goalBudgetOpen ? (
                    <label className="goal-chip-budget">
                      <span>{S.chat.goalBudgetLabel}</span>
                      <input
                        ref={goalBudgetRef}
                        value={s.goalCompose.budgetText}
                        placeholder={S.chat.goalBudgetPlaceholder}
                        title={goalBudgetInvalid ? S.chat.goalBudgetInvalid : S.chat.goalBudgetHint}
                        className={goalBudgetInvalid ? "invalid" : undefined}
                        aria-invalid={goalBudgetInvalid}
                        onChange={(e) => setState({ goalCompose: { budgetText: e.target.value } })}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            e.stopPropagation();
                            saveGoalBudget();
                          } else if (e.key === "Escape") {
                            e.preventDefault();
                            e.stopPropagation();
                            cancelGoalBudget();
                          }
                        }}
                      />
                      <button
                        type="button"
                        className="goal-chip-budget-save"
                        title={S.chat.goalBudgetSave}
                        aria-label={S.chat.goalBudgetSave}
                        disabled={goalBudgetInvalid}
                        onClick={saveGoalBudget}
                      >
                        <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" aria-hidden>
                          <path d="M3.5 8.5l3 3 6-6" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </button>
                    </label>
                  ) : (
                    <button
                      type="button"
                      className="goal-chip-budget-btn"
                      aria-label={goalBudgetSummary}
                      aria-expanded={false}
                      title={S.chat.goalBudgetHint}
                      onClick={openGoalBudget}
                    >
                      <span>{goalBudgetSummary}</span>
                      <IconChevronDown />
                    </button>
                  )}
                  <button type="button" className="goal-chip-close" title={S.chat.goalRemove} onClick={() => {
                    setState({ goalCompose: null });
                    taRef.current?.focus();
                  }}>
                    <IconClose />
                  </button>
                </span>
              </div>
            )}
            {s.pendingFiles.length > 0 && (
              <div className="file-list" inert={pendingReq ? true : undefined}>
                {s.pendingFiles.map((f, i) => (
                  <PendingFileChip key={i} file={f} onRemove={() => setState({ pendingFiles: s.pendingFiles.filter((_, j) => j !== i) })} />
                ))}
              </div>
            )}
            {pendingImages.length > 0 && !s.serverSupportsImages && (
              <div className="vision-note">{S.chat.visionFallback(s.serverMediaModel, s.serverConfigPath)}</div>
            )}
            <div className="input-ta-wrap">
            {skillToken && (
              <div className="input-ta-hl" ref={skillHlRef} aria-hidden>
                {s.inputText.slice(0, skillToken.start)}
                <mark ref={skillMarkRef} className="skill-ref-hl">{s.inputText.slice(skillToken.start, skillToken.end)}</mark>
                {s.inputText.slice(skillToken.end)}{"\n"}
              </div>
            )}
            <textarea
              ref={taRef}
              className="input-ta"
              rows={1}
              disabled={!!pendingReq}
              placeholder={compacting ? S.chat.placeholderCompacting : s.goalCompose ? S.chat.placeholderGoal : pendingReq ? S.chat.placeholderApproval : busy ? S.chat.placeholderStreaming : S.chat.placeholder}
              value={s.inputText}
              onChange={(e) => setState({ inputText: e.target.value })}
              onScroll={(e) => { if (skillHlRef.current) skillHlRef.current.scrollTop = e.currentTarget.scrollTop; }}
              onMouseMove={(e) => {
                const over = overSkillToken(e.clientX, e.clientY);
                e.currentTarget.style.cursor = over ? "pointer" : "";
                e.currentTarget.title = over ? S.chat.skillChipTip : "";
              }}
              onClick={(e) => {
                const ta = e.currentTarget;
                if (!skillToken || e.detail !== 1 || ta.selectionStart !== ta.selectionEnd) return;
                if (overSkillToken(e.clientX, e.clientY)) files.openSkill(skillToken.name);
              }}
              onKeyDown={(e) => {
                if (slashOpen && slashItems.length && !compacting) {
                  if (e.key === "ArrowDown") {
                    e.preventDefault();
                    setSlashActive((n) => (n + 1) % slashItems.length);
                    return;
                  }
                  if (e.key === "ArrowUp") {
                    e.preventDefault();
                    setSlashActive((n) => (n - 1 + slashItems.length) % slashItems.length);
                    return;
                  }
                  if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
                    e.preventDefault();
                    pickSlash(slashItems[slashActive] || slashItems[0], taRef);
                    return;
                  }
                  if (e.key === "Escape") {
                    e.preventDefault();
                    setState({ inputText: "" });
                    return;
                  }
                }
                if (e.key === "Escape" && busy && !compacting) {
                  e.preventDefault();
                  void stopGen();
                  return;
                }
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  if (compacting) {
                    showToast(S.chat.compactWait);
                    return;
                  }
                  void submit();
                }
              }}
            />
            </div>
            <div className="input-foot">
              <div className="input-foot-l" inert={pendingReq ? true : undefined}>
                <input type="file" multiple accept="image/*,*/*" style={{ display: "none" }} id="fileInput" onChange={(e) => {
                  const files = Array.from(e.target.files || []);
                  if (files.length) setState({ pendingFiles: [...getState().pendingFiles, ...files] });
                  e.currentTarget.value = "";
                }} />
                <button className="foot-btn plus-btn" title={S.chat.attach} onClick={() => document.getElementById("fileInput")?.click()}><IconPlus /></button>
                <div className="skills-wrap" ref={skillsWrapRef}>
                  <button type="button" className="foot-btn skills-btn" title={S.chat.slashSkill}
                          onClick={() => {
                            setState({ approvalMenuOpen: false, modelDDOpen: false });
                            setSkillsOpen(!skillsOpen);
                          }}>
                    <IconBook />
                    <span>{S.chat.slashSkill}</span>
                    <span className="arr"><IconChevronDown /></span>
                  </button>
                  {skillsOpen && (
                    <SkillsPicker
                      skills={s.pluginsData.skills || []}
                      onPick={(cmd) => {
                        setState({ inputText: cmd + " " });
                        setSkillsOpen(false);
                        taRef.current?.focus();
                      }}
                    />
                  )}
                </div>
                <ComposerDir />
                <div className="approval-wrap" ref={approvalWrapRef}>
                  <button className="foot-btn approval-btn" title={S.chat.permTip(s.selectedApprovalMode)}
                          onClick={() => {
                            setSkillsOpen(false);
                            setState({ approvalMenuOpen: !s.approvalMenuOpen, modelDDOpen: false });
                          }}>
                    <span className="quick-icon">{permIcon(s.selectedApprovalMode)}</span>
                    <span className="approval-label">{S.chat.permLabel(s.selectedApprovalMode)}</span>
                    <span className="approval-id">({s.selectedApprovalMode})</span>
                    <span className="arr"><IconChevronDown /></span>
                  </button>
                  {s.approvalMenuOpen && (
                    <div className="approval-dd open">
                      {(["ask", "auto", "allow-all"] as const).map((mode) => (
                        <button key={mode} className={"quick-item" + (s.selectedApprovalMode === mode ? " active" : "")}
                                title={S.chat.permHint(mode)}
                                onClick={() => {
                                  localStorage.setItem("ag_approval", mode);
                                  setState({ selectedApprovalMode: mode, approvalMenuOpen: false });
                                  pushPrefs();
                                }}>
                          <span className="quick-icon">{permIcon(mode)}</span>
                          <span className="approval-label">{S.chat.permLabel(mode)}</span>
                          <span className="approval-id">({mode})</span>
                          {s.selectedApprovalMode === mode && <span className="quick-check">✓</span>}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <div className="input-foot-r">
                <div className="ctx-wrap" ref={ctxWrapRef}>
                  <button type="button" className={"foot-btn input-ctx" + (ctxOpen ? " on" : "")}
                          title={ctxOpen ? undefined : S.chat.ctxUsage} aria-haspopup="dialog" aria-expanded={ctxOpen}
                          onClick={() => setCtxOpen((v) => !v)}>
                    <span className="ctx-ring" style={{ ["--pct" as any]: ctxPct + "%" }} />
                    {ctxPct}%
                  </button>
                  {ctxOpen && (
                    <ContextUsageTip sessionId={s.curSess} fallback={cur} />
                  )}
                </div>
                <div className="input-model-wrap" ref={modelWrapRef}>
                  <button className="foot-btn model-sel" onClick={() => {
                    setSkillsOpen(false);
                    setState({ modelDDOpen: !s.modelDDOpen, approvalMenuOpen: false });
                  }}>
                    {s.serverModelName || s.serverModel}
                  </button>
                  {s.modelDDOpen && (
                    <div className="model-dd open">
                      <div className="dd-config-card">
                        <div className="dd-config-row"><span>Profile</span><strong>{s.serverProfile || s.profilesData.active || "default"}</strong></div>
                        <div className="dd-config-row"><span>{S.chat.model}</span><strong>{s.serverProvider}/{s.serverModelName || s.serverModel}</strong></div>
                        {s.serverReasoningEffort && <div className="dd-config-row"><span>effort</span><strong>{s.serverReasoningEffort}</strong></div>}
                      </div>
                      <div className="dd-section">{S.chat.switchProfile}</div>
                      {!(s.profilesData.profiles || []).length && (
                        <div className="dd-empty">{S.chat.noProfiles}</div>
                      )}
                      {(s.profilesData.profiles || []).map((p: any) => (
                        <button key={p.name} className={"dd-profile" + (p.name === (s.serverProfile || s.profilesData.active) ? " active" : "")}
                                onClick={() => void switchProfile(p.name)}>
                          <div className="dd-p-name">{p.name}{p.name === (s.serverProfile || s.profilesData.active) ? <span className="dd-active"> ●</span> : null}</div>
                          <div className="dd-p-model">{p.model_provider}/{p.model_name}</div>
                        </button>
                      ))}
                      <button className="dd-manage" onClick={() => { setState({ modelDDOpen: false }); void primeSettings("profiles"); }}>{S.chat.manageProfiles}</button>
                    </div>
                  )}
                </div>
                <button className={"act-btn " + (showStop ? "stop" : "send")}
                        title={showStop ? S.chat.stop : S.chat.send}
                        disabled={compacting}
                        onClick={() => showStop ? stopGen() : void submit()}>
                  {showStop ? <IconStop /> : <IconSend />}
                </button>
              </div>
            </div>
          </div>
        </div>
        </div>
      </div>
      <FilesPanel root={workspace} panel={files} />
    </>
  );
}

function MessageView({ idx, m, sessionId, workspace, onOpenFile, onOpenSkill, live, pendingToolCallId }: {
  idx: number; m: ChatMsg; sessionId: string | null; workspace: string; onOpenFile: (path: string) => void;
  onOpenSkill: (name: string) => void; live?: boolean; pendingToolCallId?: string;
}) {
  const S = useStrings();
  const skill = m.role === "user"
    ? (m.skill || parseSkillTurn(m.content, invocableSkills(getState().pluginsData.skills || [])))
    : null;
  if (m.role === "user" && m.steer) {
    return (
      <div id={"msg-" + idx} className="m m-steer-wrap">
        <SteerChip text={m.content} ts={m.ts} onOpenSkill={onOpenSkill} />
      </div>
    );
  }
  if (!live) settleWork(m);
  const segs = m.role === "assistant" ? groupParts(partsOf(m)) : null;
  return (
    <div id={"msg-" + idx} className={"m " + (m.role === "user" ? "m-u" : "m-a") + (live ? " streaming" : "")}>
      <div className="msg-stack">
        {segs ? segs.map((seg, i) => {
          const isLast = !!live && i === segs.length - 1;
          if (seg.type === "work") {
            return <WorkGroup key={i} items={seg.items} isLast={isLast} live={!!live} pendingToolCallId={pendingToolCallId} />;
          }
          if (seg.part.kind === "steer") {
            return <SteerChip key={i} text={seg.part.text} ts={seg.part.ts} onOpenSkill={onOpenSkill} />;
          }
          const text = seg.part.kind === "text" ? seg.part.text : "";
          if (!text && !isLast) return null;
          return (
            <div className={"bub" + (live && isLast ? " streaming-bub" : "")} key={i}>
              <Suspense fallback={<span className="markdown-fallback">{text}</span>}>
                <ChatMarkdown text={text} streaming={!!live && isLast} />
              </Suspense>
              {live && isLast ? <span className="stream-caret" /> : null}
            </div>
          );
        }) : (
          (m.content || m.previews?.length || !m.error) && (
            <div className="bub">
              {!!m.previews?.length && (
                <div className="msg-previews">
                  {m.previews.map((src, i) => <img key={i} src={src} alt="" />)}
                </div>
              )}
              {skill
                ? <SkillTurn name={skill.name} args={skill.args} source={m.content} onOpen={onOpenSkill} />
                : m.content}
            </div>
          )
        )}
        <MessageFilesCard
          text={m.content || ""}
          steps={m.steps}
          uploaded={m.files}
          workspace={workspace || null}
          onOpenFile={onOpenFile}
          live={!!live}
        />
        {m.aborted && <div className="msg-note">{S.chat.aborted}</div>}
        {m.error && (
          <div className="msg-error">
            {S.chat.error(m.error)}
            {` - ${formatErrorStamp(m.errorTs ?? m.ts)}`}
          </div>
        )}
        <MessageFooter m={m} live={!!live} sessionId={sessionId} idx={idx} />
      </div>
    </div>
  );
}

function MessageFooter({ m, live, sessionId, idx }: {
  m: ChatMsg; live?: boolean; sessionId: string | null; idx: number;
}) {
  const S = useStrings();
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  if (live) return null;
  const copy = () => {
    // Always the original message. A rendered skill chip is a view, not the text
    // the user sent — copying it would silently lose the SKILL.md expansion.
    const text = m.content;
    if (!text) return;
    void navigator.clipboard.writeText(text).then(() => showToast(S.common.copied));
  };
  const copyBtn = (
    <button type="button" className="msg-act" title={S.common.copy} onClick={copy}>
      <IconCopy />
    </button>
  );
  if (m.role === "user") {
    return (
      <div className="msg-footer user">
        <span title={S.chat.msgTimeTip}>{S.chat.msgStamp(m.ts || Date.now())}</span>
        {copyBtn}
      </div>
    );
  }
  const durationMs = m.durationMs ?? ((m.durationSec || 0) * 1000);
  const tps = (m.llmMs && m.tokOut)
    ? m.tokOut / (m.llmMs / 1000)
    : ((m.durationSec && m.tokOut) ? m.tokOut / m.durationSec : 0);
  const hasStats = m.tokIn != null;
  const usageTip = [
    S.chat.usageTip,
    `↑ ${S.chat.ctxInput} ${fmtN(m.tokIn || 0)}`,
    m.cacheRead != null
      ? `${S.chat.cacheTokensTip} ${fmtN(m.cacheRead)}${m.cacheHitPercent != null ? ` (${S.chat.cacheHitTip} ${m.cacheHitPercent.toFixed(1)}%)` : ""}`
      : "",
    `↓ ${S.chat.ctxOutput} ${fmtN(m.tokOut || 0)}`,
    tps > 0 ? `${S.chat.tokPerSecTip} ${fmtTps(tps)}` : "",
    (m.costUsd || 0) > 0 ? `${S.chat.costUsdTip} ${fmtCost(m.costUsd || 0)}` : "",
  ].filter(Boolean).join("\n");
  const rating = m.feedback?.rating || null;
  const canAct = !!sessionId;
  const rate = (next: "up" | "down" | null, comment = "") => {
    if (!sessionId) return;
    void rateMessage(sessionId, idx, next, comment).then((ok) => {
      if (ok && next) showToast(S.chat.feedbackSaved);
    });
  };
  return (
    <div className="msg-footer">
      {copyBtn}
      {canAct && (
        <>
          <button type="button" className={"msg-act" + (rating === "up" ? " on" : "")} title={S.chat.like}
                  aria-pressed={rating === "up"}
                  onClick={() => rate(rating === "up" ? null : "up")}>
            <IconThumbUp />
          </button>
          <button type="button" className={"msg-act" + (rating === "down" ? " on" : "")} title={S.chat.dislike}
                  aria-pressed={rating === "down"}
                  onClick={() => (rating === "down" ? rate(null) : setFeedbackOpen(true))}>
            <IconThumbDown />
          </button>
          <button type="button" className="msg-act" title={S.chat.fork}
                  onClick={() => void forkAt(sessionId!, idx)}>
            <IconFork />
          </button>
        </>
      )}
      {hasStats && (
        <span className="msg-usage" title={usageTip}>
          <IconDatabase />
          {S.chat.usageTokens(fmtN((m.tokIn || 0) + (m.tokOut || 0)))}
        </span>
      )}
      {!!durationMs && <span title={S.chat.durationTip}>{fmtDurationMs(durationMs)}</span>}
      <span title={S.chat.msgTimeTip}>{S.chat.msgStamp(m.ts || Date.now())}</span>
      {feedbackOpen && (
        <FeedbackDialog
          initial={m.feedback?.comment || ""}
          onClose={() => setFeedbackOpen(false)}
          onSubmit={(comment) => { setFeedbackOpen(false); rate("down", comment.trim()); }}
        />
      )}
    </div>
  );
}

function parseApprovalRequest(data: any): ApprovalRequest | null {
  const toolCallId = String(data?.tool_call_id || "");
  if (!toolCallId) return null;
  const rawOpts = Array.isArray(data.options) ? data.options : ["allow", "allow_prefix", "deny", "deny_prefix"];
  const options = rawOpts.filter(
    (x: unknown): x is ApprovalDecision =>
      x === "allow" || x === "allow_prefix" || x === "deny" || x === "deny_prefix",
  );
  const args = (data.args && typeof data.args === "object" && !Array.isArray(data.args))
    ? data.args
    : ((data.arguments && typeof data.arguments === "object" && !Array.isArray(data.arguments))
      ? data.arguments
      : {});
  return {
    toolCallId,
    name: String(data.name || ""),
    args,
    question: String(data.question || ""),
    preview: String(data.preview || ""),
    similarLabel: String(data.similar_label || ""),
    options: options.length ? options : ["allow", "allow_prefix", "deny", "deny_prefix"],
  };
}

async function decideCurrentApproval(decision: ApprovalDecision) {
  const st = getState();
  const sessId = st.curSess || "";
  const live = st.streams[sessId];
  const req = live?.pendingApprovals?.[0];
  if (!sessId || !live || !req || live.decidingApproval) return;
  live.decidingApproval = true;
  bump();
  const res = await api.postSessionApproval(sessId, req.toolCallId, decision);
  const again = getState().streams[sessId];
  if (again) again.decidingApproval = false;
  if (res.ok || res.status === 404) {
    dequeueApproval(sessId, req.toolCallId);
  } else {
    bump();
    showToast(getStrings().chat.approvalFailed);
  }
}

async function stopGen() {
  const st = getState();
  const sessId = st.curSess || "";
  const stream = st.streams[sessId];
  if (!stream || stream.cancelling) return;
  stream.userStopped = true;
  stream.cancelling = true;
  bump();
  try {
    if (stream.runId) await api.cancelRunApi(stream.runId);
    else await api.cancelChatApi(sessId);
  } finally {
    stream.abortCtrl.abort();
  }
}

/** Coalesce stream paints: rAF, but never denser than 120ms so a fast
 *  answer does not reparse growing markdown every token (that is the
 *  flash). Structural events (tool start/end) call flushStream and paint now. */
const BUMP_MIN_INTERVAL_MS = 120;
let pageUnloading = false;
let streamRaf = 0;
let streamThrottle = 0;
let lastBumpAt = 0;
function bumpStream() {
  if (streamRaf || streamThrottle) return;
  const commit = () => {
    streamRaf = 0;
    streamThrottle = 0;
    lastBumpAt = Date.now();
    bump();
  };
  const wait = lastBumpAt + BUMP_MIN_INTERVAL_MS - Date.now();
  if (wait > 0) {
    streamThrottle = window.setTimeout(commit, wait);
    return;
  }
  streamRaf = requestAnimationFrame(commit);
}
function flushStream() {
  if (streamRaf) {
    cancelAnimationFrame(streamRaf);
    streamRaf = 0;
  }
  if (streamThrottle) {
    window.clearTimeout(streamThrottle);
    streamThrottle = 0;
  }
  lastBumpAt = Date.now();
  bump();
}

/** Enter always accepts input: while a turn or /goal is running, plain text
 *  steers the current run; attachments and slash commands queue until it ends.
 *  ``/compact`` holds the same session lock as a chat turn, so a second send
 *  409s. While it runs we block rather than queue. */
function isCompacting(sessId: string) {
  return !!getState().commandRuns[sessId];
}

function isBusy(sessId: string) {
  const st = getState();
  return !!st.streams[sessId] || !!st.goalRuns[sessId] || !!st.commandRuns[sessId];
}

function enqueueMessage(sessId: string, text: string, files: File[], opts?: { atFront?: boolean; silent?: boolean }) {
  const item = { id: uid(), sessionId: sessId, text, files, ts: Date.now() };
  const rest = getState().messageQueue;
  setState({ messageQueue: opts?.atFront ? [item, ...rest] : [...rest, item] });
  if (!opts?.silent && !opts?.atFront) showToast(getStrings().chat.queuedToast);
}

async function requeueLateSteer(sessId: string) {
  const parked = await api.takeSteerApi(sessId);
  const late = (parked.ok && parked.data?.messages) ? parked.data.messages : [];
  for (let i = late.length - 1; i >= 0; i--) {
    enqueueMessage(sessId, late[i], [], { atFront: true, silent: true });
  }
}

function editQueued(qItem: QueuedMessage) {
  setState({
    inputText: qItem.text,
    pendingFiles: [...getState().pendingFiles, ...qItem.files],
    messageQueue: getState().messageQueue.filter((x) => x.id !== qItem.id),
  });
}

function appendSteerBubble(sessId: string, text: string) {
  const live = getState().streams[sessId]?.aiMsg;
  if (live) {
    appendSteerPart(live, text);
    saveSessions();
    bump();
    return;
  }
  const sess = getState().sessions[sessId];
  if (!sess) return;
  sess.msgs.push({ role: "user", content: text, ts: Date.now(), steer: true });
  saveSessions();
  bump();
}

async function applySteer(sessId: string, text: string): Promise<boolean> {
  const S = getStrings();
  const { ok, data } = await api.steerChatApi(sessId, text);
  if (ok && data?.accepted) {
    appendSteerBubble(sessId, text);
    showToast(S.chat.interruptToast);
    return true;
  }
  enqueueMessage(sessId, text, [], { silent: true });
  showToast(S.chat.interruptQueued);
  return false;
}

function setGoalRun(sessId: string, patch: { status: string; objective: string; progress: string }) {
  setState({ goalRuns: { ...getState().goalRuns, [sessId]: patch } });
}

function clearGoalRun(sessId: string) {
  const next = { ...getState().goalRuns };
  delete next[sessId];
  setState({ goalRuns: next });
}

function ensureSession(): string | null {
  const st = getState();
  if (!st.curSess) {
    const dir = st.pendingNewChatDir || st.serverDir;
    if (!dir) { setState({ dirModal: { open: true, forNewSession: true, value: "" } }); return null; }
    createSession(dir);
  }
  return getState().curSess;
}

async function submit() {
  const st = getState();
  if (st.curSess && st.streams[st.curSess]?.pendingApprovals?.[0]) return;
  const text = st.inputText.trim();
  const files = st.pendingFiles.slice();
  const S = getStrings();

  if (st.goalCompose) {
    const budget = parseTokenBudget(st.goalCompose.budgetText);
    if (budget === null) {
      showToast(S.chat.goalBudgetInvalid);
      return;
    }
    if (!text) {
      showToast(S.chat.goalNeedObjective);
      return;
    }
    const sessId = ensureSession();
    if (!sessId) return;
    if (isCompacting(sessId)) {
      showToast(S.chat.compactWait);
      return;
    }
    setState({ inputText: "", pendingFiles: [], goalCompose: null });
    if (isBusy(sessId)) {
      enqueueMessage(sessId, `/goal ${text}`, files);
      return;
    }
    await runGoalCmd(sessId, text, budget);
    return;
  }

  if (!text && !files.length) {
    const sessId = st.curSess;
    if (sessId && isBusy(sessId) && !isCompacting(sessId)) {
      await stopGen();
    }
    return;
  }
  const cmd = parseWebCommand(text);
  if (cmd?.kind === "goal") {
    setState({ inputText: cmd.arg, pendingFiles: [], goalCompose: { budgetText: "" } });
    return;
  }
  if (cmd?.kind === "queue") {
    if (!cmd.arg && !files.length) {
      showToast(S.chat.queueNeedPrompt);
      setState({ inputText: "/queue " });
      return;
    }
    const sessId = ensureSession();
    if (!sessId) return;
    setState({ inputText: "", pendingFiles: [] });
    if (isBusy(sessId)) {
      enqueueMessage(sessId, cmd.arg, files);
      return;
    }
    await sendMessage(sessId, cmd.arg, files);
    return;
  }
  const sessId = ensureSession();
  if (!sessId) return;
  if (isCompacting(sessId)) {
    showToast(S.chat.compactWait);
    return;
  }
  setState({ inputText: "", pendingFiles: [] });
  if (isBusy(sessId)) {
    const live = getState().streams[sessId];
    if (live?.cancelling || live?.userStopped) {
      enqueueMessage(sessId, text, files);
      return;
    }
    if (!files.length && !cmd) {
      await applySteer(sessId, text);
      return;
    }
    enqueueMessage(sessId, text, files);
    return;
  }
  if (cmd) {
    await runCompact(sessId, cmd.arg);
    return;
  }
  await sendMessage(sessId, text, files);
}

async function drainQueue(sessId: string) {
  if (isBusy(sessId)) return;
  const next = getState().messageQueue.find((q) => q.sessionId === sessId);
  if (!next) return;
  setState({ messageQueue: getState().messageQueue.filter((q) => q.id !== next.id) });
  const cmd = parseWebCommand(next.text);
  if (cmd) {
    if (cmd.kind === "compact") await runCompact(sessId, cmd.arg);
    else if (cmd.kind === "goal") await runGoalCmd(sessId, cmd.arg);
    else await sendMessage(sessId, cmd.arg, next.files);
    return;
  }
  await sendMessage(sessId, next.text, next.files);
}

function markChatError(m: ChatMsg, text: string): void {
  m.error = text;
  m.errorTs = Date.now();
}

function applyLiveSseEvent(aiMsg: ChatMsg, evt: any, sessId?: string): boolean {
  if (evt.event === "thinking") {
    appendThink(aiMsg, evt.data);
    bumpStream();
    return true;
  }
  if (evt.event === "tool_call") {
    appendTool(aiMsg, evt.data.name, JSON.stringify(evt.data.args || {}), evt.data.tool_call_id);
    flushStream();
    return true;
  }
  if (evt.event === "tool_result") {
    const callId = unfinishedToolCallId(aiMsg, evt.data.tool_call_id);
    finishTool(aiMsg, evt.data.result, evt.data.diff, evt.data.tool_call_id);
    if (sessId && callId) dequeueApproval(sessId, callId);
    flushStream();
    return true;
  }
  if (evt.event === "approval_request" && sessId) {
    const req = parseApprovalRequest(evt.data);
    if (req) enqueueApproval(sessId, req);
    flushStream();
    return true;
  }
  if (evt.event === "content") {
    appendText(aiMsg, evt.data);
    bumpStream();
    return true;
  }
  if (evt.event === "error") {
    settleWork(aiMsg, Date.now());
    markChatError(aiMsg, String(evt.data));
    flushStream();
    return true;
  }
  if (evt.event === "aborted") {
    settleWork(aiMsg, Date.now());
    aiMsg.aborted = true;
    flushStream();
    return true;
  }
  return false;
}

function isDisconnectErr(e: any): boolean {
  if (!e) return false;
  if (e.name === "AbortError") return true;
  const msg = String(e.message || e).toLowerCase();
  return msg.includes("network") || msg.includes("failed to fetch") || msg.includes("load failed");
}

function takeLiveAssistant(sess: { msgs: ChatMsg[] }): ChatMsg {
  const last = sess.msgs[sess.msgs.length - 1];
  if (last?.role === "assistant") {
    const err = (last.error || "").toLowerCase();
    const disconnected = err.includes("network") || err.includes("failed to fetch") || err.includes("load failed");
    const parts = last.parts || [];
    const unfinished = parts.some(
      (p) => (p.kind === "think" && p.ms == null) || (p.kind === "tool" && p.result == null),
    );
    const blank = !last.content && !parts.length && !last.aborted && last.tokIn == null;
    if (disconnected || unfinished || blank) {
      last.content = "";
      last.parts = [];
      last.steps = [];
      delete last.error;
      last.aborted = false;
      return last;
    }
  }
  const aiMsg: ChatMsg = { role: "assistant", content: "", steps: [], parts: [], ts: Date.now(), durationSec: 0 };
  sess.msgs.push(aiMsg);
  return aiMsg;
}

async function consumeSse(
  resp: Response,
  sessId: string,
  sess: any,
  aiMsg: ChatMsg,
  t0: number,
  onStatus?: (d: any) => void,
  afterSeq = 0,
): Promise<{ lastSeq: number; terminal: boolean }> {
  const reader = resp.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let lastSeq = afterSeq;
  let terminal = false;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6);
      if (raw === "[DONE]") {
        terminal = true;
        continue;
      }
      let evt: any;
      try { evt = JSON.parse(raw); } catch { continue; }
      if (typeof evt.seq === "number") {
        if (evt.seq <= lastSeq) continue;
        lastSeq = evt.seq;
        const live = getState().streams[sessId];
        if (live) live.lastSeq = lastSeq;
      }
      if (evt.event === "done" || evt.event === "error" || evt.event === "aborted") {
        terminal = true;
      }
      if (applyLiveSseEvent(aiMsg, evt, sessId)) continue;
      if (evt.event === "status" && evt.data && onStatus) onStatus(evt.data);
      if (evt.event === "done" && evt.data) {
        const prevCost = sess.costUsd || 0;
        const turn = evt.data.turn_usage || {};
        aiMsg.durationSec = (typeof evt.data.duration_ms === "number")
          ? evt.data.duration_ms / 1000
          : (evt.data.response_time || (performance.now() - t0) / 1000);
        if (typeof evt.data.duration_ms === "number") aiMsg.durationMs = evt.data.duration_ms;
        if (typeof evt.data.llm_ms === "number") aiMsg.llmMs = evt.data.llm_ms;
        if (turn.input_tokens != null || evt.data.input_tokens != null) {
          aiMsg.tokIn = (turn.input_tokens ?? evt.data.input_tokens) || 0;
          aiMsg.tokOut = (turn.output_tokens ?? evt.data.output_tokens) || 0;
          aiMsg.cacheRead = turn.cache_read_tokens ?? evt.data.cache_read_tokens ?? 0;
          aiMsg.cacheHitPercent = turn.cache_hit_percent ?? evt.data.cache_hit_percent ?? null;
          sess.tokIn += evt.data.input_tokens || 0;
          sess.tokOut += evt.data.output_tokens || 0;
          sess.tokTotal += evt.data.total_tokens || 0;
          sess.requests += evt.data.requests || 1;
          sess.totalTime += evt.data.response_time || 0;
          sess.lastInputTokens = evt.data.input_tokens || sess.lastInputTokens;
        }
        if (evt.data.context_window) setState({ serverContextWindow: evt.data.context_window });
        if (evt.data.usage) applySessionUsage(sessId, evt.data.usage);
        const nextCost = getState().sessions[sessId]?.costUsd ?? sess.costUsd;
        if (typeof turn.cost_usd === "number") aiMsg.costUsd = turn.cost_usd;
        else if (typeof nextCost === "number") aiMsg.costUsd = Math.max(0, (nextCost || 0) - prevCost);
        if (evt.data.media_notes?.length) {
          aiMsg.content = (evt.data.media_notes as string[]).join("\n") + (aiMsg.content ? "\n\n" + aiMsg.content : "");
        }
        if (!aiMsg.content && evt.data.content) appendText(aiMsg, evt.data.content);
      }
    }
  }
  return { lastSeq, terminal };
}

async function finishLive(sessId: string, aiMsg: ChatMsg, t0: number, disconnected: boolean) {
  flushStream();
  settleWork(aiMsg, Date.now());
  if (!aiMsg.durationSec) aiMsg.durationSec = (performance.now() - t0) / 1000;
  delete getState().streams[sessId];
  const sess = getState().sessions[sessId];
  if (sess) sess.running = false;
  if (getState().curSess !== sessId) {
    if (sess) sess.unread = true;
  } else {
    markSessionRead(sessId);
  }
  saveSessions();
  bump();
  if (disconnected || pageUnloading) return;
  await requeueLateSteer(sessId);
  await drainQueue(sessId);
}

async function watchRun(
  sessId: string,
  sess: any,
  aiMsg: ChatMsg,
  t0: number,
  runId: string,
  abortCtrl: AbortController,
  onStatus?: (d: any) => void,
): Promise<boolean> {
  let lastSeq = getState().streams[sessId]?.lastSeq || 0;
  for (let attempt = 0; attempt < 20; attempt++) {
    const live = getState().streams[sessId];
    if (live) live.reconnecting = attempt > 0;
    if (attempt > 0) bump();
    try {
      const resp = await api.runEvents(runId, lastSeq, abortCtrl.signal);
      if (!resp.ok || !resp.body) {
        if (resp.status === 404) return false;
        let detail = `HTTP ${resp.status}`;
        try {
          const j = await resp.json();
          if (j?.detail) detail = String(j.detail);
        } catch { /* not json */ }
        markChatError(aiMsg, detail);
        return false;
      }
      if (live) live.reconnecting = false;
      const got = await consumeSse(resp, sessId, sess, aiMsg, t0, onStatus, lastSeq);
      lastSeq = got.lastSeq;
      if (got.terminal) return false;
      if (getState().streams[sessId]?.userStopped || abortCtrl.signal.aborted) {
        aiMsg.aborted = !!getState().streams[sessId]?.userStopped;
        return false;
      }
      if (pageUnloading) return true;
      await new Promise((r) => setTimeout(r, Math.min(4000, 300 * (attempt + 1))));
    } catch (e: any) {
      const stopped = getState().streams[sessId]?.userStopped;
      if (stopped) {
        aiMsg.aborted = true;
        return false;
      }
      if (pageUnloading) return true;
      if (!isDisconnectErr(e)) {
        markChatError(aiMsg, String(e?.message || e));
        return false;
      }
      await new Promise((r) => setTimeout(r, Math.min(4000, 300 * (attempt + 1))));
    }
  }
  return true;
}

async function resumeLiveStream(sessId: string) {
  if (getState().streams[sessId]) return;
  const sess = getState().sessions[sessId];
  if (!sess) return;
  const { ok, data } = await api.fetchActiveRun(sessId);
  const run = (ok && data?.run) ? data.run : null;
  if (!run) {
    if (sess) sess.running = false;
    if (looksDisconnectedMsg(sess.msgs[sess.msgs.length - 1])) {
      await hydrateSession(sessId, true);
    }
    if (getState().curSess === sessId) markSessionRead(sessId);
    else saveSessions();
    bump();
    return;
  }
  const aiMsg = takeLiveAssistant(sess);
  appendThink(aiMsg, "");
  const abortCtrl = new AbortController();
  getState().streams[sessId] = { abortCtrl, aiMsg, runId: run.run_id, lastSeq: 0 };
  bump();
  const t0 = performance.now();
  const disconnected = await watchRun(sessId, sess, aiMsg, t0, run.run_id, abortCtrl);
  await finishLive(sessId, aiMsg, t0, disconnected);
}

function looksDisconnectedMsg(m: ChatMsg | undefined): boolean {
  if (!m || m.role !== "assistant" || !m.error) return false;
  const err = m.error.toLowerCase();
  return err.includes("network") || err.includes("failed to fetch") || err.includes("load failed");
}

async function sendMessage(sessId: string, text: string, files: File[]) {
  const st = getState();
  const sess = st.sessions[sessId];
  if (!sess) return;
  const S = getStrings();

  const userMsg: ChatMsg = { role: "user", content: text, ts: Date.now(), files: [], previews: [] };
  const abortCtrl = new AbortController();
  const aiMsg: ChatMsg = { role: "assistant", content: "", steps: [], parts: [], ts: Date.now(), durationSec: 0 };
  appendThink(aiMsg, "");
  sess.msgs.push(userMsg);
  sess.msgs.push(aiMsg);
  st.streams[sessId] = { abortCtrl, aiMsg, preparing: files.length > 0 };
  if (!sess.named && sess.msgs.filter((m) => m.role === "user").length === 1) sess.title = (text || "Chat").slice(0, 50);
  saveSessions();
  bump();

  const t0 = performance.now();
  let disconnected = false;
  try {
    let message = text;
    const uploaded: string[] = [];
    const imageFiles = files.filter((f) => f.type.startsWith("image/"));
    const otherFiles = files.filter((f) => !f.type.startsWith("image/"));
    const previews: string[] = [];
    for (const f of imageFiles) {
      previews.push(await fileToDataUrl(f));
    }
    const images = await Promise.all(imageFiles.map(fileToImagePayload));
    for (const f of otherFiles) {
      const up = await api.uploadFileApi(f, sess.dir || st.serverDir);
      if (up.ok && up.data?.path) uploaded.push(up.data.path);
      else showToast(S.chat.uploadFailed(f.name), 3000);
    }
    if (uploaded.length) {
      const list = uploaded.join(", ");
      if (!message) message = S.chat.uploadedFiles(list);
      else message += "\n\n" + S.chat.attachmentTag(list);
    }
    userMsg.content = text || message;
    userMsg.files = uploaded;
    userMsg.previews = previews;
    if (!message && !images.length) {
      markChatError(aiMsg, S.chat.uploadFailed(files[0]?.name || "file"));
      return;
    }
    const live = getState().streams[sessId];
    if (live) live.preparing = false;
    bump();
    if (abortCtrl.signal.aborted) {
      aiMsg.aborted = true;
      return;
    }

    const created = await api.createChatRunApi({
      message,
      session_id: sessId,
      work_dir: sess.dir || st.serverDir || "",
      approval_mode: st.selectedApprovalMode,
      images,
    });
    if (!created.ok || !created.data?.run_id) {
      const detail = (created.data && (created.data as any).detail) || `HTTP ${created.status}`;
      markChatError(aiMsg, String(detail));
      return;
    }
    if (live) live.runId = created.data.run_id;
    markSessionRead(sessId);
    pushPendingName(sessId);
    if (getState().streams[sessId]?.userStopped || abortCtrl.signal.aborted) {
      await api.cancelRunApi(created.data.run_id);
      aiMsg.aborted = true;
      return;
    }
    disconnected = await watchRun(sessId, sess, aiMsg, t0, created.data.run_id, abortCtrl);
  } catch (e: any) {
    const stopped = getState().streams[sessId]?.userStopped;
    if (stopped) aiMsg.aborted = true;
    else if (isDisconnectErr(e) || pageUnloading) disconnected = true;
    else markChatError(aiMsg, String(e?.message || e));
  } finally {
    await finishLive(sessId, aiMsg, t0, disconnected);
  }
}

function parseWebCommand(text: string): { kind: "compact" | "goal" | "queue"; arg: string } | null {
  if (text === "/compact" || text.startsWith("/compact ")) {
    return { kind: "compact", arg: text.slice("/compact".length).trim() };
  }
  if (text === "/goal" || text.startsWith("/goal ")) {
    return { kind: "goal", arg: text.slice("/goal".length).trim() };
  }
  if (text === "/queue" || text.startsWith("/queue ")) {
    return { kind: "queue", arg: text.slice("/queue".length).trim() };
  }
  if (text === "/q" || text.startsWith("/q ")) {
    return { kind: "queue", arg: text.slice("/q".length).trim() };
  }
  return null;
}

function pickSlash(it: SlashItem, taRef: { current: HTMLTextAreaElement | null }) {
  if (it.cmd === "/goal") {
    setState({ inputText: "", goalCompose: { budgetText: "" } });
    return;
  }
  if (it.cmd === "/compact") {
    const st = getState();
    if (st.curSess && isCompacting(st.curSess)) {
      showToast(getStrings().chat.compactWait);
      return;
    }
    setState({ inputText: "", goalCompose: null });
    void (async () => {
      const stNow = getState();
      if (!stNow.curSess) {
        const dir = stNow.pendingNewChatDir || stNow.serverDir;
        if (!dir) { setState({ dirModal: { open: true, forNewSession: true, value: "" } }); return; }
        createSession(dir);
      }
      const sessId = getState().curSess!;
      if (isBusy(sessId)) {
        enqueueMessage(sessId, "/compact", []);
        return;
      }
      await runCompact(sessId, "");
    })();
    return;
  }
  setState({ inputText: it.cmd + " " });
  requestAnimationFrame(() => taRef.current?.focus());
}

function beginCommand(sessId: string, userText: string) {
  const sess = getState().sessions[sessId];
  if (!sess) return null;
  sess.msgs.push({ role: "user", content: userText, ts: Date.now() });
  setState({ commandRuns: { ...getState().commandRuns, [sessId]: { kind: "compact" } } });
  saveSessions();
  markSessionRead(sessId);
  return sess;
}

function endCommand(sessId: string) {
  const next = { ...getState().commandRuns };
  delete next[sessId];
  const sess = getState().sessions[sessId];
  if (sess) sess.running = false;
  if (getState().curSess !== sessId) {
    if (sess) sess.unread = true;
  } else {
    markSessionRead(sessId);
  }
  saveSessions();
  setState({ commandRuns: next });
}

async function runCompact(sessId: string, instructions: string) {
  const S = getStrings();
  if (isCompacting(sessId)) return;
  const userText = instructions ? `/compact ${instructions}` : "/compact";
  const sess = beginCommand(sessId, userText);
  if (!sess) return;
  const aiMsg: ChatMsg = { role: "assistant", content: "", ts: Date.now(), durationSec: 0 };
  try {
    const { ok, data, status } = await api.compactSessionApi(sessId, instructions);
    if (!ok) {
      const detail = (data && (data.detail || data.error)) || `HTTP ${status}`;
      markChatError(aiMsg, S.chat.compactFailed(String(detail)));
    } else {
      aiMsg.content = S.chat.compactOk(data.messages_before, data.messages_after);
      if (data.usage) applySessionUsage(sessId, data.usage);
    }
  } catch (e: any) {
    markChatError(aiMsg, S.chat.compactFailed(String(e?.message || e)));
  } finally {
    sess.msgs.push(aiMsg);
    endCommand(sessId);
    await drainQueue(sessId);
  }
}

async function runGoalCmd(sessId: string, objective: string, tokenBudget = UNLIMITED_TOKEN_BUDGET) {
  const S = getStrings();
  if (!objective || objective === "status") {
    showToast(S.chat.goalNeedObjective);
    setState({ inputText: "", goalCompose: { budgetText: "" } });
    await drainQueue(sessId);
    return;
  }
  const sess = getState().sessions[sessId];
  if (!sess) return;
  sess.msgs.push({ role: "user", content: `/goal ${objective}`, ts: Date.now() });
  const abortCtrl = new AbortController();
  const aiMsg: ChatMsg = { role: "assistant", content: "", steps: [], parts: [], ts: Date.now(), durationSec: 0 };
  appendThink(aiMsg, "");
  getState().streams[sessId] = { abortCtrl, aiMsg };
  sess.msgs.push(aiMsg);
  saveSessions();
  setGoalRun(sessId, { status: "active", objective, progress: "" });
  markSessionRead(sessId);
  bump();
  const t0 = performance.now();
  let disconnected = false;
  try {
    const resp = await api.streamGoal({
      objective,
      session_id: sessId,
      token_budget: tokenBudget,
    }, abortCtrl.signal);
    if (!resp.ok || !resp.body) {
      let detail = `HTTP ${resp.status}`;
      try {
        const j = await resp.json();
        if (j?.detail) detail = String(j.detail);
      } catch { /* not json */ }
      markChatError(aiMsg, detail);
      return;
    }
    const found = await api.fetchActiveRun(sessId);
    const live = getState().streams[sessId];
    if (live && found.ok && found.data?.run) live.runId = found.data.run.run_id;
    if (getState().streams[sessId]?.userStopped || abortCtrl.signal.aborted) {
      const runId = getState().streams[sessId]?.runId;
      if (runId) await api.cancelRunApi(runId);
      else await api.cancelChatApi(sessId);
      aiMsg.aborted = true;
      return;
    }
    const onStatus = (d: any) => {
      setGoalRun(sessId, {
        status: d.status || "active",
        objective: d.objective || objective,
        progress: d.progress || "",
      });
    };
    const got = await consumeSse(resp, sessId, sess, aiMsg, t0, onStatus);
    if (!got.terminal) {
      const runId = getState().streams[sessId]?.runId;
      if (runId) {
        disconnected = await watchRun(sessId, sess, aiMsg, t0, runId, abortCtrl, onStatus);
      }
    }
  } catch (e: any) {
    const stopped = getState().streams[sessId]?.userStopped;
    if (stopped) {
      aiMsg.aborted = true;
      const runId = getState().streams[sessId]?.runId;
      if (runId) await api.cancelRunApi(runId);
      else await api.cancelChatApi(sessId);
    }
    else if (pageUnloading) disconnected = true;
    else if (isDisconnectErr(e)) {
      const runId = getState().streams[sessId]?.runId;
      if (runId) {
        disconnected = await watchRun(sessId, sess, aiMsg, t0, runId, abortCtrl, (d) => {
          setGoalRun(sessId, {
            status: d.status || "active",
            objective: d.objective || objective,
            progress: d.progress || "",
          });
        });
      } else {
        disconnected = true;
      }
    } else markChatError(aiMsg, String(e?.message || e));
  } finally {
    clearGoalRun(sessId);
    await finishLive(sessId, aiMsg, t0, disconnected);
  }
}

function imageFilesFromClipboard(data: DataTransfer | null): File[] {
  if (!data) return [];
  const out: File[] = [];
  for (const item of Array.from(data.items || [])) {
    if (item.kind === "file" && item.type.startsWith("image/")) {
      const f = item.getAsFile();
      if (f) out.push(f);
    }
  }
  return out;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function fileToImagePayload(file: File): Promise<{ mime: string; data: string }> {
  const url = await fileToDataUrl(file);
  const comma = url.indexOf(",");
  return { mime: file.type || "image/png", data: comma >= 0 ? url.slice(comma + 1) : url };
}

function PendingFileChip({ file, onRemove }: { file: File; onRemove: () => void }) {
  const [src, setSrc] = useState("");
  useEffect(() => {
    if (!file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  return (
    <div className={"file-chip" + (src ? " img" : "")}>
      {src ? <img src={src} alt={file.name} /> : <span>{file.name}</span>}
      <span className="fx" onClick={onRemove}><IconClose /></span>
    </div>
  );
}

