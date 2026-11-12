import { lazy, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { Link } from "react-router";
import * as api from "../api";
import { ChatNav } from "../components/ChatNav";
import { ComposerDir } from "../components/ComposerDir";
import { SteerChip, WorkGroup, ApprovalCard } from "../components/WorkGroup";
import { RenameInput } from "../components/SessionTree";
import { CommandMenu, ComposerPicker, invocableSkills, slashQuery, type PickerGroup } from "../components/SlashMenu";
import { flatItems, plusMenuGroups, slashMenuGroups, type MenuItem } from "../lib/composerCommands";
import { CollapsibleUserText, FeedbackDialog, SkillTurn } from "../components/MessageExtras";
import { TurnUsagePanel } from "../components/TurnUsagePanel";
import { TodoDock } from "../components/TaskCard";
import { AutomatedNotice, ScheduledRunCard } from "../components/AutomatedNotice";
import { dockTodos } from "../lib/workActivity";
import { leadingSkillToken, parseSkillTurn } from "../lib/skillInvocation";
import { getStrings, useStrings } from "../i18n";
import {
  fmtDurationMs, fmtN, fmtRunDuration, formatErrorStamp, parseTokenBudget,
  uid, UNLIMITED_TOKEN_BUDGET, updateQueuedMessageText,
} from "../lib/format";
import { primeSettings } from "../panels/SettingsModal";
import { applyPendingModel, loadSessionModel, pickSessionModel } from "../sessionModel";
import { createSession, forkAt, hydrateSession, loadOlderMessages, markSessionRead, pushPendingName, rateMessage, renameSession, sessionHydrating, syncSessionRoundStats } from "../sessions";
import { loadCronJobs, loadPlugins } from "../data";
import {
  Logo, IconPlus, IconClose, IconChat, IconBook, IconAsk, IconAuto, IconSidebarRight,
  IconAllowAll, IconSend, IconStop, IconCopy, IconArrowDown, IconPencil,
  IconChevronDown, IconThumbUp, IconThumbDown, IconFork, IconCompact, IconSpinner,
  IconCheck, IconTrash, IconQueue, IconSearch,
} from "../icons";
import {
  bump, dequeueApproval, enqueueApproval, getState, pushPrefs, saveSessions, sessionViews, setState,
  showToast, useAppState, type ApprovalDecision, type ApprovalRequest, type ChatMsg, type QueuedMessage,
} from "../store";
import { applySessionUsage, ContextUsageTip, prefetchSessionUsage, syncSessionUsage } from "../components/ContextUsageTip";
import { rememberSteersFromMsgs, removeUndeliveredSteers } from "../lib/hydrateMessages";
import {
  allAssistantTextOf, appendSteerPart, appendText, appendThink, appendTool, ensureTrailingFinalAnswer,
  finalAnswerOf, finishTool, foldProcess, groupParts, isProcessSegment, partsOf, settleWork, unfinishedToolCallId,
  workSummary,
} from "../lib/msgParts";
import { approvalKeyAction } from "../lib/approvalKeys";
import { createStreamFollow, stickToBottom } from "../lib/streamFollow";
import { applyChatSearchHighlights } from "../lib/sessionRows";
import { BackgroundJobs } from "../components/BackgroundJobs";
import { PreviewableImage } from "../components/ImageLightbox";
import { FilesPanel } from "../workspace/FilesPanel";
import { MessageFilesCard } from "../workspace/MessageFilesCard";
import type { FileEdits } from "../lib/fileDiff";
import { useFilesPanel } from "../workspace/useFilesPanel";
import { isImeKey } from "../lib/ime";

const ChatMarkdown = lazy(() => import("../components/ChatMarkdown").then((m) => ({ default: m.ChatMarkdown })));
/** Start fetching the previous page this far before the reader reaches the top. */
const OLDER_PREFETCH_PX = 800;
/** How long a spliced-in page may keep growing before the reader's position is let go. */
const PAGING_HOLD_MS = 1500;
const msgKeys = new WeakMap<ChatMsg, number>();
let nextMsgKey = 1;
function keyOfMsg(m: ChatMsg): string | number {
  if (m.uuid) return m.uuid;
  let k = msgKeys.get(m);
  if (!k) {
    k = nextMsgKey++;
    msgKeys.set(m, k);
  }
  return k;
}

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
  const msgsInnerRef = useRef<HTMLDivElement>(null);
  const chatWrapRef = useRef<HTMLDivElement>(null);
  const inputAreaRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(createStreamFollow());
  const activeSessRef = useRef<{ id: string | null; hasMsgs: boolean }>({ id: null, hasMsgs: false });
  const prevStreamingRef = useRef(false);
  const pagingAnchorRef = useRef<{ node: HTMLElement; top: number; head: ChatMsg | undefined } | null>(null);
  // After a page is spliced in above the viewport its messages keep growing for a
  // moment (lazy markdown, round stats). `.chat` has overflow-anchor:none, so hold
  // the reader's message in place until that settles or they scroll themselves.
  const pagingHoldRef = useRef<{ node: HTMLElement; top: number; until: number } | null>(null);
  const olderFailedRef = useRef(new Set<string>());
  const [showJump, setShowJump] = useState(false);
  const [slashActive, setSlashActive] = useState(0);
  const [plusOpen, setPlusOpen] = useState(false);
  const [plusActive, setPlusActive] = useState(0);
  const plusWrapRef = useRef<HTMLDivElement>(null);
  const plusMenuRef = useRef<HTMLDivElement>(null);
  const [ctxOpen, setCtxOpen] = useState(false);
  const [titleEditing, setTitleEditing] = useState(false);
  const [goalBudgetOpen, setGoalBudgetOpen] = useState(false);
  const modelWrapRef = useRef<HTMLDivElement>(null);
  const approvalWrapRef = useRef<HTMLDivElement>(null);
  const ctxWrapRef = useRef<HTMLDivElement>(null);
  const skillHlRef = useRef<HTMLDivElement>(null);
  const skillMarkRef = useRef<HTMLElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);

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
  const slashGroups = q != null && !pendingReq ? slashMenuGroups(q, s.pluginsData.skills || [], S.chat) : [];
  const slashItems = flatItems(slashGroups);
  const plusGroups = plusMenuGroups(S.chat);
  const plusItems = flatItems(plusGroups);
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
    if (slashOpen) setPlusOpen(false);
    if (pendingReq) {
      setPlusOpen(false);
      setState({ approvalMenuOpen: false, composerPicker: null });
    }
  }, [slashOpen, pendingReq?.toolCallId]);
  useEffect(() => {
    if (!s.modelDDOpen && !s.approvalMenuOpen && !ctxOpen && !plusOpen && !s.composerPicker) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      const patch: { modelDDOpen?: false; approvalMenuOpen?: false } = {};
      if (!modelWrapRef.current?.contains(t)) patch.modelDDOpen = false;
      if (!approvalWrapRef.current?.contains(t)) patch.approvalMenuOpen = false;
      if (patch.modelDDOpen !== undefined || patch.approvalMenuOpen !== undefined) setState(patch);
      if (!ctxWrapRef.current?.contains(t)) setCtxOpen(false);
      if (!plusWrapRef.current?.contains(t)) setPlusOpen(false);
      if (getState().composerPicker && !pickerRef.current?.contains(t)) setState({ composerPicker: null });
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // A layer that owns focus handles its own Escape first (the model
      // search clears its query before closing); this only closes the rest.
      if (e.defaultPrevented) return;
      e.preventDefault();
      if (plusOpen || getState().composerPicker) requestAnimationFrame(() => taRef.current?.focus());
      setCtxOpen(false);
      setPlusOpen(false);
      setState({ modelDDOpen: false, approvalMenuOpen: false, composerPicker: null });
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [s.modelDDOpen, s.approvalMenuOpen, ctxOpen, plusOpen, s.composerPicker]);
  useEffect(() => {
    if (!plusOpen) return;
    setPlusActive(0);
    requestAnimationFrame(() => plusMenuRef.current?.focus());
  }, [plusOpen]);
  const pick = (it: MenuItem) => {
    setPlusOpen(false);
    runCommand(it, taRef);
  };
  useEffect(() => {
    if (!pendingReq) return;
    const onKey = (e: KeyboardEvent) => {
      if (s.modelDDOpen || s.approvalMenuOpen || s.composerPicker || plusOpen || ctxOpen) return;
      if (isImeKey(e)) return;
      // Typing somewhere else on the page (sidebar search, a dialog field)
      // must not decide a tool call; the composer itself is disabled here.
      const t = e.target as HTMLElement | null;
      if (t && t !== document.body && t.closest("input, textarea, select, [contenteditable='true'], .dlg-overlay, .settings-overlay")) return;
      const action = approvalKeyAction(e.key, { shiftKey: e.shiftKey });
      if (!action) return;
      e.preventDefault();
      e.stopPropagation();
      void decideCurrentApproval(action);
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [pendingReq?.toolCallId, s.modelDDOpen, s.approvalMenuOpen, s.composerPicker, plusOpen, ctxOpen]);
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
    if (!s.curSess) return;
    void syncSessionRoundStats(s.curSess);
    void syncSessionUsage(s.curSess);
    if (streaming) return;
    const sess = s.sessions[s.curSess];
    if (sess) {
      let healed = false;
      for (const m of sess.msgs) {
        if (settleWork(m)) healed = true;
      }
      if (healed) saveSessions();
    }
  }, [s.curSess, streaming]);

  const saveCurrentSessionScroll = () => {
    const el = msgsRef.current;
    const active = activeSessRef.current;
    if (!el || !active.id || !active.hasMsgs) return;
    sessionViews.update(active.id, {
      chat: {
        fromBottom: Math.max(0, el.scrollHeight - el.clientHeight - el.scrollTop),
        stick: followRef.current.stick,
      },
    });
  };

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

  /** Reading near the top pulls the previous page in, no button. A reader at the
   *  bottom (or a transcript too short to scroll) keeps the bottom pinned while it
   *  grows; one scrolled up keeps the first visible message where it was. */
  const maybeLoadOlder = (fromScroll = false) => {
    const el = msgsRef.current;
    const id = activeSessRef.current.id;
    const sess = id ? getState().sessions[id] : undefined;
    if (!el || !id || !sess || !sess.msgs.length || !sess.hasOlder || sess.loadingOlder) return;
    // A failed page is retried only when the reader scrolls again, not on every render.
    if (fromScroll) olderFailedRef.current.delete(id);
    else if (olderFailedRef.current.has(id)) return;
    if (pagingAnchorRef.current || el.scrollTop > OLDER_PREFETCH_PX) return;
    if (!followRef.current.stick) {
      const first = el.querySelector<HTMLElement>(".m");
      if (first) {
        pagingAnchorRef.current = {
          node: first,
          top: first.getBoundingClientRect().top - el.getBoundingClientRect().top,
          head: sess.msgs[0],
        };
      }
    }
    void loadOlderMessages(id).then((loaded) => {
      if (loaded) return;
      pagingAnchorRef.current = null;
      if (getState().sessions[id]?.hasOlder) olderFailedRef.current.add(id);
    });
  };

  useLayoutEffect(() => {
    const el = msgsRef.current;
    const hasMsgs = !!(cur && cur.msgs.length);
    const prev = activeSessRef.current;
    const switched = prev.id !== s.curSess;
    const justHydrated = !switched && !prev.hasMsgs && hasMsgs;
    const startedStreaming = !switched && !prevStreamingRef.current && streaming;
    activeSessRef.current = { id: s.curSess, hasMsgs };
    prevStreamingRef.current = streaming;

    if (switched) {
      pagingAnchorRef.current = null;
    }
    if (startedStreaming) {
      followRef.current.resume();
    }

    if (switched || justHydrated) {
      const saved = s.curSess ? sessionViews.get(s.curSess)?.chat : undefined;
      if (!saved || saved.stick) {
        followRef.current.resume();
        if (el) stickToBottom(el, followRef.current);
        if (hasMsgs) saveCurrentSessionScroll();
        syncJump();
        queueMicrotask(maybeLoadOlder);
        return;
      }
      if (el && hasMsgs) {
        el.scrollTop = Math.max(0, el.scrollHeight - el.clientHeight - saved.fromBottom);
        followRef.current.scrolled({
          scrollTop: el.scrollTop,
          scrollHeight: el.scrollHeight,
          clientHeight: el.clientHeight,
        });
      }
      followRef.current.leave();
      syncJump();
      queueMicrotask(maybeLoadOlder);
      return;
    }

    const anchor = pagingAnchorRef.current;
    if (anchor && el && el.contains(anchor.node) && cur?.msgs[0] !== anchor.head) {
      pagingAnchorRef.current = null;
      const nowTop = anchor.node.getBoundingClientRect().top - el.getBoundingClientRect().top;
      el.scrollTop += nowTop - anchor.top;
      pagingHoldRef.current = { node: anchor.node, top: anchor.top, until: Date.now() + PAGING_HOLD_MS };
      saveCurrentSessionScroll();
      syncJump();
      queueMicrotask(maybeLoadOlder);
      return;
    }
    if (!el || !followRef.current.stick) {
      queueMicrotask(maybeLoadOlder);
      return;
    }
    stickToBottom(el, followRef.current);
    if (hasMsgs) saveCurrentSessionScroll();
    syncJump();
    queueMicrotask(maybeLoadOlder);
  });

  useEffect(() => {
    const inner = msgsInnerRef.current;
    const el = msgsRef.current;
    if (!inner || !el) return;
    const ro = new ResizeObserver(() => {
      const hold = pagingHoldRef.current;
      if (hold) {
        if (Date.now() < hold.until && el.contains(hold.node)) {
          const nowTop = hold.node.getBoundingClientRect().top - el.getBoundingClientRect().top;
          if (nowTop !== hold.top) el.scrollTop += nowTop - hold.top;
          return;
        }
        pagingHoldRef.current = null;
      }
      if (pagingAnchorRef.current || !followRef.current.stick) return;
      stickToBottom(el, followRef.current);
      saveCurrentSessionScroll();
      syncJump();
    });
    ro.observe(inner);
    return () => ro.disconnect();
  }, []);

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
  const workspace = cur?.dir || s.pendingNewChatDir || s.serverDir || "";
  const files = useFilesPanel(s.curSess, workspace);
  useEffect(() => { setCtxOpen(false); setTitleEditing(false); }, [s.curSess]);
  // This page is where the session was left; reopening it comes back here.
  useEffect(() => {
    if (s.curSess && !s.booting) sessionViews.update(s.curSess, { page: "chat" });
  }, [s.curSess, s.booting]);
  // Neither the catalog nor this transcript has arrived yet: a placeholder,
  // not the new-chat screen, or opening a chat flashes "Agentica / welcome".
  const loadingChat = s.booting || sessionHydrating(cur);
  // Each chat runs on its own model; a new-chat screen shows the project's
  // default (or the one picked there) until the chat exists.
  useEffect(() => { void loadSessionModel(s.curSess); }, [s.curSess]);
  const chatModel = s.curSess && s.sessionModel?.id === s.curSess ? s.sessionModel : null;
  const pendingProfile = !s.curSess && s.pendingModel
    ? (s.profilesData.profiles || []).find((p: any) => p.name === s.pendingModel)
    : null;
  const shownProfile = pendingProfile?.name || chatModel?.profile || s.serverProfile || s.profilesData.active || "default";
  const shownProvider = pendingProfile?.model_provider || chatModel?.provider || s.serverProvider;
  const shownModel = pendingProfile?.model_name || chatModel?.model || s.serverModelName || s.serverModel;
  const shownEffort = pendingProfile ? "" : chatModel ? chatModel.effort : s.serverReasoningEffort;
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
      <div className={"main" + ((!cur || !cur.msgs.length) && !loadingChat ? " chat-empty" : "")}>
        <div className="topbar chat-topbar">
          <div className="topbar-row">
            <div className="tb-l">
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
                    {cur?.title || (loadingChat ? "" : S.chat.newConversation)}
                  </span>
                )}
              </div>
            </div>
            <div className="tb-r">
              {s.curSess && (
                <BackgroundJobs
                  key={s.curSess}
                  sessionId={s.curSess}
                  streaming={streaming}
                  onOpenFile={files.browsePath}
                />
              )}
              {/* dsh: the way in only shows while the panel is closed; the open
                  panel carries its own collapse button in its head. */}
              {!files.open && (
                <button type="button" className="ib" onClick={() => files.setOpen(true)}
                        title={S.chat.openWorkspace} aria-label={S.chat.openWorkspace}>
                  <IconSidebarRight />
                </button>
              )}
            </div>
          </div>
          <div className="topbar-tabs" role="tablist">
            <span className="tb-tab active" role="tab" aria-selected="true">{S.chat.chatTab}</span>
            {s.curSess && (
              <Link className="tb-tab trace-link" to={`/traces?sessionId=${s.curSess}`}>
                {S.chat.viewTrace}
              </Link>
            )}
          </div>
        </div>
        <div className="chat-wrap" ref={chatWrapRef}>
          <SearchJump
            sessionId={s.curSess}
            msgCount={cur?.msgs.length || 0}
            areaRef={msgsRef}
            onLeaveFollow={() => { followRef.current.leave(); saveCurrentSessionScroll(); }}
          />
          <div className="chat" id="chatArea" ref={msgsRef}
            onWheel={(e) => { pagingHoldRef.current = null; followRef.current.wheel(e.deltaY); saveCurrentSessionScroll(); syncJump(); }}
            onTouchStart={(e) => { pagingHoldRef.current = null; followRef.current.touchStart(e.touches[0]?.clientY ?? 0); }}
            onTouchMove={(e) => { followRef.current.touchMove(e.touches[0]?.clientY ?? 0); saveCurrentSessionScroll(); syncJump(); }}
            onTouchEnd={() => followRef.current.touchEnd()}
            onScroll={() => {
              const el = msgsRef.current;
              if (!el || activeSessRef.current.id !== s.curSess) return;
              followRef.current.scrolled({
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
              });
              saveCurrentSessionScroll();
              syncJump();
              maybeLoadOlder(true);
            }}>
            <div className="msgs" ref={msgsInnerRef}>
              {(!cur || !cur.msgs.length) && loadingChat ? (
                <div className="chat-loading" aria-live="polite"><IconSpinner /><span>{S.common.loading}</span></div>
              ) : !cur || !cur.msgs.length ? (
                <div className="welcome welcome-new">
                  <Logo className="w-icon-img" />
                  <h2>Agentica</h2>
                  <p>{S.chat.welcome}</p>
                </div>
              ) : (
                <>
                  {cur.hasOlder && (
                    <div className="chat-older" aria-live="polite">
                      {cur.loadingOlder && <><IconSpinner /><span>{S.common.loading}</span></>}
                    </div>
                  )}
                  {cur.msgs.map((m, i) => (
                    <MessageView key={keyOfMsg(m)} idx={i} m={m} sessionId={s.curSess} workspace={workspace}
                      turn={(cur.olderTurns || 0) + turnOf(cur.msgs, i)}
                      onOpenFile={files.browsePath} onOpenReview={files.openReview} onOpenSkill={files.openSkill}
                      live={!!(s.curSess && s.streams[s.curSess]?.aiMsg === m)}
                      pendingToolCallId={pendingToolCallId} />
                  ))}
                </>
              )}
            </div>
          </div>
          <ChatNav
            msgs={cur?.msgs || []}
            areaRef={msgsRef}
            onLeaveFollow={() => { followRef.current.leave(); saveCurrentSessionScroll(); }}
          />
          {showJump && (
            <button type="button" className="scroll-bottom-btn visible" title={S.chat.jumpLatest} aria-label={S.chat.jumpLatest}
                    onClick={() => {
                      const el = msgsRef.current;
                      followRef.current.resume();
                      if (el) stickToBottom(el, followRef.current);
                      saveCurrentSessionScroll();
                      setShowJump(false);
                    }}>
              <IconChevronDown />
            </button>
          )}
          <div className="input-area" ref={inputAreaRef}>
          {pendingReq && (
            <ApprovalCard
              key={pendingReq.toolCallId}
              req={pendingReq}
              queueIndex={0}
              queueTotal={pendingApprovals.length}
              busy={decidingApproval}
              onDecide={(d) => void decideCurrentApproval(d)}
            />
          )}
          {cur && <TodoDock todos={dockTodos(cur.msgs, busy)} live={streaming} />}
          {goalRun && (
            <div className="goal-bar" title={goalRun.progress}>
              <span className="goal-bar-status">{S.chat.goalBar(goalRun.status, goalRun.objective)}</span>
              {goalRun.progress && <span className="goal-bar-progress">{goalRun.progress}</span>}
            </div>
          )}
          {!!queued.length && (
            <QueueDock queued={queued} streaming={streaming} />
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
              <CommandMenu groups={slashGroups} active={slashActive} onPick={pick} onHover={setSlashActive}
                           className="slash-menu" emptyText={S.chat.slashEmpty} />
            )}
            {s.composerPicker && !slashOpen && !pendingReq && (
              <ComposerPicker
                key={s.composerPicker}
                menuRef={pickerRef}
                groups={s.composerPicker === "model" ? modelPickerGroups() : s.composerPicker === "skill" ? skillPickerGroups() : permissionPickerGroups()}
                searchPlaceholder={s.composerPicker === "model" ? S.chat.pickerSearchModel : s.composerPicker === "skill" ? S.chat.skillSearch : undefined}
                emptyText={
                  s.composerPicker === "model" && !(s.profilesData.profiles || []).length
                    ? S.chat.noProfiles
                    : s.composerPicker === "skill" && !invocableSkills(s.pluginsData.skills || []).length
                      ? S.chat.skillNone
                      : S.chat.pickerNoMatch
                }
                footer={s.composerPicker === "model"
                  ? { label: S.chat.manageProfiles, onClick: () => { setState({ composerPicker: null }); void primeSettings("profiles"); } }
                  : undefined}
                onClose={() => {
                  setState({ composerPicker: null });
                  requestAnimationFrame(() => taRef.current?.focus());
                }}
                onPick={(key) => {
                  const kind = getState().composerPicker;
                  setState({ composerPicker: null });
                  if (kind === "model") void pickSessionModel(key);
                  else if (kind === "skill") setState({ inputText: key + " " });
                  else setApprovalMode(key as "ask" | "auto" | "allow-all");
                  requestAnimationFrame(() => taRef.current?.focus());
                }}
              />
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
                if (isImeKey(e.nativeEvent)) return;
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
                    pick(slashItems[slashActive] || slashItems[0]);
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
                if (e.key === "Enter" && !e.shiftKey) {
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
                <div className="plus-wrap" ref={plusWrapRef}>
                  <button type="button" className={"foot-btn plus-btn" + (plusOpen ? " on" : "")}
                          title={plusOpen ? undefined : S.chat.plusMenu} aria-haspopup="menu" aria-expanded={plusOpen}
                          onClick={() => {
                            setState({ approvalMenuOpen: false, modelDDOpen: false, composerPicker: null });
                            setPlusOpen(!plusOpen);
                          }}>
                    <IconPlus />
                  </button>
                  {plusOpen && (
                    <CommandMenu groups={plusGroups} active={plusActive} onPick={pick} onHover={setPlusActive}
                                 className="plus-menu" menuRef={plusMenuRef}
                                 onKeyDown={(e) => {
                                   if (e.key === "ArrowDown" || e.key === "ArrowUp") {
                                     e.preventDefault();
                                     const d = e.key === "ArrowDown" ? 1 : -1;
                                     setPlusActive((n) => (n + d + plusItems.length) % plusItems.length);
                                   } else if (e.key === "Enter" || e.key === " ") {
                                     e.preventDefault();
                                     pick(plusItems[plusActive]);
                                   }
                                 }} />
                  )}
                </div>
                <ComposerDir />
                <div className="approval-wrap" ref={approvalWrapRef}>
                  <button className="foot-btn approval-btn" title={S.chat.permHint(s.selectedApprovalMode)}
                          onClick={() => {
                            setState({ approvalMenuOpen: !s.approvalMenuOpen, modelDDOpen: false, composerPicker: null });
                          }}>
                    <span className="quick-icon">{permIcon(s.selectedApprovalMode)}</span>
                    <span className="approval-label">{S.chat.permLabel(s.selectedApprovalMode)}</span>
                    <span className="arr"><IconChevronDown /></span>
                  </button>
                  {s.approvalMenuOpen && (
                    <div className="approval-dd open">
                      {(["ask", "auto", "allow-all"] as const).map((mode) => (
                        <button key={mode} className={"quick-item" + (s.selectedApprovalMode === mode ? " active" : "")}
                                title={S.chat.permHint(mode)}
                                onClick={() => setApprovalMode(mode)}>
                          <span className="quick-icon">{permIcon(mode)}</span>
                          <span className="approval-label">{S.chat.permLabel(mode)}</span>
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
                          onPointerEnter={() => prefetchSessionUsage(s.curSess)}
                          onFocus={() => prefetchSessionUsage(s.curSess)}
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
                    setState({ modelDDOpen: !s.modelDDOpen, approvalMenuOpen: false, composerPicker: null });
                  }}>
                    {shownModel}
                  </button>
                  {s.modelDDOpen && (
                    <ModelDropdown
                      profiles={s.profilesData.profiles || []}
                      shownProfile={shownProfile}
                      shownProvider={shownProvider}
                      shownModel={shownModel}
                      shownEffort={shownEffort}
                    />
                  )}
                </div>
                <button className={"act-btn " + (showStop ? "stop" : "send")}
                        title={showStop ? S.chat.stop : S.chat.send}
                        // dsh: an empty draft dims the send button; anything to send lights it.
                        disabled={compacting || (!showStop && !s.inputText.trim() && !s.pendingFiles.length)}
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

function ModelDropdown({ profiles, shownProfile, shownProvider, shownModel, shownEffort }: {
  profiles: any[]; shownProfile: string; shownProvider: string; shownModel: string; shownEffort: string;
}) {
  const S = useStrings();
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const q = query.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!q) return profiles;
    return profiles.filter((p) =>
      `${p.name || ""} ${p.model_provider || ""} ${p.model_name || ""}`.toLowerCase().includes(q));
  }, [profiles, q]);
  useEffect(() => {
    const idx = filtered.findIndex((p: any) => p.name === shownProfile);
    setCursor(idx >= 0 ? idx : 0);
  }, [q, shownProfile, filtered.length]);
  return (
    <div className="model-dd open">
      <div className="dd-config-card">
        <div className="dd-config-row"><span>Profile</span><strong>{shownProfile}</strong></div>
        <div className="dd-config-row"><span>{S.chat.model}</span><strong>{shownProvider}/{shownModel}</strong></div>
        {shownEffort && <div className="dd-config-row"><span>effort</span><strong>{shownEffort}</strong></div>}
      </div>
      <div className="dd-section">{S.chat.switchProfile}</div>
      {profiles.length > 0 && (
        <label className="dd-search">
          <IconSearch />
          <input type="search" autoFocus value={query} placeholder={S.chat.pickerSearchModel}
                 onChange={(e) => setQuery(e.target.value)}
                 onKeyDown={(e) => {
                   if (e.key === "ArrowDown" && filtered.length) {
                     e.preventDefault();
                     setCursor((c) => (c + 1) % filtered.length);
                   } else if (e.key === "ArrowUp" && filtered.length) {
                     e.preventDefault();
                     setCursor((c) => (c - 1 + filtered.length) % filtered.length);
                   } else if (e.key === "Enter" && filtered[cursor]) {
                     e.preventDefault();
                     void pickSessionModel(filtered[cursor].name);
                   } else if (e.key === "Escape") {
                     e.preventDefault();
                     if (query) setQuery(""); else setState({ modelDDOpen: false });
                   }
                 }} />
        </label>
      )}
      <div className="dd-profile-list">
        {!profiles.length && <div className="dd-empty">{S.chat.noProfiles}</div>}
        {profiles.length > 0 && !filtered.length && <div className="dd-empty">{S.chat.pickerNoMatch}</div>}
        {filtered.map((p: any, i: number) => (
          <button key={p.name}
                  className={"dd-profile" + (p.name === shownProfile ? " active" : "") + (i === cursor ? " kb-focus" : "")}
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => void pickSessionModel(p.name)}>
            <div className="dd-p-name">{p.name}{p.name === shownProfile ? <span className="dd-active"> ●</span> : null}</div>
            <div className="dd-p-model">{p.model_provider}/{p.model_name}</div>
          </button>
        ))}
      </div>
      <button className="dd-manage" onClick={() => { setState({ modelDDOpen: false }); void primeSettings("profiles"); }}>{S.chat.manageProfiles}</button>
    </div>
  );
}

/** Scrolls the chat to the turn a sidebar search hit points at. It draws
 *  nothing: finding text is the workspace search box's job now, and this is
 *  only the half that has to live next to the message list. */
function SearchJump({ sessionId, msgCount, areaRef, onLeaveFollow }: {
  sessionId: string | null;
  msgCount: number;
  areaRef: RefObject<HTMLDivElement | null>;
  onLeaveFollow: () => void;
}) {
  const s = useAppState();
  const target = s.chatSearchTarget?.sessionId === sessionId ? s.chatSearchTarget : null;
  const activeQuery = s.sidebarSearch.trim() || (target?.query.trim() ?? "");
  const lastScrolledSeq = useRef<number>(0);

  useEffect(() => {
    const root = areaRef.current;
    if (!root) return;
    let obs: MutationObserver | null = null;
    let frame = 0;
    const observe = () => obs?.observe(root, { childList: true, subtree: true });
    const run = (shouldScroll: boolean) => {
      // Highlighting rewrites text nodes under the very list it watches. A flag
      // set around the rewrite does not help: observer callbacks run as a
      // microtask, after the flag is already cleared, so each rewrite looked
      // like a fresh change and queued the next, forever. Stop watching while
      // we write and drop what the write itself queued.
      obs?.disconnect();
      try {
        const mark = applyChatSearchHighlights(root, activeQuery, target?.turn, target?.role);
        if (shouldScroll && target) {
          onLeaveFollow();
          const sel = target.role
            ? `.m[data-turn="${target.turn}"][data-role="${target.role}"]`
            : `.m[data-turn="${target.turn}"]`;
          const msgEl = root.querySelector<HTMLElement>(sel)
            || root.querySelector<HTMLElement>(`.m[data-turn="${target.turn}"]`);
          const scrollTarget = (mark && msgEl?.contains(mark)) ? mark : msgEl;
          if (scrollTarget) {
            // Smooth only for a short hop. Across a long transcript the animation
            // is cancelled by the first layout change on the way (late Markdown,
            // images), stranding the view at the bottom, far from the hit.
            const area = root.getBoundingClientRect();
            const far = Math.abs(scrollTarget.getBoundingClientRect().top - area.top) > area.height * 3;
            scrollTarget.scrollIntoView({ behavior: far ? "auto" : "smooth", block: "center" });
            if (mark && msgEl?.contains(mark)) lastScrolledSeq.current = target.seq;
          }
        }
      } finally {
        obs?.takeRecords();
        observe();
      }
    };
    const needScroll = Boolean(target && target.seq !== lastScrolledSeq.current);
    run(needScroll);
    if (!activeQuery) return;
    obs = new MutationObserver(() => {
      // Messages arrived or were paged in: highlight them, once per frame however
      // many mutations a render produced.
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        run(Boolean(target && target.seq !== lastScrolledSeq.current));
      });
    });
    observe();
    return () => {
      obs?.disconnect();
      if (frame) cancelAnimationFrame(frame);
    };
  }, [activeQuery, target?.seq, target?.turn, target?.role, msgCount, sessionId]);
  return null;
}

/** `compact · compacted N -> M messages`; opens to the rendered notes. */
function CompactCard({ c, onOpenFile }: { c: NonNullable<ChatMsg["compact"]>; onOpenFile?: (path: string) => void }) {
  const S = useStrings();
  const [open, setOpen] = useState(false);
  if (c.pending) {
    return (
      <div className="compact-card running">
        <div className="compact-head">
          <span className="work-icon"><IconSpinner /></span>
          <span>{S.chat.compactRunning}</span>
        </div>
      </div>
    );
  }
  return (
    <div className={"compact-card" + (open ? " open" : "")}>
      <div className="compact-head">
        <button type="button" className="compact-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          <span className="chev"><IconChevronDown /></span>
          <span className="work-icon"><IconCompact /></span>
          <span className="compact-kind">compact</span>
          <span className="compact-sep">·</span>
          <span>{c.before != null ? S.chat.compactDone(c.before, c.after ?? 0) : c.auto ? S.chat.compactDoneAuto : S.chat.compactDoneHistory}</span>
        </button>
        {c.notesPath && (
          <span className="compact-path-wrap">
            <button
              type="button"
              className="compact-path"
              title={c.notesPath}
              onClick={() => (onOpenFile ? onOpenFile(c.notesPath!) : setOpen(!open))}
            >
              {c.notesPath.includes("/") ? (
                <>
                  <span className="compact-path-dir">{c.notesPath.slice(0, c.notesPath.lastIndexOf("/") + 1)}</span>
                  <span className="compact-path-name">{c.notesPath.slice(c.notesPath.lastIndexOf("/") + 1)}</span>
                </>
              ) : (
                <span className="compact-path-name">{c.notesPath}</span>
              )}
            </button>
            <button
              type="button"
              className="compact-path-copy"
              title={S.files.copyPath}
              aria-label={S.files.copyPath}
              onClick={() => {
                void navigator.clipboard.writeText(c.notesPath!).then(() => showToast(S.common.copied));
              }}
            >
              <IconCopy />
            </button>
          </span>
        )}
      </div>
      {open && (c.notes?.trim()
        ? (
          <div className="compact-body">
            <Suspense fallback={<pre className="markdown-fallback">{c.notes}</pre>}>
              <ChatMarkdown text={c.notes} streaming={false} />
            </Suspense>
          </div>
        )
        : <div className="compact-empty">{S.chat.compactNoNotes}</div>)}
    </div>
  );
}

/** Round number of message `idx`: real questions up to it (steer chips and
 *  automated reports are not rounds), the same count the chat nav shows. */
function turnOf(msgs: ChatMsg[], idx: number): number {
  let n = 0;
  for (let i = 0; i <= idx && i < msgs.length; i++) {
    const m = msgs[i];
    if (m.role === "user" && !m.steer && !m.automated) n++;
  }
  return Math.max(1, n);
}

function MessageView({ idx, m, sessionId, workspace, turn, onOpenFile, onOpenReview, onOpenSkill, live, pendingToolCallId }: {
  idx: number; m: ChatMsg; sessionId: string | null; workspace: string; turn: number; onOpenFile: (path: string) => void;
  onOpenReview: (turn: number, files: FileEdits[], path: string) => void;
  onOpenSkill: (name: string) => void; live?: boolean; pendingToolCallId?: string;
}) {
  const S = useStrings();
  const skill = m.role === "user"
    ? (m.skill || parseSkillTurn(m.content, invocableSkills(getState().pluginsData.skills || [])))
    : null;
  if (m.role === "user" && m.automated) {
    return (
      <div id={"msg-" + idx} className="m m-a m-auto">
        {m.cronJob
          ? <ScheduledRunCard job={m.cronJob} text={m.content} ts={m.ts} />
          : <AutomatedNotice text={m.content} ts={m.ts} />}
      </div>
    );
  }
  if (m.role === "user" && m.steer) {
    return (
      <div id={"msg-" + idx} className="m m-steer-wrap">
        <SteerChip text={m.content} ts={m.ts} onOpenSkill={onOpenSkill} />
      </div>
    );
  }
  if (m.role === "assistant" && m.compact) {
    return (
      <div id={"msg-" + idx} className="m m-a">
        <div className="msg-stack"><CompactCard c={m.compact} onOpenFile={onOpenFile} /></div>
      </div>
    );
  }
  if (!live) settleWork(m);
  const segs = m.role === "assistant" ? foldProcess(groupParts(partsOf(m))) : null;
  const workSegCount = segs ? segs.filter((seg) => seg.type === "work").length : 0;
  const hasWork = workSegCount > 0;
  const hasSteer = Boolean(segs?.some((seg) => seg.type === "single" && seg.part.kind === "steer"));
  const hasPendingApproval = Boolean(
    pendingToolCallId && segs?.some(
      (seg) => seg.type === "work" && seg.items.some((p) => p.kind === "tool" && p.toolCallId === pendingToolCallId),
    ),
  );
  const st = getState();
  const searchFocused = !!(
    st.chatSearchTarget
    && st.chatSearchTarget.sessionId === sessionId
    && st.chatSearchTarget.turn === turn
    && (!st.chatSearchTarget.role || st.chatSearchTarget.role === m.role)
  );
  const [processOpen, setProcessOpen] = useState(() => Boolean(m.aborted || m.error || hasSteer || searchFocused));
  useEffect(() => {
    if (m.aborted || m.error || hasSteer || searchFocused) setProcessOpen(true);
  }, [m.aborted, m.error, hasSteer, searchFocused]);
  const effectiveProcessOpen = processOpen || hasPendingApproval;
  // While streaming the process shows unless the reader folds it; settled
  // turns keep their own default (folded) above.
  const [liveFolded, setLiveFolded] = useState(false);
  const liveNow = useLiveClock(!!live && m.role === "assistant");
  const rawDur = m.durationMs ?? ((m.durationSec || 0) * 1000);
  const workDur = segs
    ? segs.reduce((acc, seg) => (seg.type === "work" ? acc + workSummary(seg.items, Date.now(), false).ms : acc), 0)
    : 0;
  const turnDurationMs = rawDur > 0 ? rawDur : workDur;
  const showLiveProcess = !!live && m.role === "assistant";
  const showTurnProcess = !live && hasWork;
  const processOpenNow = showLiveProcess ? (!liveFolded || hasPendingApproval) : effectiveProcessOpen;
  const turnProcessLabel = showLiveProcess
    ? S.chat.turnProcessRunningFor(fmtRunDuration(liveNow - (m.ts || liveNow), S.chat.runDurationUnits))
    : m.aborted
    ? S.chat.turnProcessStopped
    : m.error
      ? S.chat.turnProcessFailed
      : turnDurationMs > 0
        ? `${S.chat.turnProcessTook}${fmtRunDuration(turnDurationMs, S.chat.runDurationUnits)}`
        : S.chat.turnProcessWorked;
  const finalAnswer = m.role === "assistant" ? finalAnswerOf(m) : "";
  const filesCardText = m.role === "assistant" ? finalAnswer : (m.content || "");
  const copyText = m.role === "assistant"
    ? ((!effectiveProcessOpen && finalAnswer) ? finalAnswer : allAssistantTextOf(m))
    : m.content;
  return (
    <div id={"msg-" + idx} data-turn={turn} data-role={m.role}
         className={"m " + (m.role === "user" ? "m-u" : "m-a") + (live ? " streaming" : "") + (searchFocused ? " search-focus" : "")}>
      <div className="msg-stack">
        {(showTurnProcess || showLiveProcess) && (
          <button
            type="button"
            className={"turn-process" + (processOpenNow ? " open" : "") + (showLiveProcess ? " live" : "")}
            aria-expanded={processOpenNow}
            onClick={() => (showLiveProcess ? setLiveFolded((v) => !v) : setProcessOpen((v) => !v))}
          >
            <span className={"turn-process-label" + (showLiveProcess ? " shimmer" : "")}>{turnProcessLabel}</span>
            <span className="turn-process-chev"><IconChevronDown /></span>
          </button>
        )}
        {segs ? segs.map((seg, i) => {
          const isLast = !!live && i === segs.length - 1;
          if ((showTurnProcess || showLiveProcess) && !processOpenNow && isProcessSegment(segs, i)) {
            return null;
          }
          if (seg.type === "work") {
            return (
              <WorkGroup
                key={i}
                items={seg.items}
                isLast={isLast}
                live={!!live}
                pendingToolCallId={pendingToolCallId}
                onOpenFile={onOpenFile}
              />
            );
          }
          if (seg.part.kind === "steer") {
            return <SteerChip key={i} text={seg.part.text} ts={seg.part.ts} onOpenSkill={onOpenSkill} />;
          }
          const text = seg.part.kind === "text" ? seg.part.text : "";
          if (!text) return null;
          return (
            <div className={"bub" + (live && isLast ? " streaming-bub" : "")} key={i}>
              <Suspense fallback={<span className="markdown-fallback">{text}</span>}>
                <ChatMarkdown text={text} streaming={!!live && isLast} />
              </Suspense>
            </div>
          );
        }) : (
          (m.content || m.previews?.length || !m.error) && (
            <div className="bub">
              {!!m.previews?.length && (
                <div className="msg-previews">
                  {m.previews.map((src, i) => <PreviewableImage key={i} src={src} alt="" />)}
                </div>
              )}
              {skill
                ? <SkillTurn name={skill.name} args={skill.args} source={m.content} onOpen={onOpenSkill} />
                : m.role === "user"
                  ? <CollapsibleUserText forceOpen={searchFocused}>{m.content}</CollapsibleUserText>
                  : m.content}
            </div>
          )
        )}
        <MessageFilesCard
          text={filesCardText}
          steps={m.steps}
          uploaded={m.files}
          workspace={workspace || null}
          onOpenFile={onOpenFile}
          onOpenDiff={(files, path) => onOpenReview(turn, files, path)}
          live={!!live}
          role={m.role}
        />
        {m.aborted && <div className="msg-note">{S.chat.aborted}</div>}
        {m.error && (
          <div className="msg-error">
            {S.chat.error(m.error)}
            {` - ${formatErrorStamp(m.errorTs ?? m.ts)}`}
          </div>
        )}
        <MessageFooter m={m} live={!!live} sessionId={sessionId} idx={idx} copyText={copyText} />
      </div>
    </div>
  );
}

/** Wall clock that ticks once a second while `active`, for the live `已处理 …` header. */
function useLiveClock(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [active]);
  return now;
}

function MessageFooter({ m, live, sessionId, idx, copyText }: {
  m: ChatMsg; live?: boolean; sessionId: string | null; idx: number; copyText?: string;
}) {
  const S = useStrings();
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [usageOpen, setUsageOpen] = useState(false);
  if (live) return null;
  const copy = () => {
    // Always the original message for user turns (a rendered skill chip is a view,
    // not the text the user sent). For assistant turns, copy the visible final
    // answer when collapsed, or clean paragraph-joined text when expanded.
    const text = copyText ?? m.content;
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
  const hasStats = m.tokIn != null;
  const rating = m.feedback?.rating || null;
  const canAct = !!sessionId;
  const rate = (next: "up" | "down" | null, comment = "") => {
    if (!sessionId) return;
    void rateMessage(sessionId, idx, next, comment).then((ok) => {
      if (ok && next) showToast(S.chat.feedbackSaved);
    });
  };
  return (
    // Pinned while the usage panel is open: the footer otherwise fades out as
    // soon as the pointer leaves the reply for the panel.
    <div className={"msg-footer" + (usageOpen ? " pinned" : "")}>
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
      {hasStats && <TurnUsagePanel m={m} onOpenChange={setUsageOpen} />}
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

async function requeueLateSteer(sessId: string, aiMsg?: ChatMsg) {
  const parked = await api.takeSteerApi(sessId);
  const late = (parked.ok && parked.data?.messages) ? parked.data.messages : [];
  if (late.length && aiMsg) {
    removeUndeliveredSteers(sessId, aiMsg, late);
    bump();
  }
  for (let i = late.length - 1; i >= 0; i--) {
    enqueueMessage(sessId, late[i], [], { atFront: true, silent: true });
  }
}

/**
 * dsh `QueueDock`: queued messages stay in a list above the composer and edit
 * in place so queue order (`1, 2, 3…`) is preserved and the main composer's
 * draft is never overwritten.
 */
function QueueDock({ queued, streaming }: { queued: QueuedMessage[]; streaming: boolean }) {
  const S = useStrings();
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (editing && !queued.some((q) => q.id === editing.id)) setEditing(null);
  }, [editing, queued]);

  const expanded = queued.length === 1 || !collapsed || editing !== null;
  const saveEdit = (qItem: QueuedMessage) => {
    if (!editing || editing.id !== qItem.id) return;
    const trimmed = editing.text.trim();
    if (!trimmed && !qItem.files.length) return;
    setState({
      messageQueue: updateQueuedMessageText(getState().messageQueue, qItem.id, editing.text),
    });
    setEditing(null);
  };
  const removeQueued = (id: string) => {
    setState({ messageQueue: getState().messageQueue.filter((x) => x.id !== id) });
  };
  const steerNow = async (qItem: QueuedMessage) => {
    if (!streaming || qItem.files.length > 0 || !qItem.text.trim()) return;
    removeQueued(qItem.id);
    await applySteer(qItem.sessionId, qItem.text.trim());
  };

  return (
    <div className="queue-dock" data-queue-dock="">
      <div className="queue-dock-panel">
        {queued.length > 1 && (
          <button
            type="button"
            className="queue-dock-head"
            aria-expanded={expanded}
            disabled={editing !== null}
            onClick={() => setCollapsed((v) => !v)}
          >
            <span className="queue-dock-lead" aria-hidden="true"><IconQueue /></span>
            <span className="queue-dock-count">{S.chat.queuedCount(queued.length)}</span>
            <span className={"queue-dock-chev" + (expanded ? " open" : "")} aria-hidden="true">
              <IconChevronDown />
            </span>
          </button>
        )}
        {expanded && (
          <ul className="queue-dock-list">
            {queued.map((qItem, idx) => {
              const isEditing = editing?.id === qItem.id;
              return (
                <li key={qItem.id} className="queue-row">
                  {queued.length === 1 ? (
                    <span className="queue-dock-lead" aria-hidden="true"><IconQueue /></span>
                  ) : (
                    <span className="queue-idx">{idx + 1}</span>
                  )}
                  {isEditing ? (
                    <QueueInlineEditor
                      value={editing.text}
                      label={S.chat.editQueued}
                      onChange={(text) => setEditing({ id: qItem.id, text })}
                      onSave={() => saveEdit(qItem)}
                      onCancel={() => setEditing(null)}
                    />
                  ) : (
                    <>
                      {qItem.files.length > 0 && (
                        <span className="queue-files">
                          {qItem.files.map((f, fi) => (
                            <span key={fi} className="queue-file-tag" title={f.name}>{f.name}</span>
                          ))}
                        </span>
                      )}
                      <span
                        className="queue-preview"
                        title={qItem.text}
                        onDoubleClick={() => setEditing({ id: qItem.id, text: qItem.text })}
                      >
                        {qItem.text || S.chat.attachmentOnly}
                      </span>
                    </>
                  )}
                  <div className="queue-actions">
                    {isEditing ? (
                      <>
                        <button
                          type="button"
                          className="queue-act"
                          title={S.chat.saveQueued}
                          aria-label={S.chat.saveQueued}
                          disabled={!editing.text.trim() && !qItem.files.length}
                          onClick={() => saveEdit(qItem)}
                        >
                          <IconCheck />
                        </button>
                        <button
                          type="button"
                          className="queue-act"
                          title={S.chat.cancelEditQueued}
                          aria-label={S.chat.cancelEditQueued}
                          onClick={() => setEditing(null)}
                        >
                          <IconClose />
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          type="button"
                          className="queue-act"
                          title={S.chat.editQueued}
                          aria-label={S.chat.editQueued}
                          onClick={() => setEditing({ id: qItem.id, text: qItem.text })}
                        >
                          <IconPencil />
                        </button>
                        <button
                          type="button"
                          className="queue-act"
                          title={S.chat.removeQueued}
                          aria-label={S.chat.removeQueued}
                          onClick={() => removeQueued(qItem.id)}
                        >
                          <IconTrash />
                        </button>
                        {streaming && qItem.files.length === 0 && (
                          <button
                            type="button"
                            className="queue-act"
                            title={S.chat.steerQueued}
                            aria-label={S.chat.steerQueued}
                            onClick={() => void steerNow(qItem)}
                          >
                            <IconSend />
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

function QueueInlineEditor({
  value, label, onChange, onSave, onCancel,
}: {
  value: string;
  label: string;
  onChange: (value: string) => void;
  onSave: () => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(120, node.scrollHeight + node.offsetHeight - node.clientHeight)}px`;
  }, [value]);
  return (
    <textarea
      ref={ref}
      autoFocus
      rows={1}
      className="queue-edit-ta"
      aria-label={label}
      value={value}
      onChange={(e) => onChange(e.currentTarget.value)}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          onCancel();
          return;
        }
        if (isImeKey(e.nativeEvent)) return;
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          onSave();
        }
      }}
    />
  );
}

function appendSteerBubble(sessId: string, text: string) {
  const live = getState().streams[sessId]?.aiMsg;
  const sess = getState().sessions[sessId];
  if (live) {
    appendSteerPart(live, text);
    if (sess) rememberSteersFromMsgs(sessId, sess.msgs, sess.olderTurns || 0);
    saveSessions();
    bump();
    return;
  }
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

/** When the composer last handed text off. Empty Enter stops a running turn,
 *  so a key-repeat or double press right after sending must not land as
 *  "stop" on the turn it just started. */
let lastSubmitAt = 0;
const STOP_GRACE_MS = 600;

async function submit() {
  const st = getState();
  if (st.curSess && st.streams[st.curSess]?.pendingApprovals?.[0]) return;
  const text = st.inputText.trim();
  const files = st.pendingFiles.slice();
  const S = getStrings();
  if (text || files.length) lastSubmitAt = Date.now();

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
    if (sessId && isBusy(sessId) && !isCompacting(sessId) && Date.now() - lastSubmitAt > STOP_GRACE_MS) {
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
    // The agent changed the user's automations: the header chip and sidebar follow.
    if (evt.data?.name === "cronjob") void loadCronJobs();
    const callId = unfinishedToolCallId(aiMsg, evt.data.tool_call_id);
    // `task`: keep the subagent's JSON payload (the card reads it, as it
    // does from the session log after a refresh) rather than the text recap.
    const result = evt.data.task && typeof evt.data.task === "object" ? JSON.stringify(evt.data.task) : evt.data.result;
    finishTool(aiMsg, result, evt.data.diff, evt.data.tool_call_id);
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
      if (evt.event === "tool_completed") {
        void syncSessionUsage(sessId);
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
          if (turn.cache_write_tokens != null) aiMsg.cacheWrite = turn.cache_write_tokens;
          aiMsg.requests = turn.api_calls ?? evt.data.requests ?? aiMsg.requests;
          if (Array.isArray(turn.models) && turn.models.length) aiMsg.model = turn.models.join(", ");
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
        if (evt.data.content) ensureTrailingFinalAnswer(aiMsg, String(evt.data.content));
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
  if (sess) {
    sess.running = false;
    rememberSteersFromMsgs(sessId, sess.msgs, sess.olderTurns || 0);
  }
  if (getState().curSess !== sessId) {
    if (sess) sess.unread = true;
  } else {
    markSessionRead(sessId);
  }
  saveSessions();
  bump();
  if (disconnected || pageUnloading) return;
  await requeueLateSteer(sessId, aiMsg);
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

/** Sessions a resume is already attaching to. switchTo and syncFromServer
 *  can both ask for one; without this both pass the `streams` check before
 *  either has awaited the active run, and the turn gets two streams. */
const resuming = new Set<string>();

async function resumeLiveStream(sessId: string) {
  if (getState().streams[sessId] || resuming.has(sessId)) return;
  resuming.add(sessId);
  try {
    await attachLiveStream(sessId);
  } finally {
    resuming.delete(sessId);
  }
}

async function attachLiveStream(sessId: string) {
  const sess = getState().sessions[sessId];
  if (!sess) return;
  const { ok, data } = await api.fetchActiveRun(sessId);
  if (getState().streams[sessId]) return;
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
  if (run.cron_job && run.automated_message) {
    if ((!sess.title || sess.title === "Chat") && run.cron_job.name?.trim()) {
      sess.title = run.cron_job.name.trim();
    }
    // A scheduled run streaming in: show its card, once, above the reply.
    const last = [...sess.msgs].reverse().find((m) => m.role === "user");
    if (!last?.cronJob || last.content !== run.automated_message) {
      sess.msgs.push({
        role: "user", automated: true, content: run.automated_message, ts: Date.now(),
        cronJob: run.cron_job,
      });
    }
  } else if (run.user_message) {
    let lastUser: ChatMsg | undefined;
    for (let i = sess.msgs.length - 1; i >= 0; i--) {
      if (sess.msgs[i].role === "user" && !sess.msgs[i].automated) {
        lastUser = sess.msgs[i];
        break;
      }
    }
    if (!lastUser || lastUser.content.trim() !== run.user_message.trim()) {
      sess.msgs.push({
        role: "user",
        content: run.user_message,
        ts: Date.now(),
        files: [],
        previews: [],
      });
    }
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
  if (!sess.named && (!sess.title || sess.title === "Chat") && sess.msgs.filter((m) => m.role === "user" && !m.automated).length === 1) {
    sess.title = (text || "Chat").slice(0, 50);
  }
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

    // A model picked on the new-chat screen must be on the chat before its
    // first run builds the agent.
    await applyPendingModel(sessId);
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

/** Run a `+` / `/` menu item. Built-ins act directly; a skill puts its slash
 *  command in the composer so the user can add arguments before sending. */
function runCommand(it: MenuItem, taRef: { current: HTMLTextAreaElement | null }) {
  const focus = () => requestAnimationFrame(() => taRef.current?.focus());
  switch (it.id) {
    case "file":
      document.getElementById("fileInput")?.click();
      return;
    case "goal":
      setState({ inputText: "", goalCompose: { budgetText: "" } });
      focus();
      return;
    case "skill":
    case "permission":
    case "model":
      // The `/permission` text the user typed is not a message. Open on the
      // next tick: a pick fires on mousedown, and the document's outside-click
      // handler runs after it and would close a dropdown opened right now.
      setState({ inputText: clearIfSlash() });
      setTimeout(() => setState({ composerPicker: it.id as "model" | "permission" | "skill", modelDDOpen: false, approvalMenuOpen: false }), 0);
      return;
    case "export": {
      setState({ inputText: clearIfSlash() });
      const id = getState().curSess;
      if (!id || !getState().sessions[id]?.msgs.length) {
        showToast(getStrings().chat.exportNothing);
        return;
      }
      void exportSessionLog(id);
      return;
    }
    case "compact": {
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
    default:
      // `/queue` and skills: a prefix for what the user types next.
      setState({ inputText: it.cmd + " " });
      focus();
  }
}

function setApprovalMode(mode: "ask" | "auto" | "allow-all") {
  localStorage.setItem("ag_approval", mode);
  setState({ selectedApprovalMode: mode, approvalMenuOpen: false });
  pushPrefs();
}

/** Model picker rows: profiles grouped by provider, model name first. */
function modelPickerGroups(): PickerGroup[] {
  const st = getState();
  const chat = st.curSess && st.sessionModel?.id === st.curSess ? st.sessionModel.profile : "";
  const active = (!st.curSess && st.pendingModel) || chat || st.serverProfile || st.profilesData.active;
  const byProvider = new Map<string, PickerGroup>();
  for (const p of st.profilesData.profiles || []) {
    const provider = String(p.model_provider || "");
    let g = byProvider.get(provider);
    if (!g) byProvider.set(provider, g = { title: provider || "—", options: [] });
    const model = String(p.model_name || p.name);
    g.options.push({ key: p.name, label: model, sub: p.name !== model ? p.name : undefined, selected: p.name === active });
  }
  return [...byProvider.values()];
}

function permissionPickerGroups(): PickerGroup[] {
  const S = getStrings();
  const cur = getState().selectedApprovalMode;
  return [{
    title: S.chat.pickerPermTitle,
    options: (["ask", "auto", "allow-all"] as const).map((mode) => ({
      key: mode, label: S.chat.permLabel(mode), sub: S.chat.permHint(mode), icon: permIcon(mode), selected: cur === mode,
    })),
  }];
}

function skillPickerGroups(): PickerGroup[] {
  const S = getStrings();
  const items = invocableSkills(getState().pluginsData.skills || []);
  if (!items.length) return [];
  return [{
    title: S.chat.cmdGroupSkills,
    options: items.map((it) => ({
      key: it.cmd, label: it.name, sub: it.desc || undefined, icon: <IconBook />,
    })),
  }];
}

/**
 * Download the session JSONL. Fetch first so a failure reaches the page (a
 * bare `<a download>` to a 401/404 only gets the browser's "couldn't get the
 * file"), then let the user choose where to save it when the browser can.
 */
async function exportSessionLog(sessId: string) {
  const S = getStrings();
  const name = `${sessId}.jsonl`;
  let r: Response;
  try {
    r = await fetch(`/api/sessions/${encodeURIComponent(sessId)}/export`);
  } catch (e: any) {
    showToast(S.chat.exportFailed(String(e?.message || e)), 4000);
    return;
  }
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { const j = await r.json(); if (j?.detail) detail = String(j.detail); } catch { /* not json */ }
    // An unknown route 404s with FastAPI's bare "Not Found": a gateway started
    // before this endpoint existed.
    showToast(r.status === 404 && detail === "Not Found" ? S.chat.exportNoRoute : S.chat.exportFailed(detail), 4000);
    return;
  }
  const picker = (window as any).showSaveFilePicker as undefined | ((o: any) => Promise<any>);
  if (picker) {
    try {
      const handle = await picker({
        suggestedName: name,
        types: [{ description: "JSONL", accept: { "application/x-ndjson": [".jsonl"] } }],
      });
      const out = await handle.createWritable();
      if (r.body) await r.body.pipeTo(out);
      else { await out.write(await r.blob()); await out.close(); }
      showToast(S.chat.exportSaved(handle.name || name));
      return;
    } catch (e: any) {
      if (e?.name === "AbortError") { void r.body?.cancel(); return; }
      if (r.bodyUsed) { showToast(S.chat.exportFailed(String(e?.message || e)), 4000); return; }
      // No picker allowed here (lost user activation, embedder policy):
      // fall through to the browser's own download.
    }
  }
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** The composer text, minus a lone `/command` the user was typing. */
function clearIfSlash() {
  const text = getState().inputText;
  return slashQuery(text) != null ? "" : text;
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
  // In the transcript from the start, so progress shows where the result will
  // land rather than in a bar under the composer.
  const aiMsg: ChatMsg = { role: "assistant", content: "", ts: Date.now(), durationSec: 0, compact: { pending: true } };
  sess.msgs.push(aiMsg);
  bump();
  try {
    const { ok, data, status } = await api.compactSessionApi(sessId, instructions);
    if (!ok) {
      const detail = (data && (data.detail || data.error)) || `HTTP ${status}`;
      delete aiMsg.compact;
      markChatError(aiMsg, S.chat.compactFailed(String(detail)));
    } else {
      aiMsg.content = S.chat.compactOk(data.messages_before, data.messages_after);
      aiMsg.compact = {
        before: data.messages_before,
        after: data.messages_after,
        notes: data.notes || "",
        notesPath: data.notes_path || "",
      };
      if (data.usage) applySessionUsage(sessId, data.usage);
    }
  } catch (e: any) {
    delete aiMsg.compact;
    markChatError(aiMsg, S.chat.compactFailed(String(e?.message || e)));
  } finally {
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
    await applyPendingModel(sessId);
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
      {src ? <PreviewableImage src={src} alt={file.name} /> : <span>{file.name}</span>}
      <span className="fx" onClick={onRemove}><IconClose /></span>
    </div>
  );
}

