import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import * as api from "../api";
import { ComposerDir } from "../components/ComposerDir";
import { promptRename } from "../components/SessionTree";
import { SlashMenu, SkillsPicker, filterSlashItems, slashQuery, webSlashItems, type SlashItem } from "../components/SlashMenu";
import { getStrings, useStrings } from "../i18n";
import { fmtCost, fmtDurationMs, fmtN, fmtTps, uid } from "../lib/format";
import { primeSettings, switchProfile } from "../panels/SettingsModal";
import { createSession, syncSessionRoundStats } from "../sessions";
import { loadPlugins } from "../data";
import {
  Logo, IconPlus, IconClose, IconSidebar, IconChat, IconAsk, IconAuto,
  IconAllowAll, IconSend, IconStop, IconCopy, IconArrowDown, IconBook, IconDatabase,
  IconChevronDown,
} from "../icons";
import {
  bump, getState, pushPrefs, saveSessions, setState,
  showToast, useAppState, type ChatMsg,
} from "../store";
import { applySessionUsage, ContextUsageTip } from "../components/ContextUsageTip";
import { FilesPanel } from "../workspace/FilesPanel";
import { MessageFilesCard } from "../workspace/MessageFilesCard";
import { useFilesPanel } from "../workspace/useFilesPanel";

function permIcon(mode: string) {
  if (mode === "ask") return <IconAsk />;
  if (mode === "allow-all") return <IconAllowAll />;
  return <IconAuto />;
}

export function ChatPage() {
  const s = useAppState();
  const S = useStrings();
  const taRef = useRef<HTMLTextAreaElement>(null);
  const msgsRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const [slashActive, setSlashActive] = useState(0);
  const [skillsOpen, setSkillsOpen] = useState(false);

  const streaming = !!s.streams[s.curSess || ""];
  const goalRun = s.curSess ? s.goalRuns[s.curSess] : undefined;
  const busy = streaming || !!goalRun;
  const cur = s.curSess ? s.sessions[s.curSess] : null;
  const queued = s.messageQueue.filter((q) => q.sessionId === s.curSess);
  const q = slashQuery(s.inputText);
  const slashItems = q != null
    ? filterSlashItems(webSlashItems(s.pluginsData.skills || [], S.chat), q)
    : [];
  const slashOpen = q != null;
  const pendingImages = s.pendingFiles.filter((f) => f.type.startsWith("image/"));

  useEffect(() => { void loadPlugins(); }, []);
  useEffect(() => { if (slashOpen) setSkillsOpen(false); }, [slashOpen]);
  useEffect(() => {
    if (!s.curSess) return;
    void syncSessionRoundStats(s.curSess);
  }, [s.curSess, streaming]);

  useEffect(() => {
    const el = msgsRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [cur?.msgs.length, cur?.msgs[cur.msgs.length - 1]?.content, streaming]);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 220) + "px";
  }, [s.inputText]);

  useEffect(() => { setSlashActive(0); }, [s.inputText]);

  const windowSize = s.serverContextWindow || 128000;
  const occupancy = cur?.contextTokens || cur?.lastInputTokens || 0;
  const ctxPct = Math.min(100, Math.round((occupancy / windowSize) * 100));
  const files = useFilesPanel(s.curSess);
  const workspace = cur?.dir || s.pendingNewChatDir || s.serverDir || "";
  const [ctxReady, setCtxReady] = useState(false);
  useEffect(() => { setCtxReady(false); }, [s.curSess]);

  return (
    <>
      <div className="main">
        <div className="topbar">
          <div className="tb-l">
            <button className="ib sidebar-expand-btn" onClick={() => setState({ sidebarCollapsed: false })} title={S.nav.expand}><IconSidebar /></button>
            <div className="tb-chat-info">
              {cur && <span className="tb-chat-icon"><IconChat /></span>}
              <span className="tb-chat-title" title={S.chat.dblclickRename} onDoubleClick={() => s.curSess && promptRename(s.curSess)}>
                {cur?.title || S.chat.newConversation}
              </span>
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
        <div className="chat-wrap">
          <div className="chat" id="chatArea" ref={msgsRef} onScroll={() => {
            const el = msgsRef.current;
            if (!el) return;
            const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
            stickRef.current = atBottom;
            setShowJump(!atBottom && !!(cur && cur.msgs.length));
          }}>
            <div className="msgs">
              {!cur || !cur.msgs.length ? (
                <div className="welcome welcome-new">
                  <Logo className="w-icon-img" />
                  <h2>Agentica</h2>
                  <p>{S.chat.welcome}</p>
                </div>
              ) : cur.msgs.map((m, i) => (
                <MessageView key={i} m={m} workspace={workspace} onOpenFile={files.browsePath}
                  live={streaming && i === cur.msgs.length - 1 && m.role === "assistant"} />
              ))}
            </div>
          </div>
          {showJump && (
            <button type="button" className="scroll-bottom-btn visible" title={S.chat.jumpLatest}
                    onClick={() => {
                      const el = msgsRef.current;
                      if (el) el.scrollTop = el.scrollHeight;
                      stickRef.current = true;
                      setShowJump(false);
                    }}>
              <IconArrowDown />
            </button>
          )}
        </div>
        <div className="input-area">
          {goalRun && (
            <div className="goal-bar" title={goalRun.progress}>
              <span className="goal-bar-status">{S.chat.goalBar(goalRun.status, goalRun.objective)}</span>
              {goalRun.progress && <span className="goal-bar-progress">{goalRun.progress}</span>}
            </div>
          )}
          {!!queued.length && (
            <div className="queue-bar">
              <span className="queue-label">{S.chat.queuedCount(queued.length)}</span>
              {queued.map((qItem) => (
                <span className="queue-chip" key={qItem.id} title={qItem.text}>
                  {qItem.text.slice(0, 40) || S.chat.attachmentOnly}
                  <button onClick={() => setState({ messageQueue: getState().messageQueue.filter((x) => x.id !== qItem.id) })}><IconClose /></button>
                </span>
              ))}
            </div>
          )}
          <div className="input-box" onPaste={(e) => {
            const files = imageFilesFromClipboard(e.clipboardData);
            if (!files.length) return;
            e.preventDefault();
            setState({ pendingFiles: [...getState().pendingFiles, ...files] });
          }} onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("dragover"); }}
             onDragLeave={(e) => e.currentTarget.classList.remove("dragover")}
             onDrop={(e) => {
               e.preventDefault();
               e.currentTarget.classList.remove("dragover");
               const files = Array.from(e.dataTransfer.files || []);
               if (files.length) setState({ pendingFiles: [...getState().pendingFiles, ...files] });
             }}>
            {slashOpen && (
              <SlashMenu items={slashItems} active={slashActive} onPick={(it) => pickSlash(it, taRef)} />
            )}
            {s.pendingFiles.length > 0 && (
              <div className="file-list">
                {s.pendingFiles.map((f, i) => (
                  <PendingFileChip key={i} file={f} onRemove={() => setState({ pendingFiles: s.pendingFiles.filter((_, j) => j !== i) })} />
                ))}
              </div>
            )}
            {pendingImages.length > 0 && !s.serverSupportsImages && (
              <div className="vision-note">{S.chat.visionFallback(s.serverMediaModel, s.serverConfigPath)}</div>
            )}
            <textarea
              ref={taRef}
              className="input-ta"
              rows={1}
              placeholder={busy ? S.chat.placeholderStreaming : S.chat.placeholder}
              value={s.inputText}
              onChange={(e) => setState({ inputText: e.target.value })}
              onKeyDown={(e) => {
                if (slashOpen && slashItems.length) {
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
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void submit();
                }
              }}
            />
            <div className="input-foot">
              <div className="input-foot-l">
                <input type="file" multiple accept="image/*,*/*" style={{ display: "none" }} id="fileInput" onChange={(e) => {
                  const files = Array.from(e.target.files || []);
                  if (files.length) setState({ pendingFiles: [...getState().pendingFiles, ...files] });
                  e.currentTarget.value = "";
                }} />
                <button className="foot-btn plus-btn" title={S.chat.attach} onClick={() => document.getElementById("fileInput")?.click()}><IconPlus /></button>
                <div className="skills-wrap">
                  <button type="button" className="foot-btn skills-btn" title={S.chat.slashSkill}
                          onClick={() => {
                            setState({ approvalMenuOpen: false });
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
                <div className="approval-wrap">
                  <button className="foot-btn approval-btn" title={S.chat.permTip(s.selectedApprovalMode)}
                          onClick={() => {
                            setSkillsOpen(false);
                            setState({ approvalMenuOpen: !s.approvalMenuOpen });
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
                <div className="ctx-wrap" onMouseEnter={() => setCtxReady(true)}>
                  <span className="foot-btn input-ctx" title={S.chat.ctxUsage}>
                    <span className="ctx-ring" style={{ ["--pct" as any]: ctxPct + "%" }} />
                    {ctxPct}%
                  </span>
                  {ctxReady && (
                    <ContextUsageTip sessionId={s.curSess} fallback={cur} />
                  )}
                </div>
                <div className="input-model-wrap">
                  <button className="foot-btn model-sel" onClick={() => setState({ modelDDOpen: !s.modelDDOpen })}>
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
                <button className={"act-btn " + (streaming ? "stop" : "send")} onClick={() => streaming ? stopGen() : void submit()}>
                  {streaming ? <IconStop /> : <IconSend />}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <FilesPanel root={workspace} panel={files} />
    </>
  );
}

function MessageView({ m, workspace, onOpenFile, live }: {
  m: ChatMsg; workspace: string; onOpenFile: (path: string) => void; live?: boolean;
}) {
  const S = useStrings();
  return (
    <div className={"m " + (m.role === "user" ? "m-u" : "m-a")}>
      <div className="msg-stack">
        {m.role === "assistant" && m.steps && m.steps.length > 0 && (
          <details className="steps-block" open>
            <summary className="steps-summary">
              {m.durationSec ? S.chat.ranFor(Math.round(m.durationSec)) : S.chat.running}
              {" · "}
              {S.chat.toolCalls(m.steps.filter((x) => x.type === "tool").length)}
            </summary>
            {m.steps.map((st, i) => (
              st.type === "thinking" ? (
                <div className="step-think" key={i}>{st.text}</div>
              ) : (
                <details className="step-tool" key={i}>
                  <summary>
                    <span className="step-tool-name">{st.name}</span>
                    <span className="step-tool-args">{st.argsStr}</span>
                  </summary>
                  {st.argsStr && <pre className="step-pre">{st.argsStr}</pre>}
                  {st.result != null && <pre className="step-pre out">{String(st.result).slice(0, 8000)}</pre>}
                </details>
              )
            ))}
          </details>
        )}
        {(m.content || m.previews?.length || !m.error) && (
          <div className="bub">
            {!!m.previews?.length && (
              <div className="msg-previews">
                {m.previews.map((src, i) => <img key={i} src={src} alt="" />)}
              </div>
            )}
            {m.role === "user" ? m.content : (
              <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                {m.content || ""}
              </Markdown>
            )}
          </div>
        )}
        <MessageFilesCard
          text={m.content || ""}
          steps={m.steps}
          uploaded={m.files}
          workspace={workspace || null}
          onOpenFile={onOpenFile}
        />
        {m.aborted && <div className="msg-note">{S.chat.aborted}</div>}
        {m.error && <div className="msg-error">{S.chat.error(m.error)}</div>}
        <MessageFooter m={m} live={!!live} />
      </div>
    </div>
  );
}

function MessageFooter({ m, live }: { m: ChatMsg; live?: boolean }) {
  const S = useStrings();
  if (live) return null;
  const showStats = m.role === "assistant" && m.tokIn != null;
  const durationMs = m.durationMs ?? ((m.durationSec || 0) * 1000);
  const tps = (m.llmMs && m.tokOut)
    ? m.tokOut / (m.llmMs / 1000)
    : ((m.durationSec && m.tokOut) ? m.tokOut / m.durationSec : 0);
  const copy = () => {
    if (!m.content) return;
    void navigator.clipboard.writeText(m.content).then(() => showToast(S.common.copied));
  };
  return (
    <div className={"msg-footer" + (m.role === "user" ? " user" : "")}>
      <span title={S.chat.msgTimeTip}>{S.chat.msgStamp(m.ts || Date.now())}</span>
      {showStats && (
        <>
          <span title={S.chat.ctxInput}>↑ {fmtN(m.tokIn || 0)}</span>
          {(m.cacheRead != null) && (
            <span className="msg-cache" title={`${S.chat.cacheTokensTip} · ${S.chat.cacheHitTip}`}>
              <IconDatabase />
              {fmtN(m.cacheRead)}{m.cacheHitPercent != null ? ` / ${m.cacheHitPercent.toFixed(1)}%` : ""}
            </span>
          )}
          <span title={S.chat.ctxOutput}>↓ {fmtN(m.tokOut || 0)}</span>
          {tps > 0 && <span title={S.chat.tokPerSecTip}>{fmtTps(tps)}</span>}
          {(m.costUsd || 0) > 0 && <span title={S.chat.costUsdTip}>{fmtCost(m.costUsd || 0)}</span>}
          {!!durationMs && <span title={S.chat.durationTip}>{fmtDurationMs(durationMs)}</span>}
        </>
      )}
      <button type="button" className="msg-copy" title={S.common.copy} onClick={copy}>
        <IconCopy />
      </button>
    </div>
  );
}

function stopGen() {
  const st = getState();
  st.streams[st.curSess || ""]?.abortCtrl.abort();
}

/** Enter always accepts input: while a turn or /goal is running the line
 *  is queued rather than dropped, and the queue drains when the run ends. */
function isBusy(sessId: string) {
  const st = getState();
  return !!st.streams[sessId] || !!st.goalRuns[sessId];
}

function enqueueMessage(sessId: string, text: string, files: File[]) {
  setState({
    messageQueue: [...getState().messageQueue, { id: uid(), sessionId: sessId, text, files, ts: Date.now() }],
  });
  showToast(getStrings().chat.queuedToast);
}

function setGoalRun(sessId: string, patch: { status: string; objective: string; progress: string }) {
  setState({ goalRuns: { ...getState().goalRuns, [sessId]: patch } });
}

function clearGoalRun(sessId: string) {
  const next = { ...getState().goalRuns };
  delete next[sessId];
  setState({ goalRuns: next });
}

async function submit() {
  const st = getState();
  const text = st.inputText.trim();
  const files = st.pendingFiles.slice();
  if (!text && !files.length) return;
  if (!st.curSess) {
    const dir = st.pendingNewChatDir || st.serverDir;
    if (!dir) { setState({ dirModal: { open: true, forNewSession: true, value: "" } }); return; }
    createSession(dir);
  }
  const sessId = getState().curSess!;
  const cmd = parseWebCommand(text);
  setState({ inputText: "", pendingFiles: [] });
  if (isBusy(sessId)) {
    enqueueMessage(sessId, text, files);
    return;
  }
  if (cmd) {
    if (cmd.kind === "compact") await runCompact(sessId, cmd.arg);
    else await runGoalCmd(sessId, cmd.arg);
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
    else await runGoalCmd(sessId, cmd.arg);
    return;
  }
  await sendMessage(sessId, next.text, next.files);
}

async function sendMessage(sessId: string, text: string, files: File[]) {
  const st = getState();
  const sess = st.sessions[sessId];
  if (!sess) return;
  const S = getStrings();
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
  if (!message && !images.length) return;

  const userMsg: ChatMsg = { role: "user", content: text || message, ts: Date.now(), files: uploaded, previews };
  sess.msgs.push(userMsg);
  if (sess.msgs.filter((m) => m.role === "user").length === 1) sess.title = (text || message).slice(0, 50);
  sess.ts = Date.now();
  saveSessions();

  const abortCtrl = new AbortController();
  const aiMsg: ChatMsg = { role: "assistant", content: "", steps: [], ts: Date.now(), durationSec: 0 };
  st.streams[sessId] = { abortCtrl, aiMsg };
  sess.msgs.push(aiMsg);
  bump();

  const t0 = performance.now();
  try {
    const resp = await api.streamChat({
      message,
      session_id: sessId,
      work_dir: sess.dir || st.serverDir || "",
      approval_mode: st.selectedApprovalMode,
      images,
    }, abortCtrl.signal);
    // A 4xx/5xx here has no SSE body to read; without this the turn just
    // stopped with an empty bubble and no reason shown.
    if (!resp.ok || !resp.body) {
      let detail = `HTTP ${resp.status}`;
      try {
        const j = await resp.json();
        if (j?.detail) detail = String(j.detail);
      } catch { /* not json */ }
      aiMsg.error = detail;
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let curThinking = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6);
        if (raw === "[DONE]") continue;
        let evt: any;
        try { evt = JSON.parse(raw); } catch { continue; }
        if (evt.event === "thinking") {
          curThinking += evt.data;
          const last = aiMsg.steps![aiMsg.steps!.length - 1];
          if (last && last.type === "thinking") last.text = curThinking;
          else aiMsg.steps!.push({ type: "thinking", text: curThinking });
          bump();
        } else if (evt.event === "tool_call") {
          curThinking = "";
          aiMsg.steps!.push({ type: "tool", name: evt.data.name, argsStr: JSON.stringify(evt.data.args || {}) });
          bump();
        } else if (evt.event === "tool_result") {
          const steps = aiMsg.steps || [];
          for (let i = steps.length - 1; i >= 0; i--) {
            if (steps[i].type === "tool" && steps[i].result == null) { steps[i].result = evt.data.result; break; }
          }
          bump();
        } else if (evt.event === "content") {
          aiMsg.content += evt.data;
          bump();
        } else if (evt.event === "error") {
          aiMsg.error = String(evt.data);
          bump();
        } else if (evt.event === "done" && evt.data) {
          const prevCost = sess.costUsd || 0;
          const turn = evt.data.turn_usage || {};
          aiMsg.durationSec = (typeof evt.data.duration_ms === "number")
            ? evt.data.duration_ms / 1000
            : (evt.data.response_time || (performance.now() - t0) / 1000);
          if (typeof evt.data.duration_ms === "number") aiMsg.durationMs = evt.data.duration_ms;
          if (typeof evt.data.llm_ms === "number") aiMsg.llmMs = evt.data.llm_ms;
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
          if (evt.data.context_window) setState({ serverContextWindow: evt.data.context_window });
          if (evt.data.usage) applySessionUsage(sessId, evt.data.usage);
          const nextCost = getState().sessions[sessId]?.costUsd ?? sess.costUsd;
          aiMsg.costUsd = (typeof turn.cost_usd === "number") ? turn.cost_usd : Math.max(0, (nextCost || 0) - prevCost);
          if (evt.data.media_notes?.length) {
            aiMsg.content = (evt.data.media_notes as string[]).join("\n") + (aiMsg.content ? "\n\n" + aiMsg.content : "");
          }
        }
      }
    }
  } catch (e: any) {
    if (e?.name === "AbortError") aiMsg.aborted = true;
    else aiMsg.error = String(e?.message || e);
  } finally {
    if (!aiMsg.durationSec) aiMsg.durationSec = (performance.now() - t0) / 1000;
    delete getState().streams[sessId];
    if (getState().curSess !== sessId) {
      const other = getState().sessions[sessId];
      if (other) other.unread = true;
    }
    saveSessions();
    bump();
    await drainQueue(sessId);
  }
}

function parseWebCommand(text: string): { kind: "compact" | "goal"; arg: string } | null {
  if (text === "/compact" || text.startsWith("/compact ")) {
    return { kind: "compact", arg: text.slice("/compact".length).trim() };
  }
  if (text === "/goal" || text.startsWith("/goal ")) {
    return { kind: "goal", arg: text.slice("/goal".length).trim() };
  }
  return null;
}

function pickSlash(it: SlashItem, taRef: { current: HTMLTextAreaElement | null }) {
  if (it.cmd === "/compact") {
    setState({ inputText: "" });
    void (async () => {
      const st = getState();
      if (!st.curSess) {
        const dir = st.pendingNewChatDir || st.serverDir;
        if (!dir) { setState({ dirModal: { open: true, forNewSession: true, value: "" } }); return; }
        createSession(dir);
      }
      await runCompact(getState().curSess!, "");
    })();
    return;
  }
  setState({ inputText: it.cmd + " " });
  requestAnimationFrame(() => taRef.current?.focus());
}

function pushLocalTurn(sessId: string, userText: string, assistantText: string, error?: string) {
  const sess = getState().sessions[sessId];
  if (!sess) return;
  sess.msgs.push({ role: "user", content: userText, ts: Date.now() });
  sess.msgs.push({
    role: "assistant", content: assistantText, ts: Date.now(),
    error, durationSec: 0,
  });
  sess.ts = Date.now();
  saveSessions();
  bump();
}

async function runCompact(sessId: string, instructions: string) {
  const S = getStrings();
  try {
    const { ok, data, status } = await api.compactSessionApi(sessId, instructions);
    if (!ok) {
      const detail = (data && (data.detail || data.error)) || `HTTP ${status}`;
      pushLocalTurn(sessId, instructions ? `/compact ${instructions}` : "/compact", "", S.chat.compactFailed(String(detail)));
      return;
    }
    const msg = S.chat.compactOk(data.messages_before, data.messages_after);
    pushLocalTurn(sessId, instructions ? `/compact ${instructions}` : "/compact", msg);
    if (data.usage) applySessionUsage(sessId, data.usage);
  } finally {
    await drainQueue(sessId);
  }
}

async function runGoalCmd(sessId: string, objective: string) {
  const S = getStrings();
  if (!objective || objective === "status") {
    showToast(S.chat.goalNeedObjective);
    setState({ inputText: "/goal " });
    await drainQueue(sessId);
    return;
  }
  const sess = getState().sessions[sessId];
  if (!sess) return;
  sess.msgs.push({ role: "user", content: `/goal ${objective}`, ts: Date.now() });
  const aiMsg: ChatMsg = { role: "assistant", content: "", ts: Date.now() };
  sess.msgs.push(aiMsg);
  saveSessions();
  setGoalRun(sessId, { status: "active", objective, progress: "" });
  bump();
  try {
    const resp = await api.streamGoal({ objective, session_id: sessId });
    if (!resp.ok || !resp.body) {
      let detail = `HTTP ${resp.status}`;
      try {
        const j = await resp.json();
        if (j?.detail) detail = String(j.detail);
      } catch { /* not json */ }
      aiMsg.error = detail;
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6);
        if (raw === "[DONE]") continue;
        let evt: any;
        try { evt = JSON.parse(raw); } catch { continue; }
        if (evt.event === "status" && evt.data) {
          const d = evt.data;
          setGoalRun(sessId, {
            status: d.status || "active",
            objective: d.objective || objective,
            progress: d.progress || "",
          });
        } else if (evt.event === "error") {
          aiMsg.error = String(evt.data);
          bump();
        } else if (evt.event === "done" && evt.data) {
          aiMsg.content = evt.data.content || evt.data.reason || evt.data.status || "";
          bump();
        }
      }
    }
  } catch (e: any) {
    aiMsg.error = String(e?.message || e);
  } finally {
    clearGoalRun(sessId);
    saveSessions();
    bump();
    await drainQueue(sessId);
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

