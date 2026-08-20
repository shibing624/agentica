import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import * as api from "../api";
import { DirModal } from "../components/DirModal";
import { loadCronJobs, loadPlugins, loadProfiles, loadStatus } from "../data";
import { agoStr, fmtN, fmtTime, shortenPath, uid } from "../lib/format";
import { PluginsPanel } from "../panels/PluginsPanel";
import { primeSettings, SettingsModal, switchProfile } from "../panels/SettingsModal";
import {
  archiveSession, createSession, deleteSession, loadSessions, newChat,
  renameSession, switchTo,
} from "../sessions";
import {
  Logo, IconPlus, IconSearch, IconClose, IconClock, IconPlug, IconTrace,
  IconSidebar, IconFolder, IconPencil, IconArchive, IconGear, IconProfiles,
  IconUser, IconChat, IconCopy, IconFinder, IconTerminal, IconAsk, IconAuto,
  IconAllowAll, IconSend, IconStop,
} from "../icons";
import {
  bump, getState, projectIdForDir, projectNameForDir, saveSessions, setState,
  showToast, useAppState, type ChatMsg, type Session,
} from "../store";

export function ChatPage() {
  const s = useAppState();
  const taRef = useRef<HTMLTextAreaElement>(null);
  const msgsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void (async () => {
      await loadStatus();
      await loadSessions();
      await Promise.all([loadProfiles(), loadPlugins(), loadCronJobs()]);
    })();
  }, []);

  const streaming = !!s.streams[s.curSess || ""];
  const cur = s.curSess ? s.sessions[s.curSess] : null;
  const queued = s.messageQueue.filter((q) => q.sessionId === s.curSess);

  useEffect(() => {
    const el = msgsRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [cur?.msgs.length, cur?.msgs[cur.msgs.length - 1]?.content]);

  const ctxPct = Math.min(100, Math.round(((cur?.lastInputTokens || 0) / (s.serverContextWindow || 128000)) * 100));

  return (
    <>
      <aside className={"sidebar" + (s.sidebarCollapsed ? " collapsed" : "")}>
        <div className="side-head">
          <div className="brand"><Logo /><span>Agentica</span></div>
          <button className="ib" onClick={() => setState({ sidebarCollapsed: !s.sidebarCollapsed })} title="收起侧栏"><IconSidebar /></button>
        </div>
        <nav className="side-nav" aria-label="Primary">
          <button className="side-nav-item" onClick={() => newChat()} title="新建对话">
            <span className="nav-icon"><IconPlus /></span>
            <span className="nav-label">新建对话</span>
          </button>
          <label className="side-nav-item side-search-item" title="搜索">
            <span className="nav-icon"><IconSearch /></span>
            <input placeholder="搜索会话" value={s.sidebarSearch} onChange={(e) => setState({ sidebarSearch: e.target.value })} />
            {s.sidebarSearch ? (
              <button className="search-clear" onClick={(e) => { e.preventDefault(); setState({ sidebarSearch: "" }); }}>
                <IconClose />
              </button>
            ) : null}
          </label>
          <button className="side-nav-item" onClick={() => void primeSettings("cron")} title="定时任务">
            <span className="nav-icon"><IconClock /></span>
            <span className="nav-label">定时任务</span>
            {s.cronJobs.length ? <span className="nav-badge">{s.cronJobs.length}</span> : null}
          </button>
          <button className="side-nav-item" onClick={() => { setState({ pluginsPanelOpen: true }); void loadPlugins(); }} title="插件">
            <span className="nav-icon"><IconPlug /></span>
            <span className="nav-label">插件</span>
          </button>
          <Link className="side-nav-item" to="/traces" title="轨迹">
            <span className="nav-icon"><IconTrace /></span>
            <span className="nav-label">轨迹观测</span>
          </Link>
        </nav>
        <div className="project-list-label">项目</div>
        <SessionTree />
        <div className="account-wrap">
          <div className={"account-pop" + (s.accountPanelOpen ? " open" : "")} onClick={(e) => e.stopPropagation()}>
            <div className="account-pop-usage">
              <div className="ctx-tip-header">Token 用量 <span className="account-usage-scope">全部会话</span></div>
              <div className="ctx-tip-row"><span>输入</span><span>{fmtN(accountUsage(s.sessions).tokIn)}</span></div>
              <div className="ctx-tip-row"><span>输出</span><span>{fmtN(accountUsage(s.sessions).tokOut)}</span></div>
              <div className="ctx-tip-row ctx-tip-total"><span>合计</span><span>{fmtN(accountUsage(s.sessions).tokTotal)}</span></div>
              {accountUsage(s.sessions).totalTime > 0 && (
                <div className="ctx-tip-row"><span>耗时</span><span>{fmtTime(accountUsage(s.sessions).totalTime)}</span></div>
              )}
            </div>
            {([["settings", "常规设置", <IconGear key="g" />], ["profiles", "Profile", <IconProfiles key="p" />],
               ["cron", "定时任务", <IconClock key="c" />], ["archived", "归档会话", <IconArchive key="a" />]] as const).map(
              ([tab, label, icon]) => (
                <button key={tab} className="account-action" onClick={() => { setState({ accountPanelOpen: false }); void primeSettings(tab); }}>
                  {icon}<span>{label}</span>
                </button>
              ))}
          </div>
          <button className="account-entry" onClick={() => setState({ accountPanelOpen: !s.accountPanelOpen })} title="账户">
            <span className="account-avatar"><IconUser /></span>
            <span className="account-meta">
              <span className="account-name">{s.serverProfile || "User"}</span>
              <span className="account-sub">用量、归档与设置</span>
            </span>
          </button>
        </div>
      </aside>
      <div className={"account-pop-backdrop" + (s.accountPanelOpen ? " open" : "")} onClick={() => setState({ accountPanelOpen: false })} />
      <div className="main">
        <div className="topbar">
          <div className="tb-l">
            <button className="ib sidebar-expand-btn" onClick={() => setState({ sidebarCollapsed: false })} title="展开侧栏"><IconSidebar /></button>
            <div className="tb-chat-info">
              {cur && <span className="tb-chat-icon"><IconChat /></span>}
              <span className="tb-chat-title" title="双击重命名" onDoubleClick={() => s.curSess && promptRename(s.curSess)}>
                {cur?.title || "新对话"}
              </span>
              {s.curSess && <Link className="trace-link" to={`/traces?sessionId=${s.curSess}`}>查看轨迹</Link>}
            </div>
          </div>
          <div className="tb-r">
            <div className="tb-info">
              <span className="dir-wrap" onClick={() => setState({ dirModal: { open: true, forNewSession: !s.curSess, value: cur?.dir || s.serverDir } })}>
                目录: <b>{shortenPath(cur?.dir || s.serverDir || "-")}</b>
              </span>
              <button className="ib tb-dir-act" title="复制路径" onClick={() => { const p = cur?.dir || s.serverDir; if (p) { void navigator.clipboard.writeText(p); showToast("已复制"); } }}><IconCopy /></button>
              <button className="ib tb-dir-act" title="在 Finder 打开" onClick={() => { const p = cur?.dir || s.serverDir; if (p) void api.openPathApi(p, "finder"); }}><IconFinder /></button>
              <button className="ib tb-dir-act" title="在终端打开" onClick={() => { const p = cur?.dir || s.serverDir; if (p) void api.openPathApi(p, "terminal"); }}><IconTerminal /></button>
            </div>
          </div>
        </div>
        <div className="chat-wrap">
          <div className="chat" id="chatArea">
            <div className="msgs" ref={msgsRef}>
              {!cur || !cur.msgs.length ? (
                <div className="welcome welcome-new">
                  <Logo className="w-icon-img" />
                  <h2>Agentica</h2>
                  <p>今天想做点什么？</p>
                </div>
              ) : cur.msgs.map((m, i) => <MessageView key={i} m={m} />)}
            </div>
          </div>
        </div>
        <div className="input-area">
          {!!queued.length && (
            <div className="queue-bar">
              <span className="queue-label">排队中 {queued.length} 条</span>
              {queued.map((q) => (
                <span className="queue-chip" key={q.id} title={q.text}>
                  {q.text.slice(0, 40) || "(附件)"}
                  <button onClick={() => setState({ messageQueue: getState().messageQueue.filter((x) => x.id !== q.id) })}><IconClose /></button>
                </span>
              ))}
            </div>
          )}
          <div className="input-box">
            {s.pendingFiles.length > 0 && (
              <div className="file-list">
                {s.pendingFiles.map((f, i) => (
                  <div className="file-chip" key={i}>
                    <span>{f.name}</span>
                    <span className="fx" onClick={() => setState({ pendingFiles: s.pendingFiles.filter((_, j) => j !== i) })}><IconClose /></span>
                  </div>
                ))}
              </div>
            )}
            <textarea
              ref={taRef}
              className="input-ta"
              rows={1}
              placeholder={streaming ? "回车可排队，等当前回答结束后发送…" : "发消息…"}
              value={s.inputText}
              onChange={(e) => setState({ inputText: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void submit();
                }
              }}
            />
            <div className="input-foot">
              <div className="input-foot-l">
                <input type="file" multiple style={{ display: "none" }} id="fileInput" onChange={(e) => {
                  const files = Array.from(e.target.files || []);
                  if (files.length) setState({ pendingFiles: [...getState().pendingFiles, ...files] });
                  e.currentTarget.value = "";
                }} />
                <button className="foot-btn plus-btn" title="附件" onClick={() => document.getElementById("fileInput")?.click()}><IconPlus /></button>
                <div className="approval-wrap">
                  <button className="foot-btn approval-btn" onClick={() => setState({ approvalMenuOpen: !s.approvalMenuOpen })}>
                    <span className="quick-icon">{s.selectedApprovalMode === "ask" ? <IconAsk /> : s.selectedApprovalMode === "allow-all" ? <IconAllowAll /> : <IconAuto />}</span>
                    <span className="approval-label">{s.selectedApprovalMode}</span>
                  </button>
                  {s.approvalMenuOpen && (
                    <div className="approval-dd open">
                      {(["ask", "auto", "allow-all"] as const).map((mode) => (
                        <button key={mode} className={"quick-item" + (s.selectedApprovalMode === mode ? " active" : "")}
                                onClick={() => { localStorage.setItem("ag_approval", mode); setState({ selectedApprovalMode: mode, approvalMenuOpen: false }); }}>
                          <span className="quick-icon">{mode === "ask" ? <IconAsk /> : mode === "allow-all" ? <IconAllowAll /> : <IconAuto />}</span>
                          <span>{mode}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <div className="input-foot-r">
                <div className="ctx-wrap">
                  <button className="foot-btn input-ctx" onClick={() => setState({ ctxTipOpen: !s.ctxTipOpen })} title="上下文占用">
                    <span className="ctx-ring" style={{ ["--pct" as any]: ctxPct + "%" }} />
                    {ctxPct}%
                  </button>
                  {s.ctxTipOpen && (
                    <div className="ctx-tip open">
                      <div className="ctx-tip-header">上下文</div>
                      <div className="ctx-tip-row"><span>上次输入</span><span>{fmtN(cur?.lastInputTokens || 0)}</span></div>
                      <div className="ctx-tip-row"><span>窗口</span><span>{fmtN(s.serverContextWindow)}</span></div>
                      <div className="ctx-tip-row"><span>本会话输入</span><span>{fmtN(cur?.tokIn || 0)}</span></div>
                      <div className="ctx-tip-row"><span>本会话输出</span><span>{fmtN(cur?.tokOut || 0)}</span></div>
                      <div className="ctx-tip-row ctx-tip-total"><span>请求数</span><span>{cur?.requests || 0}</span></div>
                    </div>
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
                        <div className="dd-config-row"><span>模型</span><strong>{s.serverProvider}/{s.serverModelName || s.serverModel}</strong></div>
                        {s.serverReasoningEffort && <div className="dd-config-row"><span>effort</span><strong>{s.serverReasoningEffort}</strong></div>}
                      </div>
                      <div className="dd-section">切换 Profile</div>
                      {!(s.profilesData.profiles || []).length && (
                        <div className="dd-empty">没有 profile，去设置里新建</div>
                      )}
                      {(s.profilesData.profiles || []).map((p: any) => (
                        <button key={p.name} className={"dd-profile" + (p.name === (s.serverProfile || s.profilesData.active) ? " active" : "")}
                                onClick={() => void switchProfile(p.name)}>
                          <div className="dd-p-name">{p.name}{p.name === (s.serverProfile || s.profilesData.active) ? <span className="dd-active"> ●</span> : null}</div>
                          <div className="dd-p-model">{p.model_provider}/{p.model_name}</div>
                        </button>
                      ))}
                      <button className="dd-manage" onClick={() => { setState({ modelDDOpen: false }); void primeSettings("profiles"); }}>管理 Profile…</button>
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
      {s.dirModal.open && <DirModal />}
      {s.settingsModal.open && <SettingsModal />}
      {s.pluginsPanelOpen && <PluginsPanel />}
    </>
  );
}

function MessageView({ m }: { m: ChatMsg }) {
  return (
    <div className={"m " + (m.role === "user" ? "m-u" : "m-a")}>
      <div className="msg-stack">
        {m.role === "assistant" && m.steps && m.steps.length > 0 && (
          <details className="steps-block" open>
            <summary className="steps-summary">
              {m.durationSec ? `思考并执行 ${Math.round(m.durationSec)}s` : "执行中…"} · {m.steps.filter((x) => x.type === "tool").length} 次工具调用
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
        {!!m.files?.length && (
          <div className="msg-files">
            {m.files.map((f) => <span className="file-chip" key={f} title={f}>{f.split("/").pop()}</span>)}
          </div>
        )}
        {(m.content || !m.error) && (
          <div className="bub">
            {m.role === "user" ? m.content : <Markdown remarkPlugins={[remarkGfm]}>{m.content || ""}</Markdown>}
          </div>
        )}
        {m.aborted && <div className="msg-note">已中止</div>}
        {m.error && <div className="msg-error">出错了：{m.error}</div>}
      </div>
    </div>
  );
}

function SessionTree() {
  const s = useAppState();
  const q = s.sidebarSearch.toLowerCase();
  const groups = useMemo(() => {
    const by: Record<string, { id: string; name: string; dir: string; sessions: { id: string; session: Session }[] }> = {};
    for (const [id, sess] of Object.entries(s.sessions)) {
      if (sess.archived) continue;
      if (q && !sess.title.toLowerCase().includes(q)) continue;
      const pid = sess.projectId || projectIdForDir(sess.dir);
      if (!by[pid]) by[pid] = { id: pid, name: projectNameForDir(sess.dir), dir: sess.dir, sessions: [] };
      by[pid].sessions.push({ id, session: sess });
    }
    return Object.values(by).sort((a, b) => (b.sessions[0]?.session.ts || 0) - (a.sessions[0]?.session.ts || 0));
  }, [s.sessions, s.sidebarSearch]);
  return (
    <div className="s-list">
      {!groups.length && <div className="s-empty">{q ? "没有匹配的会话" : "还没有会话"}</div>}
      {groups.map((g) => (
        <div className="p-group" key={g.id}>
          <div className="p-head">
            <div className="p-main">
              <div className="p-title">
                <span className="p-icon"><IconFolder /></span>
                <span className="p-title-text">{g.name}</span>
                <span className="p-count">{g.sessions.length}</span>
              </div>
              <div className="p-dir" title={g.dir}>{shortenPath(g.dir)}</div>
            </div>
          </div>
          {g.sessions.sort((a, b) => b.session.ts - a.session.ts).map(({ id, session }) => (
            <div key={id} className={"s-item" + (id === s.curSess ? " active" : "")} onClick={() => switchTo(id)}>
              <div className="s-main">
                <span className="ti">{session.title}</span>
              </div>
              <span className="mt">{agoStr(session.ts)}</span>
              <div className="s-actions">
                <button className="db" title="重命名" onClick={(e) => { e.stopPropagation(); promptRename(id); }}><IconPencil /></button>
                <button className="db" title="归档" onClick={(e) => { e.stopPropagation(); archiveSession(id); }}><IconArchive /></button>
                <button className="db" title="删除" onClick={(e) => { e.stopPropagation(); deleteSession(id); }}><IconClose /></button>
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function promptRename(id: string) {
  const sess = getState().sessions[id];
  if (!sess) return;
  const name = window.prompt("重命名会话", sess.title);
  if (name) renameSession(id, name);
}

function accountUsage(sessions: Record<string, Session>) {
  let tokIn = 0, tokOut = 0, tokTotal = 0, totalTime = 0;
  for (const sess of Object.values(sessions)) {
    tokIn += sess.tokIn || 0;
    tokOut += sess.tokOut || 0;
    tokTotal += sess.tokTotal || 0;
    totalTime += sess.totalTime || 0;
  }
  return { tokIn, tokOut, tokTotal, totalTime };
}

function stopGen() {
  const st = getState();
  st.streams[st.curSess || ""]?.abortCtrl.abort();
}

/** Enter always accepts input: while a turn is streaming the line is queued
 *  rather than dropped, and the queue drains when the turn ends. */
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
  setState({ inputText: "", pendingFiles: [] });
  if (getState().streams[sessId]) {
    setState({ messageQueue: [...getState().messageQueue, { id: uid(), sessionId: sessId, text, files, ts: Date.now() }] });
    showToast("已加入队列，当前回答结束后发送");
    return;
  }
  await sendMessage(sessId, text, files);
}

async function drainQueue(sessId: string) {
  const next = getState().messageQueue.find((q) => q.sessionId === sessId);
  if (!next) return;
  setState({ messageQueue: getState().messageQueue.filter((q) => q.id !== next.id) });
  await sendMessage(sessId, next.text, next.files);
}

async function sendMessage(sessId: string, text: string, files: File[]) {
  const st = getState();
  const sess = st.sessions[sessId];
  if (!sess) return;
  let message = text;
  const uploaded: string[] = [];
  for (const f of files) {
    const up = await api.uploadFileApi(f, sess.dir || st.serverDir);
    if (up.ok && up.data?.path) uploaded.push(up.data.path);
    else showToast(`上传失败：${f.name}`, 3000);
  }
  if (uploaded.length) {
    if (!message) message = "我上传了文件：" + uploaded.join(", ");
    else message += "\n\n[附件：" + uploaded.join(", ") + "]";
  }
  if (!message) return;

  const userMsg: ChatMsg = { role: "user", content: text || message, ts: Date.now(), files: uploaded };
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
      user_id: "default",
      work_dir: sess.dir || st.serverDir || "",
      approval_mode: st.selectedApprovalMode,
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
          aiMsg.durationSec = evt.data.response_time || (performance.now() - t0) / 1000;
          sess.tokIn += evt.data.input_tokens || 0;
          sess.tokOut += evt.data.output_tokens || 0;
          sess.tokTotal += evt.data.total_tokens || 0;
          sess.requests += evt.data.requests || 1;
          sess.totalTime += evt.data.response_time || 0;
          sess.lastInputTokens = evt.data.input_tokens || sess.lastInputTokens;
          if (evt.data.context_window) setState({ serverContextWindow: evt.data.context_window });
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
