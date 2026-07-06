// ============ CHAT RENDER + SEND/STREAM ============
import { state, persistSessionMeta } from './state.js';
import { nextTick } from './vendor/petite-vue.js';
import { esc, fmtTime, highlightCode } from './utils.js';
import { md } from './markdown.js';
import {
  toolIcon, toolDisplay, toolSecClass, isRichTool, fmtToolArgsHtml, fmtToolArgs,
  fmtTodoBodyHtml, fmtTaskBodyHtml, renderToolRow, TOOL_VISIBLE_LIMIT,
} from './tools-render.js';
import { showToast, showConfirm, openDirModalForNewSession } from './modals.js';
import { renderSidebar } from './sidebar.js';
import { save, forkSession } from './sessions.js';
import { uploadFiles } from './files.js';
import { streamChat } from './api.js';

// ============ CHAT RENDER ============
export function renderChat() {
  const c = document.getElementById('messages');
  if (!state.curSess || !state.sessions[state.curSess] || !state.sessions[state.curSess].msgs.length) {
    if (!state.curSess || !state.sessions[state.curSess]) {
      // 没有 session — 显示引导选择目录
      c.innerHTML = `<div class="welcome"><img class="w-icon-img" src="${document.querySelector('.brand-logo').src}" alt="logo"><h2>Agentica</h2><p>选择一个项目目录开始工作</p><button class="open-folder-btn" onclick="newSession()">Open Folder</button></div>`;
    } else {
      c.innerHTML = `<div class="welcome"><img class="w-icon-img" src="${document.querySelector('.brand-logo').src}" alt="logo"><h2>Agentica</h2><p>AI Agent — send a message to get started.</p></div>`;
    }
    return;
  }
  c.innerHTML = state.sessions[state.curSess].msgs.map((m, i) => renderMsg(m, i)).join('');
  highlightCode(c);
  scrollEnd();
}

function renderMsg(m, idx) {
  if (m.role === 'user') return renderUserMsg(m, idx);
  return renderAssistantMsg(m, idx);
}

function renderUserMsg(m, idx) {
  let h = `<div class="m m-u"><div class="msg-stack"><div class="bub">${esc(m.content)}`;
  if (m.files && m.files.length) h += m.files.map(f => `<br><span class="m-file">📎 ${esc(f)}</span>`).join('');
  h += `</div>${renderMsgMeta(m)}${renderMsgActions(idx, 'user')}</div></div>`;
  return h;
}

function renderAssistantMsg(m, idx) {
  let h = '<div class="m m-a"><div class="msg-stack">';
  h += renderStepSections(m.steps || [], m.durationSec || 0, false);
  h += `<div class="bub">${md(m.content)}</div>${renderMsgMeta(m)}${renderMsgActions(idx, 'assistant', m)}</div></div>`;
  return h;
}

// Pre-existing helper (kept for parity with the original monolith); not
// currently invoked by any render path, but harmless to retain.
function groupSteps(steps) {
  const sections = [];
  for (const st of steps) {
    if (st.type === 'thinking') {
      const last = sections[sections.length - 1];
      if (last && last.type === 'thinking') last.items.push(st);
      else sections.push({ type: 'thinking', items: [st] });
    } else if (st.type === 'tool') {
      const last = sections[sections.length - 1];
      if (last && last.type === 'tools') last.items.push(st);
      else sections.push({ type: 'tools', items: [st] });
    }
  }
  return sections;
}

function renderStepSections(steps, durationSec, isStreaming) {
  const thinkingItems = steps.filter(st => st.type === 'thinking');
  const toolItems = steps.filter(st => st.type === 'tool');
  let h = '';
  if (thinkingItems.length) h += renderThinkingBlock(thinkingItems, durationSec, isStreaming);
  if (toolItems.length) h += renderToolSection(toolItems);
  return h;
}

function renderThinkingBlock(items, durationSec, isStreaming) {
  const text = items.map(s => s.text).join('\n');
  const label = isStreaming ? '正在思考...' : `已处理 ${Math.max(0, Math.round(durationSec || 0))}s`;
  const open = isStreaming ? ' open' : '';
  const arrowOpen = isStreaming ? ' open' : '';
  return `<div class="think-row has-body" onclick="toggleThinkBody(this)">
    <span class="tg-arrow${arrowOpen}">&#x25B8;</span>
    <span class="think-icon">💭</span>
    <span class="think-lbl">${esc(label)}</span>
  </div><div class="think-body${open}">${esc(text)}</div>`;
}

function renderToolSection(items) {
  let h = '';
  const specials = items.filter(s => s.name === 'write_todos' || s.name === 'read_todos' || s.name === 'task');
  const normals = items.filter(s => s.name !== 'write_todos' && s.name !== 'read_todos' && s.name !== 'task');
  for (const st of specials) {
    const icon = toolIcon(st.name || '');
    const dname = toolDisplay(st.name || 'tool');
    const sclass = toolSecClass(st.name || '');
    const argsHtml = isRichTool(st.name) ? fmtToolArgsHtml(st.name, st.rawArgs, st.argsStr) : esc(st.argsStr || '');
    const isTodo = (st.name === 'write_todos' || st.name === 'read_todos');
    const isTask = (st.name === 'task');
    const todoBody = isTodo ? fmtTodoBodyHtml(st.rawArgs) : '';
    const taskBody = isTask ? fmtTaskBodyHtml(st.result) : '';
    const hasBody = st.result || todoBody || taskBody;
    h += `<div class="sec-block ${sclass}"><div class="sec-toggle"${hasBody ? ' onclick="toggleSec(this)" style="cursor:pointer"' : ' style="cursor:default"'}>
      ${hasBody ? '<span class="arrow">&#x25B8;</span>' : ''}
      <span class="sec-icon">${icon}</span>
      <span class="sec-lbl">${esc(dname)}</span>
      <span class="sec-detail">${argsHtml}</span>
    </div>`;
    if (hasBody) {
      let bodyH = '';
      if (todoBody) bodyH += todoBody;
      if (taskBody) bodyH += taskBody;
      if (st.result && !taskBody) bodyH += `<div class="tg-result open" style="border:none;margin:0;padding:3px 12px">${esc(st.result)}</div>`;
      h += `<div class="sec-body">${bodyH}</div>`;
    }
    h += '</div>';
  }
  if (normals.length) {
    h += '<div class="tool-group">';
    const showAll = normals.length <= TOOL_VISIBLE_LIMIT + 1;
    const visible = showAll ? normals : normals.slice(0, TOOL_VISIBLE_LIMIT);
    const hidden = showAll ? [] : normals.slice(TOOL_VISIBLE_LIMIT);
    for (const st of visible) h += renderToolRow(st);
    if (hidden.length) {
      const gid = 'tg_' + Math.random().toString(36).slice(2, 8);
      h += `<div class="tg-more" onclick="toggleToolGroup(this,'${gid}')">… ${hidden.length} more tools</div>`;
      h += `<div id="${gid}" style="display:none">`;
      for (const st of hidden) h += renderToolRow(st);
      h += '</div>';
    }
    h += '</div>';
  }
  return h;
}

function renderMsgMeta(m) {
  const parts = [];
  if (m.ts) parts.push(new Date(m.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  if (m.durationSec) parts.push(fmtTime(m.durationSec));
  if (!parts.length) return '';
  return `<div class="m-meta">${parts.map(esc).join(' · ')}</div>`;
}

// Icon set follows the Feather-icons visual language (clean, thin outline,
// widely recognized shapes) so message actions read clearly at 14-16px.
const ICON_COPY = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="6" y="6" width="8.7" height="8.7" rx="1.3" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M3.3 10H2.7a1.3 1.3 0 01-1.3-1.3V2.7a1.3 1.3 0 011.3-1.3h6a1.3 1.3 0 011.3 1.3v1" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ICON_EDIT = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M11.55 1.45a1.55 1.55 0 012.2 2.2l-8.2 8.2-3.3.9.9-3.3z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M10.5 2.5l3 3" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>';
const ICON_LIKE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5.75 14h5.1a1.5 1.5 0 001.45-1.12l1.15-4.3A1.5 1.5 0 0012 6.7H9.2l.45-2.55A1.8 1.8 0 007.88 2h-.2L4.5 6.25V14zM2 7h2.5v7H2z" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linejoin="round"/></svg>';
const ICON_DISLIKE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10.25 2h-5.1A1.5 1.5 0 003.7 3.12l-1.15 4.3A1.5 1.5 0 004 9.3h2.8l-.45 2.55A1.8 1.8 0 008.12 14h.2l3.18-4.25V2zM14 9h-2.5V2H14z" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linejoin="round"/></svg>';
const ICON_FORK = '<svg viewBox="0 0 16 16" aria-hidden="true"><line x1="4" y1="2" x2="4" y2="10" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><circle cx="12" cy="4" r="2" fill="none" stroke="currentColor" stroke-width="1.3"/><circle cx="4" cy="12" r="2" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M12 6a6 6 0 01-6 6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>';
const ICON_RETRY = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13.5 8A5.5 5.5 0 104.9 12.3" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/><path d="M13.5 4v4h-4" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function renderMsgActions(idx, role, msg) {
  const copy = `<button class="msg-act" onclick="copyMsg(${idx})" title="复制消息" aria-label="Copy message">${ICON_COPY}</button>`;
  if (role === 'user') {
    return `<div class="m-actions">${copy}<button class="msg-act" onclick="editUserMsg(${idx})" title="编辑并从这里继续" aria-label="Edit message">${ICON_EDIT}</button></div>`;
  }
  return `<div class="m-actions">${copy}
    <button class="msg-act" onclick="forkFromMsg(${idx})" title="Fork from here" aria-label="Fork from here">${ICON_FORK}</button>
    <button class="msg-act" onclick="retryMsg(${idx})" title="重新生成这条回答" aria-label="Retry response">${ICON_RETRY}</button>
  </div>`;
}

export async function copyMsg(idx) {
  const s = state.sessions[state.curSess]; if (!s) return;
  const m = s.msgs[idx]; if (!m) return;
  const text = m.content || '';
  try {
    await navigator.clipboard.writeText(text);
    showToast('已复制', 1200);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    document.execCommand('copy');
    ta.remove();
    showToast('已复制', 1200);
  }
}

export function editUserMsg(idx) {
  if (state.streaming) return;
  const s = state.sessions[state.curSess]; if (!s || !s.msgs[idx] || s.msgs[idx].role !== 'user') return;
  const msg = s.msgs[idx];
  s.msgs.splice(idx);
  s.ts = Date.now();
  s.editedFromMsgIdx = idx;
  persistSessionMeta(state.curSess, { editedFromMsgIdx: idx });
  save(); renderChat(); renderSidebar();
  state.inputText = msg.content || '';
  const ta = document.getElementById('inputTa');
  nextTick(() => autoResize(ta));
  state.pendingFiles = [];
  ta.focus();
  showToast('已载入编辑内容，发送后从这里继续', 1800);
}

export function setMsgFeedback(idx, value) {
  const s = state.sessions[state.curSess]; if (!s || !s.msgs[idx]) return;
  const msg = s.msgs[idx];
  if (msg.role !== 'assistant') return;
  msg.feedback = msg.feedback === value ? null : value;
  s.ts = Date.now();
  save(); renderChat();
}

export function forkFromMsg(idx) {
  if (state.streaming) return;
  if (!state.curSess || !state.sessions[state.curSess]) return;
  forkSession(state.curSess, idx);
}

export function deleteMsg(idx) {
  if (state.streaming) return;
  const s = state.sessions[state.curSess]; if (!s || !s.msgs[idx]) return;
  const msg = s.msgs[idx];
  const remove = () => {
    if (msg.role === 'user') {
      let end = idx + 1;
      while (end < s.msgs.length && s.msgs[end].role !== 'user') end++;
      s.msgs.splice(idx, end - idx);
    } else {
      s.msgs.splice(idx, 1);
    }
    s.ts = Date.now(); save(); renderChat(); renderSidebar();
    showToast('已删除', 1200);
  };
  if (msg.role === 'user') showConfirm('删除这轮用户消息及其后的回答？', remove);
  else remove();
}

export async function retryMsg(idx) {
  if (state.streaming) return;
  const s = state.sessions[state.curSess]; if (!s || !s.msgs[idx]) return;
  let userIdx = idx;
  while (userIdx >= 0 && s.msgs[userIdx].role !== 'user') userIdx--;
  if (userIdx < 0) return;
  const userMsg = s.msgs[userIdx];
  s.msgs.splice(userIdx);
  save(); renderChat(); renderSidebar();
  state.inputText = userMsg.content || '';
  state.pendingFiles = [];
  await sendMessage();
}

export async function retryLastMsg() {
  if (state.streaming) return;
  const s = state.sessions[state.curSess]; if (!s || !s.msgs.length) return;
  let lastUserIdx = -1;
  for (let i = s.msgs.length - 1; i >= 0; i--) { if (s.msgs[i].role === 'user') { lastUserIdx = i; break } }
  if (lastUserIdx < 0) return;
  retryMsg(lastUserIdx);
}

export function toggleSec(el) {
  const arrow = el.querySelector('.arrow');
  const body = el.nextElementSibling;
  const open = body.classList.toggle('open');
  arrow.classList.toggle('open', open);
}

// Toggle tool result visibility (flat row style)
export function toggleToolResult(row) {
  const result = row.nextElementSibling;
  if (!result || !result.classList.contains('tg-result')) return;
  const arrow = row.querySelector('.tg-arrow');
  const open = result.classList.toggle('open');
  if (arrow) arrow.classList.toggle('open', open);
}

// Toggle thinking body visibility (flat row style)
export function toggleThinkBody(row) {
  const body = row.nextElementSibling;
  if (!body || !body.classList.contains('think-body')) return;
  const arrow = row.querySelector('.tg-arrow');
  const open = body.classList.toggle('open');
  if (arrow) arrow.classList.toggle('open', open);
}

// Toggle collapsed tool group
export function toggleToolGroup(moreEl, gid) {
  const group = document.getElementById(gid);
  if (!group) return;
  const show = group.style.display === 'none';
  group.style.display = show ? '' : 'none';
  moreEl.textContent = show ? '… collapse' : '… ' + (group.children.length) + ' more tools';
}

export function scrollEnd() {
  const a = document.getElementById('chatArea');
  state.userScrolledUp = false;
  state._scrollLock = false;
  requestAnimationFrame(() => { a.scrollTop = a.scrollHeight });
  updateScrollBtn();
}

// Check if user is near the bottom (within half a screen)
function isNearBottom() {
  const a = document.getElementById('chatArea');
  return (a.scrollHeight - a.scrollTop - a.clientHeight) < a.clientHeight * 0.5;
}

// Auto-scroll only if user hasn't deliberately scrolled up
export function autoScroll() {
  // 一旦锁定（用户上翻），完全停止自动滚动，只更新按钮
  if (state._scrollLock || state.userScrolledUp) { updateScrollBtn(); return; }
  if (isNearBottom()) {
    scrollEnd();
  } else {
    updateScrollBtn();
  }
}

export function updateScrollBtn() {
  const btn = document.getElementById('scrollBottomBtn');
  if (!btn) return;
  const a = document.getElementById('chatArea');
  const dist = a.scrollHeight - a.scrollTop - a.clientHeight;
  // 流式输出时：只要用户上翻（_scrollLock）或不在底部，就显示下箭头
  // 非流式时：滚动超过半屏才显示
  const show = state.streaming ? (dist > 30) : (dist > a.clientHeight * 0.5);
  btn.classList.toggle('visible', show);
}

// exported for main.js's chatArea scroll/wheel/touchmove listeners
export { isNearBottom };

// ============ SEND / STREAM ============
export function handleKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!state.streaming) sendMessage() } }
export function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 200) + 'px' }

export function onAction() {
  if (state.streaming) { stopGen(); return }
  sendMessage();
}

export async function sendMessage() {
  const originalText = state.inputText.trim();
  let text = originalText;
  if (!text && !state.pendingFiles.length) return;
  if (state.streaming) return;

  if (!state.curSess) {
    openDirModalForNewSession();
    return;
  }
  const s = state.sessions[state.curSess];

  // upload files first
  let uploadedPaths = [];
  if (state.pendingFiles.length) {
    const targetDir = s.dir || state.serverDir || '';
    uploadedPaths = await uploadFiles(targetDir);
    if (!text) text = 'I uploaded files: ' + uploadedPaths.join(', ');
    else text += '\n\n[Attached files: ' + uploadedPaths.join(', ') + ']';
  }

  // add user msg
  const userMsg = { role: 'user', content: originalText || text, ts: Date.now() };
  if (uploadedPaths.length) userMsg.files = uploadedPaths;
  s.msgs.push(userMsg);
  if (s.msgs.filter(m => m.role === 'user').length === 1) {
    s.title = userMsg.content.slice(0, 50);
  }
  s.ts = Date.now();
  save(); renderChat(); renderSidebar();

  // clear
  state.inputText = '';
  const ta = document.getElementById('inputTa');
  ta.style.height = 'auto';

  // stream
  state.streaming = true;
  state.abortCtrl = new AbortController();

  const aiStart = performance.now();
  const aiMsg = { role: 'assistant', content: '', steps: [], ts: Date.now(), durationSec: 0 };
  s.msgs.push(aiMsg);
  appendLive();

  // current thinking accumulator
  let curThinking = '';
  let approxIn = 0, approxOut = 0;
  approxIn = Math.ceil(text.length / 3.5);

  try {
    const composedText = [
      state.goalInput ? `[Goal]\n${state.goalInput}` : '',
      state.skillInput ? `[Skill]\n${state.skillInput}` : '',
      state.toolInput ? `[Tool]\n${state.toolInput}` : '',
      text,
    ].filter(Boolean).join('\n\n');
    const resp = await streamChat({
      message: composedText,
      session_id: state.curSess,
      user_id: 'default',
      work_dir: s.dir || state.serverDir || '',
      goal: state.goalInput,
      skill: state.skillInput,
      tool: state.toolInput,
      approval_mode: state.selectedApprovalMode,
    }, state.abortCtrl.signal);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6); if (raw === '[DONE]') continue;
        try {
          const evt = JSON.parse(raw);
          if (evt.event === 'thinking') {
            // Accumulate thinking text into current thinking step
            curThinking += evt.data;
            approxOut += Math.ceil(evt.data.length / 3.5);
            const last = aiMsg.steps[aiMsg.steps.length - 1];
            if (last && last.type === 'thinking') {
              last.text = curThinking;
            } else {
              aiMsg.steps.push({ type: 'thinking', text: curThinking });
            }
            updateLiveSteps(aiMsg);
          } else if (evt.event === 'tool_call') {
            // Reset thinking accumulator — next thinking chunk starts fresh
            curThinking = '';
            const name = evt.data.name || 'tool';
            const argsStr = fmtToolArgs(name, evt.data.args);
            aiMsg.steps.push({ type: 'tool', name: name, text: name, argsStr: argsStr, rawArgs: evt.data.args });
            updateLiveSteps(aiMsg);
          } else if (evt.event === 'tool_result') {
            // Attach result to the last matching tool step (by name, fallback to last tool)
            const rName = evt.data && evt.data.name ? evt.data.name : null;
            const res = evt.data && evt.data.result ? evt.data.result : (typeof evt.data === 'string' ? evt.data : JSON.stringify(evt.data));
            let target = null;
            if (rName) {
              for (let i = aiMsg.steps.length - 1; i >= 0; i--) {
                if (aiMsg.steps[i].type === 'tool' && aiMsg.steps[i].name === rName && !aiMsg.steps[i].result) { target = aiMsg.steps[i]; break }
              }
            }
            if (!target) target = findLastTool(aiMsg.steps);
            if (target) target.result = (target.result || '') + res;
            updateLiveSteps(aiMsg);
          } else if (evt.event === 'content') {
            if (curThinking) { curThinking = '' }
            aiMsg.content += evt.data;
            approxOut += Math.ceil(evt.data.length / 3.5);
            updateLiveContent(aiMsg);
          } else if (evt.event === 'done') {
            if (evt.data) {
              const gotIn = evt.data.input_tokens || 0;
              const gotOut = evt.data.output_tokens || 0;
              const gotTotal = evt.data.total_tokens || 0;
              const gotReqs = evt.data.requests || 0;
              const gotTime = evt.data.response_time || 0;
              aiMsg.durationSec = gotTime || ((performance.now() - aiStart) / 1000);
              const entries = evt.data.request_entries || [];
              if (evt.data.context_window) state.serverContextWindow = evt.data.context_window;
              if (gotIn > 0 || gotOut > 0) {
                s.tokIn = (s.tokIn || 0) + gotIn;
                s.tokOut = (s.tokOut || 0) + gotOut;
                s.tokTotal = (s.tokTotal || 0) + (gotTotal || (gotIn + gotOut));
              } else {
                s.tokIn = (s.tokIn || 0) + approxIn;
                s.tokOut = (s.tokOut || 0) + approxOut;
                s.tokTotal = (s.tokTotal || 0) + approxIn + approxOut;
              }
              // 保存最后一次请求的 input_tokens（代表当前上下文实际占用）
              if (entries.length > 0) {
                const lastEntry = entries[entries.length - 1];
                s.lastInputTokens = lastEntry.input_tokens || gotIn || 0;
              } else if (gotIn > 0) {
                s.lastInputTokens = gotIn;
              } else {
                // fallback: 用近似值
                s.lastInputTokens = approxIn;
              }
              s.requests = (s.requests || 0) + (gotReqs || 1);
              s.totalTime = (s.totalTime || 0) + gotTime;
              if (!s.usageEntries) s.usageEntries = [];
              const baseIdx = s.usageEntries.length;
              for (let i = 0; i < entries.length; i++) {
                s.usageEntries.push({ ...entries[i], request_index: baseIdx + i + 1 });
              }
              if (!entries.length) {
                const entryIn = gotIn || approxIn;
                const entryOut = gotOut || approxOut;
                s.usageEntries.push({ request_index: baseIdx + 1, input_tokens: entryIn, output_tokens: entryOut, total_tokens: gotTotal || (entryIn + entryOut), response_time: gotTime || undefined });
              }
            }
          } else if (evt.event === 'error') {
            aiMsg.content += '\n\n**Error:** ' + evt.data;
            updateLiveContent(aiMsg);
          }
        } catch { }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') aiMsg.content += '\n\n**Error:** ' + err.message;
    else aiMsg.content += aiMsg.content ? '\n\n*(stopped)*' : '*(stopped)*';
  }

  if (!aiMsg.durationSec) aiMsg.durationSec = (performance.now() - aiStart) / 1000;
  state.streaming = false; state.abortCtrl = null; state.userScrolledUp = false; state._scrollLock = false;
  state.goalInput = '';
  state.skillInput = '';
  state.toolInput = '';
  updateScrollBtn();
  s.ts = Date.now(); save();
  renderChat(); renderSidebar();
  document.getElementById('inputTa').focus();
}

function findLastTool(steps) {
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].type === 'tool') return steps[i];
  }
  return null;
}

export function stopGen() { if (state.abortCtrl) state.abortCtrl.abort() }

// ============ LIVE DOM ============
function appendLive() {
  const c = document.getElementById('messages');
  const w = c.querySelector('.welcome'); if (w) w.remove();
  const div = document.createElement('div');
  div.className = 'm m-a streaming'; div.id = 'live';
  div.innerHTML = `<div class="msg-stack"><div id="live-sections"></div>
    <div class="bub" id="live-bub"></div></div>`;
  c.appendChild(div); scrollEnd();
}

function updateLiveContent(msg) {
  const el = document.getElementById('live-bub');
  if (el) { el.innerHTML = md(msg.content); highlightCode(el); autoScroll() }
}

function updateLiveSteps(msg) {
  const el = document.getElementById('live-sections');
  if (!el) return;
  el.innerHTML = renderStepSections(msg.steps || [], msg.durationSec || 0, true);
  const bodies = el.querySelectorAll('.think-body.open');
  if (bodies.length) { const last = bodies[bodies.length - 1]; last.scrollTop = last.scrollHeight };
  autoScroll();
}

