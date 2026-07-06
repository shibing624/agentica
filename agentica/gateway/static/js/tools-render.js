// ============ TOOL RESULT RENDERING (icons, arg formatting, diff tiers) ============
import { esc, shortenFilePath } from './utils.js';

const TOOL_ICONS = {
  ls: '📁', read_file: '📖', write_file: '✏️', edit_file: '✂️', multi_edit_file: '✂️',
  glob: '🔍', grep: '🔎', execute: '⚡', web_search: '🌐',
  fetch_url: '🔗', write_todos: '📋', read_todos: '📋',
  task: '🤖', save_memory: '💾', default: '🔧',
};
export function toolIcon(name) { return TOOL_ICONS[name] || TOOL_ICONS.default }

// Display name mapping for tools
const TOOL_DISPLAY = { 'write_todos': 'todos', 'read_todos': 'todos' };
export function toolDisplay(name) { return TOOL_DISPLAY[name] || name }

// Determine section CSS class based on tool type
export function toolSecClass(name) {
  if (name === 'write_todos' || name === 'read_todos') return 'sec-todos';
  if (name === 'task') return 'sec-task';
  return 'sec-tools';
}

// How many tool rows to show before collapsing (for history)
export const TOOL_VISIBLE_LIMIT = 3;

// Tools that need HTML-rendered args (not just plain text escape)
const RICH_TOOLS = new Set(['task', 'write_todos', 'read_todos', 'read_file', 'write_file', 'edit_file', 'multi_edit_file']);
export function isRichTool(name) { return RICH_TOOLS.has(name) }

// Render a single tool row (flat, minimal style)
export function renderToolRow(st) {
  const icon = toolIcon(st.name || '');
  const dname = toolDisplay(st.name || 'tool');
  const argsHtml = isRichTool(st.name) ? fmtToolArgsHtml(st.name, st.rawArgs, st.argsStr) : esc(st.argsStr || '');
  const hasResult = !!st.result;
  const isTodo = (st.name === 'write_todos' || st.name === 'read_todos');
  const isTaskTool = (st.name === 'task');
  const todoBody = isTodo ? fmtTodoBodyHtml(st.rawArgs) : '';
  const taskBody = isTaskTool ? fmtTaskBodyHtml(st.result) : '';
  const hasExtra = hasResult || todoBody || taskBody;
  let h = `<div class="tg-row${hasExtra ? ' has-result' : ''}"${hasExtra ? ` onclick="toggleToolResult(this)"` : ''}>`
  if (hasExtra) h += `<span class="tg-arrow">&#x25B8;</span>`;
  h += `<span class="tg-icon">${icon}</span><span class="tg-name">${esc(dname)}</span><span class="tg-args">${argsHtml}</span></div>`;
  if (hasExtra) {
    let body = '';
    if (todoBody) body += todoBody;
    if (taskBody) body += taskBody;
    if (hasResult && !taskBody) body += fmtToolResultHtml(st.name, st.rawArgs, st.result);
    h += `<div class="tg-result">${body}</div>`;
  }
  return h;
}

// Smart format tool args for display
export function fmtToolArgs(name, args) {
  if (!args || typeof args !== 'object') return args ? String(args) : '';
  try {
    switch (name) {
      case 'read_file': {
        const f = args.file_path || args.file || args.path || args.filename || '';
        const short = shortenFilePath(f);
        const parts = [short];
        if (args.offset != null || args.limit != null) {
          const s = (args.offset || 0) + 1, e = args.limit ? (args.offset || 0) + args.limit : '';
          parts.push(`(L${s}${e ? '-' + e : ''})`);
        } else if (args.start_line != null || args.end_line != null) {
          const s = args.start_line || 1, e = args.end_line || '';
          parts.push(`(L${s}${e ? '-' + e : ''})`);
        }
        return parts.join(' ');
      }
      case 'write_file': {
        const f = args.file_path || args.file || args.path || args.filename || '';
        const short = shortenFilePath(f);
        const lines = args._lines || 0;
        return short + (lines ? ' +' + lines : '');
      }
      case 'edit_file': {
        const f = args.file_path || args.file || args.path || args.filename || '';
        const short = shortenFilePath(f);
        const add = args._diff_add || 0, del = args._diff_del || 0;
        const diff = (add || del) ? ` +${add} -${del}` : '';
        return short + diff;
      }
      case 'execute': {
        return args.command || args.cmd || JSON.stringify(args).slice(0, 200);
      }
      case 'ls': {
        return args.directory || args.path || args.dir || '.';
      }
      case 'glob': {
        const p = args.pattern || ''; const d = args.directory || args.path || '';
        return p + (d ? ' in ' + d : '');
      }
      case 'grep': {
        const p = args.pattern || ''; const d = args.directory || args.path || '';
        return '"' + p + '"' + (d ? ' in ' + d : '');
      }
      case 'web_search': {
        return args.query || args.q || JSON.stringify(args).slice(0, 150);
      }
      case 'fetch_url': {
        return args.url || JSON.stringify(args).slice(0, 200);
      }
      case 'task': {
        return args.description || args.task || JSON.stringify(args).slice(0, 150);
      }
      case 'write_todos': case 'read_todos': {
        return fmtTodoArgs(args);
      }
      case 'save_memory': {
        const c = args.content || args.text || '';
        return c.length > 100 ? c.slice(0, 100) + '…' : c;
      }
      default: {
        const s = JSON.stringify(args);
        return s.length > 200 ? s.slice(0, 200) + '…' : s;
      }
    }
  } catch { return JSON.stringify(args).slice(0, 200) }
}

// Generate HTML body for todo list (used in sec-body)
export function fmtTodoBodyHtml(args) {
  const todos = args && (args.todos || args.items);
  if (!Array.isArray(todos) || !todos.length) return '';
  return '<div class="todo-list">' + todos.map(t => {
    const st = t.status === 'completed' ? '✅' : t.status === 'in_progress' ? '🔄' : t.status === 'cancelled' ? '❌' : '⬜';
    return `<div class="todo-row"><span class="todo-st">${st}</span><span class="todo-txt">${esc(t.content || t.text || '')}</span></div>`;
  }).join('') + '</div>';
}

// Generate HTML body for task (subagent) result — shows inner tool calls + execution summary
export function fmtTaskBodyHtml(resultStr) {
  if (!resultStr) return '';
  let data;
  try { data = JSON.parse(resultStr); } catch { return ''; }
  if (!data || !data._task_meta) return '';
  if (!data.success) {
    return `<div class="task-inner"><div class="task-inner-row"><span class="ti-info" style="color:var(--fg)">⚠ ${esc(data.error || 'Unknown error')}</span></div></div>`;
  }
  const summary = data.tool_calls_summary || [];
  const maxShown = 8;
  let h = '<div class="task-inner">';
  for (let i = 0; i < Math.min(summary.length, maxShown); i++) {
    const tc = summary[i];
    const icon = toolIcon(tc.name || '');
    const info = tc.info || '';
    const shortInfo = info.length > 90 ? info.slice(0, 87) + '…' : info;
    h += `<div class="task-inner-row"><span class="ti-icon">${icon}</span><span class="ti-name">${esc(tc.name || '')}</span><span class="ti-info">${esc(shortInfo)}</span></div>`;
  }
  if (summary.length > maxShown) {
    h += `<div class="task-inner-more">… and ${summary.length - maxShown} more tool calls</div>`;
  }
  const parts = [];
  if (data.tool_count > 0) parts.push(`${data.tool_count} tool uses`);
  if (data.execution_time != null) parts.push(`cost: ${data.execution_time.toFixed(1)}s`);
  if (parts.length) h += `<div class="task-summary">Execution Summary: ${parts.join(', ')}</div>`;
  h += '</div>';
  return h;
}

// ============ TOOL RESULT TIERS (P0-b, frontend plan C) ============
// edit_file/multi_edit_file -> unified line diff from args old/new
// write_file -> content preview (first N lines)
// execute/shell -> head + tail with omitted marker
// read tools -> plain escaped result (already concise)

export function lineDiff(a, b) {
  // LCS-based line diff. a/b are arrays of lines. Returns [{t:'eq'|'add'|'del',s}].
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) {
    dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  }
  const ops = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push({ t: 'eq', s: a[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push({ t: 'del', s: a[i] }); i++; }
    else { ops.push({ t: 'add', s: b[j] }); j++; }
  }
  while (i < n) { ops.push({ t: 'del', s: a[i] }); i++; }
  while (j < m) { ops.push({ t: 'add', s: b[j] }); j++; }
  return ops;
}

export function buildEditDiffHtml(oldStr, newStr, maxLines) {
  maxLines = maxLines || 60;
  const ops = lineDiff((oldStr || '').split('\n'), (newStr || '').split('\n'));
  let h = '<div class="tool-diff">';
  let shown = 0;
  for (const op of ops) {
    if (shown >= maxLines) { h += `<div class="diff-more">… ${ops.length - shown} more lines</div>`; break; }
    const cls = op.t === 'add' ? 'diff-add' : op.t === 'del' ? 'diff-del' : 'diff-ctx';
    const sign = op.t === 'add' ? '+' : op.t === 'del' ? '-' : ' ';
    h += `<div class="diff-line ${cls}"><span class="diff-sign">${sign}</span><span class="diff-txt">${esc(op.s)}</span></div>`;
    shown++;
  }
  h += '</div>';
  return h;
}

export function buildWritePreviewHtml(content, maxLines) {
  maxLines = maxLines || 20;
  const lines = String(content || '').split('\n');
  let h = '<div class="tool-preview">';
  const shown = lines.slice(0, maxLines);
  for (let i = 0; i < shown.length; i++) {
    h += `<div class="pv-line"><span class="pv-ln">${i + 1}</span><span class="pv-txt">${esc(shown[i])}</span></div>`;
  }
  if (lines.length > maxLines) h += `<div class="pv-more">… ${lines.length - maxLines} more lines</div>`;
  h += '</div>';
  return h;
}

export function buildHeadTailHtml(text, headN, tailN) {
  const lines = String(text || '').split('\n');
  if (lines.length <= headN + tailN + 1) return '<div class="tool-output">' + esc(text) + '</div>';
  let h = '<div class="tool-output">';
  h += lines.slice(0, headN).map(l => `<div class="op-line">${esc(l)}</div>`).join('');
  h += `<div class="op-omitted">… ${lines.length - headN - tailN} lines omitted</div>`;
  h += lines.slice(-tailN).map(l => `<div class="op-line">${esc(l)}</div>`).join('');
  h += '</div>';
  return h;
}

export function fmtToolResultHtml(name, rawArgs, result) {
  if (name === 'edit_file') {
    const a = rawArgs || {};
    return buildEditDiffHtml(a.old_string || '', a.new_string || '');
  }
  if (name === 'multi_edit_file') {
    const a = rawArgs || {};
    const edits = a.edits || [];
    if (!edits.length) return esc(result || '');
    let h = '';
    edits.forEach((e, i) => {
      if (i > 0) h += '<div class="diff-sep"></div>';
      h += buildEditDiffHtml(e.old_string || '', e.new_string || '');
    });
    return h;
  }
  if (name === 'write_file') {
    const a = rawArgs || {};
    return buildWritePreviewHtml(a.content || '');
  }
  if (name === 'execute' || name === 'shell') {
    return buildHeadTailHtml(result || '', 8, 8);
  }
  return esc(result || '');
}

// Format todo/task args as readable text
export function fmtTodoArgs(args) {
  if (!args) return '';
  const todos = args.todos || args.items;
  if (Array.isArray(todos)) {
    return todos.map(t => {
      const st = t.status === 'completed' ? '✅' : t.status === 'in_progress' ? '🔄' : t.status === 'cancelled' ? '❌' : '⬜';
      return `${st} ${t.content || t.text || ''}`;
    }).join(' | ');
  }
  const s = JSON.stringify(args);
  return s.length > 200 ? s.slice(0, 200) + '…' : s;
}

// Rich HTML format for tool args (task, todo, file tools)
export function fmtToolArgsHtml(name, args, argsStr) {
  if (!args || typeof args !== 'object') return esc(argsStr || '');
  if (name === 'read_file') {
    const f = args.file_path || args.file || args.path || args.filename || '';
    const short = shortenFilePath(f);
    let h = `<span class="file-path" title="${esc(f)}">${esc(short)}</span>`;
    if (args.offset != null || args.limit != null) {
      const s = (args.offset || 0) + 1, e = args.limit ? (args.offset || 0) + args.limit : '';
      h += ` <span class="line-range">L${s}${e ? '-' + e : ''}</span>`;
    } else if (args.start_line != null || args.end_line != null) {
      const s = args.start_line || 1, e = args.end_line || '';
      h += ` <span class="line-range">L${s}${e ? '-' + e : ''}</span>`;
    }
    return h;
  }
  if (name === 'write_file') {
    const f = args.file_path || args.file || args.path || args.filename || '';
    const short = shortenFilePath(f);
    const lines = args._lines || 0;
    let h = `<span class="file-path" title="${esc(f)}">${esc(short)}</span>`;
    if (lines) h += ` <span class="diff-add">+${lines}</span>`;
    return h;
  }
  if (name === 'edit_file') {
    const f = args.file_path || args.file || args.path || args.filename || '';
    const short = shortenFilePath(f);
    const add = args._diff_add || 0, del = args._diff_del || 0;
    let h = `<span class="file-path" title="${esc(f)}">${esc(short)}</span>`;
    if (add || del) h += ` <span class="diff-add">+${add}</span> <span class="diff-del">-${del}</span>`;
    return h;
  }
  if (name === 'multi_edit_file') {
    const f = args.file_path || args.file || args.path || args.filename || '';
    const short = shortenFilePath(f);
    const add = args._diff_add || 0, del = args._diff_del || 0, cnt = args._edit_count || 0;
    let h = `<span class="file-path" title="${esc(f)}">${esc(short)}</span>`;
    if (cnt) h += ` <span style="opacity:.6">${cnt} edits</span>`;
    if (add || del) h += ` <span class="diff-add">+${add}</span> <span class="diff-del">-${del}</span>`;
    return h;
  }
  if (name === 'task') {
    const desc = args.description || args.task || '';
    const prompt = args.prompt || '';
    let h = `<span style="font-weight:600">${esc(desc)}</span>`;
    if (prompt) {
      const short = prompt.length > 120 ? prompt.slice(0, 120) + '…' : prompt;
      h += `<br><span style="opacity:.6;font-size:10px">${esc(short)}</span>`;
    }
    return h;
  }
  if (name === 'write_todos' || name === 'read_todos') {
    const todos = args.todos || args.items;
    if (Array.isArray(todos)) {
      const icons = todos.map((t, i) => {
        const st = t.status === 'completed' ? '✅' : t.status === 'in_progress' ? '🔄' : t.status === 'cancelled' ? '❌' : '⬜';
        return `${i + 1} ${st}`;
      });
      return `<span style="font-size:11px">${todos.length} items · ${icons.join(' ')}</span>`;
    }
  }
  return esc(argsStr || '');
}
