// ============ MARKDOWN ============
import { esc } from './utils.js';

export function md(text) {
  if (!text) return '';
  let h = text;
  // Preserve LaTeX formulas before escaping HTML
  const mathBlocks = [];
  // Block math: $$...$$  or \[...\]
  h = h.replace(/\$\$([\s\S]*?)\$\$/g, (_, m) => { mathBlocks.push({ tex: m.trim(), display: true }); return `%%MATH${mathBlocks.length - 1}%%` });
  h = h.replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => { mathBlocks.push({ tex: m.trim(), display: true }); return `%%MATH${mathBlocks.length - 1}%%` });
  // Inline math: $...$  or \(...\)
  h = h.replace(/\$([^\$\n]+?)\$/g, (_, m) => { mathBlocks.push({ tex: m.trim(), display: false }); return `%%MATH${mathBlocks.length - 1}%%` });
  h = h.replace(/\\\((.*?)\\\)/g, (_, m) => { mathBlocks.push({ tex: m.trim(), display: false }); return `%%MATH${mathBlocks.length - 1}%%` });

  // Extract code blocks first to protect them from escaping and <br> conversion
  const codeBlocks = [];
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const escaped = code.trim().replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    codeBlocks.push(`<pre><code class="${lang ? 'language-' + lang : ''}">${escaped}</code></pre>`);
    return `%%CODE${codeBlocks.length - 1}%%`;
  });

  // Preserve safe inline HTML tags before escaping (br, ul, ol, li, strong, em, p, table, thead, tbody, tr, th, td, hr, sub, sup, del, ins, mark)
  const safeHtmlBlocks = [];
  const safeTagsRe = /<\/?(br|ul|ol|li|strong|em|b|i|p|table|thead|tbody|tr|th|td|hr|sub|sup|del|ins|mark|div|span|a)(\s[^>]*)?\s*\/?>/gi;
  h = h.replace(safeTagsRe, (m) => { safeHtmlBlocks.push(m); return `%%SAFE${safeHtmlBlocks.length - 1}%%` });

  h = h.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Restore safe HTML tags
  h = h.replace(/%%SAFE(\d+)%%/g, (_, i) => safeHtmlBlocks[parseInt(i)] || '');

  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/^###### (.+)$/gm, '<h6>$1</h6>');
  h = h.replace(/^##### (.+)$/gm, '<h5>$1</h5>');
  h = h.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  h = h.replace(/^---+$/gm, '<hr>');
  h = h.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
  h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.*?)\*/g, '<em>$1</em>');
  h = h.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%;border-radius:8px;margin:8px 0">');
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  h = h.replace(/^\|(.+)\|\s*\n\|[-| :]+\|\s*\n((?:\|.+\|\s*\n?)*)/gm, (_, header, body) => {
    const ths = header.split('|').map(s => s.trim()).filter(Boolean).map(s => `<th>${s}</th>`).join('');
    const rows = body.trim().split('\n').map(row => {
      const tds = row.split('|').map(s => s.trim()).filter(Boolean).map(s => `<td>${s}</td>`).join('');
      return `<tr>${tds}</tr>`;
    }).join('');
    return `<table><thead><tr>${ths}</tr></thead><tbody>${rows}</tbody></table>`;
  });
  h = h.replace(/^(\d+)\. (.+)$/gm, '<oli>$2</oli>');
  h = h.replace(/((<oli>.*<\/oli>\n?)+)/g, m => '<ol>' + m.replace(/<\/?oli>/g, s => s.replace('oli', 'li')).replace(/\n/g, '') + '</ol>');
  h = h.replace(/^[-*] (.+)$/gm, '<uli>$1</uli>');
  h = h.replace(/((<uli>.*<\/uli>\n?)+)/g, m => '<ul>' + m.replace(/<\/?uli>/g, s => s.replace('uli', 'li')).replace(/\n/g, '') + '</ul>');
  h = h.replace(/\n\n/g, '</p><p>');
  h = h.replace(/\n/g, '<br>');
  h = '<p>' + h + '</p>';
  h = h.replace(/<p><\/p>/g, '');
  h = h.replace(/<p>(<(?:h[1-6]|pre|table|ul|ol|blockquote|hr|div)[^>]*>)/g, '$1');
  h = h.replace(/(<\/(?:h[1-6]|pre|table|ul|ol|blockquote|hr|div)>)<\/p>/g, '$1');
  h = h.replace(/<br>(<(?:ul|ol|h[1-6]|pre|table|blockquote|hr|div)[^>]*>)/g, '$1');
  h = h.replace(/(<\/(?:ul|ol|h[1-6]|pre|table|blockquote|hr|div)>)<br>/g, '$1');

  // Restore code blocks (protected from <br> conversion)
  h = h.replace(/%%CODE(\d+)%%/g, (_, i) => codeBlocks[parseInt(i)] || '');

  // Render LaTeX math
  h = h.replace(/%%MATH(\d+)%%/g, (_, i) => {
    const m = mathBlocks[parseInt(i)];
    if (!m) return '';
    try {
      if (typeof katex !== 'undefined') {
        return katex.renderToString(m.tex, { displayMode: m.display, throwOnError: false, strict: false });
      }
    } catch (e) { }
    return m.display ? `<div class="katex-display">${esc(m.tex)}</div>` : `<span>${esc(m.tex)}</span>`;
  });
  return h;
}
