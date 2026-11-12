import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

globalThis.localStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
};

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const { ChatMarkdown } = await vite.ssrLoadModule("/src/components/ChatMarkdown.tsx");
after(() => vite.close());

function render(text, streaming) {
  return renderToStaticMarkup(React.createElement(ChatMarkdown, { text, streaming }));
}

test("numeric ranges do not turn into strikethrough in chat markdown", () => {
  const text = "- **策略数与\"少打扰\"冲突**：`产品方案:171/:298` 的 P2 = 300~1,500 条/周，按 500 账户算 = 每账户每 2~3 天一条策略，和 §3.4\"没有变化不通知、少而有用\"的承诺张力明显。";
  for (const streaming of [true, false]) {
    const html = render(text, streaming);
    assert.equal(html.includes("<del>"), false, `streaming=${streaming} must not create <del>`);
    assert.match(html, /300~1,500 条\/周/);
    assert.match(html, /每账户每 2~3 天一条策略/);
  }
});

test("double tildes still render strikethrough in chat markdown", () => {
  const text = "~~obsolete~~ and 300~1,500";
  for (const streaming of [true, false]) {
    const html = render(text, streaming);
    assert.match(html, /<del>obsolete<\/del>/);
    assert.match(html, /300~1,500/);
  }
});

test("LLM <br> / <br/> / <br /> render as line breaks in prose and GFM tables without enabling raw HTML", () => {
  const prose =
    "① 199 / 199 题（100%） 严格跨两个不同历史日期（d_old < d_new < d_probe），客户在 d_new 亲口声明“将旧值改为新值、旧值作废”，直接触发自建 Jev 的 ADD_OVERRIDE 归档旧节点（而 ADD-only 的 mem0 无法删除旧节点）；<br>② fact_1（仅新值）与 superseded（仅旧值）100% 零重叠";
  for (const streaming of [true, false]) {
    const html = render(prose, streaming);
    assert.equal(html.includes("&lt;br&gt;"), false, `streaming=${streaming} should render <br> instead of escaping`);
    assert.match(html, /<br\s*\/?>/);
    assert.match(html, /d_old &lt; d_new &lt; d_probe/);
  }

  const table = [
    "| 指标 | **Internal-1108**<br>(4B / 9B %) |",
    "|---|---|",
    "| 覆盖 | ① 第一条<br/>② 第二条<br />③ 第三条 |",
  ].join("\n");
  for (const streaming of [true, false]) {
    const html = render(table, streaming);
    assert.equal(html.includes("&lt;br"), false);
    assert.equal((html.match(/<br\s*\/?>/g) || []).length, 3);
    assert.match(html, /<div class="md-table"><table>/);
  }
});

test("code spans, code fences, and non-br HTML tags stay safely escaped", () => {
  const text = "行内 `<br>`、占位符 <user_id> 与 <script>alert(1)</script>\n\n```html\n<div>a<br>b</div>\n```";
  for (const streaming of [true, false]) {
    const html = render(text, streaming);
    assert.match(html, /<code>&lt;br&gt;<\/code>/);
    assert.match(html, /&lt;user_id&gt;/);
    assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
    assert.equal(html.includes("<script>"), false);
    assert.equal(/<br\s*\/?>/i.test(html), false, `streaming=${streaming} must not emit a real <br> inside code`);
    assert.match(html, streaming ? /a&lt;br&gt;b/ : /&lt;<span class="hljs-name">br<\/span>&gt;/);
  }
});

