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
const { ChatMarkdown, MarkdownLink } = await vite.ssrLoadModule("/src/components/ChatMarkdown.tsx");
const M = await vite.ssrLoadModule("/src/lib/markdownPlugins.ts");
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

const PDF_URL = "file:///Users/xuming/Documents/Codes/memhub/exp/memhub_exp/paper/acl_industry/arxiv_final.pdf";
const PDF_PATH = "/Users/xuming/Documents/Codes/memhub/exp/memhub_exp/paper/acl_industry/arxiv_final.pdf";
const TGZ_URL = "file:///Users/xuming/Documents/Codes/memhub/tmp/arxiv_memory_preprint_source.tar.gz";
const TGZ_PATH = "/Users/xuming/Documents/Codes/memhub/tmp/arxiv_memory_preprint_source.tar.gz";

test("local file URLs become gateway content URLs with a local_file marker", () => {
  const href = M.markdownUrlTransform(PDF_URL);
  const url = new URL(href, "http://app.local");
  assert.equal(url.pathname, "/api/workspace/content");
  assert.equal(url.searchParams.get("local_file"), "1");
  assert.equal(url.searchParams.get("root"), "/Users/xuming/Documents/Codes/memhub/exp/memhub_exp/paper/acl_industry");
  assert.equal(url.searchParams.get("path"), "arxiv_final.pdf");
  assert.equal(url.searchParams.get("download"), null);

  const encoded = M.markdownUrlTransform("computer:///Users/xuming/tmp/a%20b.md");
  const encodedUrl = new URL(encoded, "http://app.local");
  assert.equal(encodedUrl.searchParams.get("root"), "/Users/xuming/tmp");
  assert.equal(encodedUrl.searchParams.get("path"), "a b.md");
  assert.equal(encodedUrl.searchParams.get("download"), null);

  const windows = M.markdownUrlTransform("file:///C:/Users/xuming/a.pdf");
  const windowsUrl = new URL(windows, "http://app.local");
  assert.equal(windowsUrl.searchParams.get("root"), "C:/Users/xuming");
  assert.equal(windowsUrl.searchParams.get("path"), "a.pdf");

  const archive = new URL(M.markdownUrlTransform(TGZ_URL), "http://app.local");
  assert.equal(archive.searchParams.get("download"), "1");
  assert.equal(M.markdownUrlTransform("https://example.com/a?b=1"), "https://example.com/a?b=1");
  assert.equal(M.markdownUrlTransform("javascript:alert(1)"), "");
});

test("a generated local-file href resolves back to its absolute path and previewability", () => {
  assert.deepEqual(M.localFileLinkTarget(M.markdownUrlTransform(PDF_URL)), { path: PDF_PATH, previewable: true });
  assert.deepEqual(M.localFileLinkTarget(M.markdownUrlTransform(TGZ_URL)), { path: TGZ_PATH, previewable: false });
  assert.deepEqual(
    M.localFileLinkTarget(M.markdownUrlTransform("computer:///Users/xuming/tmp/a%20b.md")),
    { path: "/Users/xuming/tmp/a b.md", previewable: true },
  );
  assert.deepEqual(
    M.localFileLinkTarget(M.markdownUrlTransform("file:///C:/Users/xuming/a.pdf")),
    { path: "C:/Users/xuming/a.pdf", previewable: true },
  );
  assert.equal(M.localFileLinkTarget("/api/workspace/content?root=/tmp&path=a.pdf"), null);
  assert.equal(M.localFileLinkTarget("https://example.com"), null);
  assert.equal(M.localFileLinkTarget(""), null);
});

test("local file links open by path; unsupported files use the OS default app", () => {
  const previewed = [];
  const opened = [];
  const external = (path, app) => { opened.push([path, app]); return Promise.resolve({ ok: true }); };
  M.openLocalFile({ path: PDF_PATH, previewable: true }, (p) => previewed.push(p), external);
  assert.deepEqual(previewed, [PDF_PATH]);
  assert.deepEqual(opened, []);

  M.openLocalFile({ path: TGZ_PATH, previewable: false }, (p) => previewed.push(p), external);
  assert.deepEqual(previewed, [PDF_PATH]);
  assert.deepEqual(opened, [[TGZ_PATH, "finder"]]);

  M.openLocalFile({ path: PDF_PATH, previewable: true }, undefined, external);
  assert.deepEqual(opened, [[TGZ_PATH, "finder"], [PDF_PATH, "finder"]]);
});

test("chat markdown never keeps an empty or file-scheme href for local files", () => {
  for (const streaming of [true, false]) {
    const html = render(`[新版 arxiv_final.pdf](${PDF_URL})`, streaming);
    assert.equal(html.includes('href=""'), false, `streaming=${streaming} must not render href=""`);
    assert.equal(html.includes('href="file:'), false, `streaming=${streaming} must not render a raw file URL`);
    assert.match(html, /<a [^>]*href="\/api\/workspace\/content\?[^"]*local_file=1"/);
    assert.ok(html.includes(`data-open-file="${PDF_PATH}"`), html);
  }
});

test("a markdown link click calls the preview callback with the decoded path and prevents navigation", () => {
  const opened = [];
  const href = M.markdownUrlTransform(PDF_URL);
  const el = MarkdownLink({ href, children: "pdf", onOpenFile: (p) => opened.push(p) });
  assert.equal(el.type, "a");
  assert.equal(el.props.href, href);
  assert.equal(el.props["data-open-file"], PDF_PATH);
  let prevented = false;
  el.props.onClick({ preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.deepEqual(opened, [PDF_PATH]);
});

test("sanitized empty hrefs and ordinary web links do not navigate the whole page", () => {
  const empty = MarkdownLink({ href: "", children: "unsafe" });
  assert.equal(empty.type, "a");
  assert.equal(empty.props.href, undefined);
  assert.ok(empty.props["aria-disabled"]);
  assert.equal(empty.props.onClick, undefined);

  const web = MarkdownLink({ href: "https://example.com/a", children: "web" });
  assert.equal(web.type, "a");
  assert.equal(web.props.href, "https://example.com/a");
  assert.equal(web.props.onClick, undefined);
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

