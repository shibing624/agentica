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

