import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const {
  buildChatNavTicks, previewText, CHAT_NAV_MIN_TICKS, PROMPT_PREVIEW_LIMIT, ANSWER_PREVIEW_LIMIT,
} = await vite.ssrLoadModule("/src/lib/chatNav.ts");
after(() => vite.close());

test("each tick carries the truncated query and skips steer chips", () => {
  const ticks = buildChatNavTicks([
    { role: "user", content: "first question" },
    { role: "assistant", content: "first answer" },
    { role: "user", content: "interrupt", steer: true },
    { role: "user", content: "second question" },
    { role: "assistant", parts: [{ kind: "think", text: "hmm" }, { kind: "text", text: "second answer" }] },
  ]);
  assert.equal(CHAT_NAV_MIN_TICKS, 2);
  assert.equal(ticks.length, 2);
  assert.deepEqual(ticks.map((t) => t.idx), [0, 3]);
  assert.equal(ticks[0].query, "first question");
  assert.equal(ticks[1].query, "second question");
});

test("preview collapses whitespace and truncates to 50 by default", () => {
  assert.equal(previewText("  hello   world  ", 8), "hello w…");
  const long = "一二三四五六七八九十".repeat(6);
  assert.equal(long.length, 60);
  assert.equal(previewText(long).length, 50);
  // A clipped preview says so, so the popover never reads as the whole turn.
  assert.equal(previewText(long), long.slice(0, 49) + "…");
  assert.equal(buildChatNavTicks([{ role: "user", content: long }])[0].query, long.slice(0, 49) + "…");
  assert.equal(previewText("short"), "short");
});

test("each tick carries the turn's own final answer, bounded", () => {
  const ticks = buildChatNavTicks([
    { role: "user", content: "q1" },
    { role: "assistant", parts: [{ kind: "think", text: "hmm" }, { kind: "text", text: "a1" }] },
    { role: "user", content: "q2" },
    { role: "assistant", content: "a2" },
    { role: "user", content: "q3" },
  ]);
  assert.deepEqual(ticks.map((t) => t.answer), ["a1", "a2", ""]);
  // The answer belongs to its own turn, never to the next one's.
  assert.equal(ticks[0].query, "q1");

  const long = "答".repeat(200);
  const [only] = buildChatNavTicks([
    { role: "user", content: "q" },
    { role: "assistant", content: long },
  ]);
  assert.equal(ANSWER_PREVIEW_LIMIT, 120);
  assert.equal(only.answer.length, 120);
  assert.equal(only.answer, "答".repeat(119) + "…");
  assert.equal(PROMPT_PREVIEW_LIMIT, 50);
});

test("a turn that ended with tool output still shows the reply before it", () => {
  // The reply came mid-turn and the turn ended on a tool call: `content` still
  // holds what was said, so the rail does not fall silent on that turn.
  const ticks = buildChatNavTicks([
    { role: "user", content: "q1" },
    {
      role: "assistant",
      content: "第一个回答",
      parts: [
        { kind: "text", text: "第一个回答" },
        { kind: "tool", name: "execute", argsStr: "{}", result: "ok" },
      ],
    },
    { role: "user", content: "q2" },
  ]);
  assert.equal(ticks[0].answer, "第一个回答");
});

test("steer chips and automated reports never become a tick's answer", () => {
  const ticks = buildChatNavTicks([
    { role: "user", content: "q1" },
    { role: "user", content: "interrupt", steer: true },
    { role: "assistant", content: "answer one" },
    { role: "user", content: "[Background terminal #1 finished]", automated: true },
    { role: "assistant", content: "report answer" },
  ]);
  assert.equal(ticks.length, 1);
  // The report's answer still belongs to the turn it answered (turn 1),
  // because the report itself is not a question.
  assert.equal(ticks[0].answer, "report answer");
});

test("a preview drops Markdown that only reads well when rendered", () => {
  // A table's pipe grid and rule are unreadable at 50 characters; only the
  // words survive, so the popover does not show `| 你要的效果 | 之前实际状态 |`.
  const md = [
    "核对完成。逐项结论：",
    "",
    "| 你要的效果 | 之前 | 现在 |",
    "|---|---|---|",
    "| 分类标题 | 有 | 不变 |",
    "",
    "1. 第一项",
    "2. 第二项",
  ].join("\n");
  assert.equal(previewText(md, 120), "核对完成。逐项结论： 1. 第一项 2. 第二项");

  // Inline markers, links and headings reduce to their words.
  assert.equal(previewText("## 标题\n\n**粗** 与 `code` 与 [链接](https://x.y)"), "标题 粗 与 code 与 链接");
  // A numbered heading in a real reply: the marker is per line, so it must not
  // survive as `## 1.` once the lines are joined.
  assert.equal(previewText("## 1. 无法插话\n原因如下"), "1. 无法插话 原因如下");
  assert.equal(previewText("![截图](/tmp/a.png) 说明"), "截图 说明");
  // A fenced block is code, not prose; it never enters a preview.
  assert.equal(previewText("说明\n```py\nprint(1)\n```\n结束"), "说明 结束");
  assert.equal(previewText("---\n正文"), "正文");
});

test("automated reports get no minimap tick", () => {
  const ticks = buildChatNavTicks([
    { role: "user", content: "real question" },
    { role: "assistant", content: "a" },
    { role: "user", content: "[Background terminal #1 finished]", automated: true },
    { role: "assistant", content: "b" },
    { role: "user", content: "second question" },
  ]);
  assert.deepEqual(ticks.map((t) => t.idx), [0, 4]);
});
