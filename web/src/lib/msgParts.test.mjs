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
  allAssistantTextOf,
  appendSteerPart,
  appendText,
  appendThink,
  appendTool,
  ensureTrailingFinalAnswer,
  finalAnswerOf,
  finishThink,
  finishTool,
  groupParts,
  isProcessSegment,
  partsOf,
  settleWork,
  workSummary,
} = await vite.ssrLoadModule("/src/lib/msgParts.ts");
const { eventsToChatMsgs } = await vite.ssrLoadModule("/src/lib/hydrateMessages.ts");
after(() => vite.close());

function msg(parts) {
  return { role: "assistant", content: "", parts, steps: [] };
}

test("finishThink freezes every open think, not only the last", () => {
  const m = msg([
    { kind: "think", text: "a", t0: 1000 },
    { kind: "tool", name: "glob", argsStr: "{}", t0: 1300, ms: 40, result: "ok" },
    { kind: "think", text: "b", t0: 1400 },
  ]);
  finishThink(m, 2000);
  assert.equal(m.parts[0].ms, 300);
  assert.equal(m.parts[2].ms, 600);
  assert.equal(workSummary(m.parts, 9_000_000).running, false);
});

test("earlier think duration is next tool t0, not wall clock at settle", () => {
  const m = msg([
    { kind: "think", text: "a", t0: 1000 },
    { kind: "tool", name: "glob", argsStr: "{}", t0: 1300, ms: 20, result: "ok" },
    { kind: "think", text: "b", t0: 1500 },
    { kind: "tool", name: "execute", argsStr: "{}", t0: 1800, ms: 50, result: "ok" },
  ]);
  const changed = settleWork(m);
  assert.equal(changed, true);
  assert.equal(m.parts[0].ms, 300);
  assert.equal(m.parts[2].ms, 300);
  const sum = workSummary(m.parts, Date.now());
  assert.equal(sum.running, false);
  assert.equal(sum.ms, 300 + 20 + 300 + 50);
});

test("settling a stale card does not use Date.now() for a trailing think", () => {
  const t0 = Date.now() - 23 * 3600_000;
  const m = msg([
    { kind: "think", text: "left open", t0 },
    { kind: "text", text: "done" },
  ]);
  settleWork(m);
  assert.equal(m.parts[0].ms, 0);
  assert.equal(workSummary(m.parts).running, false);
});

test("workSummary is not running when the message is no longer live", () => {
  const items = [
    { kind: "think", text: "a", t0: 1 },
    { kind: "tool", name: "glob", argsStr: "{}", t0: 2, result: "ok", ms: 10 },
  ];
  assert.equal(workSummary(items, 50_000, false).running, false);
  assert.equal(workSummary(items, 50_000, true).running, true);
});

test("appendThink after a frozen slice opens a new think", () => {
  const m = msg([{ kind: "think", text: "a", t0: 1000, ms: 200 }]);
  m.steps = [{ type: "thinking", text: "a", t0: 1000, ms: 200 }];
  appendThink(m, " more");
  assert.equal(m.parts.length, 2);
  assert.equal(m.parts[0].text, "a");
  assert.equal(m.parts[1].kind, "think");
  assert.equal(m.parts[1].text, " more");
  assert.equal(m.parts[1].ms, undefined);
});

test("settleWork drops the empty placeholder think of a turn stopped before any output", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendThink(m, "");
  assert.equal(m.parts.length, 1);
  assert.equal(settleWork(m), true);
  assert.deepEqual(m.parts, []);
  assert.deepEqual(m.steps, []);
});

test("settleWork keeps a think that has text", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendThink(m, "planning the change");
  settleWork(m);
  assert.equal(m.parts.length, 1);
  assert.equal(m.parts[0].text, "planning the change");
  assert.equal(m.steps.length, 1);
});

test("CoT-hiding models: intermediate commentary between tool calls folds with turn process", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendThink(m, "");
  appendText(m, "我先查看 ChatPage.tsx 的渲染结构。");
  appendTool(m, "read_file", JSON.stringify({ file_path: "web/src/pages/ChatPage.tsx" }), "c1");
  finishTool(m, "", undefined, "c1");
  appendText(m, "接着运行前端单测确认现状。");
  appendTool(m, "execute", JSON.stringify({ command: "npm test", description: "运行前端单元测试" }), "c2");
  finishTool(m, "ok\n[Exit code: 0]", undefined, "c2");
  appendText(m, "已完成全部优化：\n1. 流式平铺命令简述\n2. 折叠隐藏过渡说明");
  settleWork(m, Date.now());

  const segs = groupParts(partsOf(m));
  assert.equal(segs.length, 5);
  assert.deepEqual(
    segs.map((_, i) => isProcessSegment(segs, i)),
    [true, true, true, true, false],
  );
  assert.equal(finalAnswerOf(m), "已完成全部优化：\n1. 流式平铺命令简述\n2. 折叠隐藏过渡说明");
  assert.equal(
    allAssistantTextOf(m),
    "我先查看 ChatPage.tsx 的渲染结构。\n\n接着运行前端单测确认现状。\n\n已完成全部优化：\n1. 流式平铺命令简述\n2. 折叠隐藏过渡说明",
  );
});

test("plain Q&A without work keeps its text outside process folding", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendText(m, "直接给出的最终回答");
  const segs = groupParts(partsOf(m));
  assert.equal(segs.length, 1);
  assert.equal(isProcessSegment(segs, 0), false);
  assert.equal(finalAnswerOf(m), "直接给出的最终回答");
});

test("user steer chip stays outside process folding while surrounding commentary folds", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendText(m, "先读文件");
  appendTool(m, "read_file", JSON.stringify({ file_path: "a.py" }), "c1");
  finishTool(m, "", undefined, "c1");
  appendSteerPart(m, "顺便检查 b.py");
  appendText(m, "收到，一起检查 b.py");
  appendTool(m, "read_file", JSON.stringify({ file_path: "b.py" }), "c2");
  finishTool(m, "", undefined, "c2");
  appendText(m, "两处都检查完毕。");

  const segs = groupParts(partsOf(m));
  // [text, work, steer, text, work, final text]
  assert.deepEqual(
    segs.map((_, i) => isProcessSegment(segs, i)),
    [true, true, false, true, true, false],
  );
  assert.equal(finalAnswerOf(m), "两处都检查完毕。");
});

test("hydrated JSONL turn folds Claude/GPT-5 intermediate content and keeps final answer", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "排查问题", timestamp: "2026-11-11T06:00:00Z" },
    {
      type: "assistant",
      content: "我先看看目录与配置。",
      tool_calls: [{ id: "tc1", function: { name: "execute", arguments: JSON.stringify({ command: "ls", description: "列出项目根目录" }) } }],
      timestamp: "2026-11-11T06:00:02Z",
    },
    { type: "tool", tool_call_id: "tc1", tool_name: "execute", content: "web\nagentica\n[Exit code: 0]", timestamp: "2026-11-11T06:00:03Z" },
    {
      type: "assistant",
      content: "再读取 WorkGroup.tsx。",
      tool_calls: [{ id: "tc2", function: { name: "read_file", arguments: JSON.stringify({ file_path: "web/src/components/WorkGroup.tsx" }) } }],
      timestamp: "2026-11-11T06:00:05Z",
    },
    { type: "tool", tool_call_id: "tc2", tool_name: "read_file", content: "", timestamp: "2026-11-11T06:00:06Z" },
    {
      type: "assistant",
      content: "根因已定位并修复完成。",
      timestamp: "2026-11-11T06:00:10Z",
    },
  ]);
  assert.equal(msgs.length, 2);
  const ai = msgs[1];
  const segs = groupParts(partsOf(ai));
  assert.deepEqual(
    segs.map((_, i) => isProcessSegment(segs, i)),
    [true, true, true, true, false],
  );
  // A reloaded turn has the same shape as a live one: the two spoken lines are
  // their own replies between the step groups, not rows inside them.
  assert.deepEqual(
    segs.map((s) => (s.type === "work" ? "work" : s.part.text)),
    ["我先看看目录与配置。", "work", "再读取 WorkGroup.tsx。", "work", "根因已定位并修复完成。"],
  );
  assert.equal(finalAnswerOf(ai), "根因已定位并修复完成。");
});

test("ensureTrailingFinalAnswer appends goal final_answer even when earlier rounds spoke", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendText(m, "第一轮先跑验证脚本。");
  appendTool(m, "execute", JSON.stringify({ command: "pytest", description: "跑测试" }), "c1");
  finishTool(m, "passed\n[Exit code: 0]", undefined, "c1");
  ensureTrailingFinalAnswer(m, "目标已完成：所有测试均通过。");
  assert.equal(finalAnswerOf(m), "目标已完成：所有测试均通过。");

  // Does not duplicate if trailing answer already exists.
  ensureTrailingFinalAnswer(m, "不会重复追加");
  assert.equal(finalAnswerOf(m), "目标已完成：所有测试均通过。");
});

test("an intermediate reply stands between two work groups, never inside one", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendText(m, "我先查看目录。");
  appendTool(m, "execute", JSON.stringify({ command: "ls", description: "列出目录" }), "c1");
  finishTool(m, "a\n[Exit code: 0]", undefined, "c1");
  appendThink(m, "看完目录，接着读文件。\n\n");
  appendText(m, "再读 README。");
  appendTool(m, "read_file", JSON.stringify({ file_path: "README.md" }), "c2");
  finishTool(m, "", undefined, "c2");
  appendText(m, "最终结论。");
  settleWork(m, Date.now());

  const segs = groupParts(partsOf(m));
  // G1 -> 中间回复 -> G2 -> 最终回答: the reply is its own segment, and the two
  // tool stretches are two groups rather than one merged run.
  assert.deepEqual(segs.map((s) => s.type), ["single", "work", "single", "work", "single"]);
  assert.deepEqual(
    segs.map((_, i) => isProcessSegment(segs, i)),
    [true, true, true, true, false],
  );
  // dsh rule: a reply never joins the steps. Anything text-shaped inside a work
  // group would be hidden behind the group's own fold — two clicks, not one.
  assert.equal(segs.some((s) => s.type === "work" && s.items.some((p) => p.kind === "text")), false);
  assert.equal(segs[2].part.text, "再读 README。");
  assert.equal(segs[4].part.text, "最终结论。");
  assert.equal(finalAnswerOf(m), "最终结论。");
});

test("an intermediate reply is hidden by the turn fold, shown when it opens", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendThink(m, "先看看");
  appendTool(m, "read_file", JSON.stringify({ file_path: "a.py" }), "c1");
  finishTool(m, "", undefined, "c1");
  appendText(m, "中间说明");
  appendTool(m, "read_file", JSON.stringify({ file_path: "b.py" }), "c2");
  finishTool(m, "", undefined, "c2");
  appendText(m, "最终回答");

  const segs = groupParts(partsOf(m));
  const hiddenWhenCollapsed = segs
    .map((_, i) => isProcessSegment(segs, i))
    .filter(Boolean)
    .length;
  // The reply sits inside `已完成，用时 …` with the steps, so collapsing the turn
  // leaves only the answer.
  const replyIndex = segs.findIndex((s) => s.type === "single" && s.part.text === "中间说明");
  assert.equal(isProcessSegment(segs, replyIndex), true);
  assert.equal(hiddenWhenCollapsed, segs.length - 1);
  const finalIndex = segs.findIndex((s) => s.type === "single" && s.part.text === "最终回答");
  assert.equal(isProcessSegment(segs, finalIndex), false);
});

test("a turn with only a reply has no process segments", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendText(m, "直接回答");
  const segs = groupParts(partsOf(m));
  assert.deepEqual(segs.map((s) => s.type), ["single"]);
  assert.equal(isProcessSegment(segs, 0), false);
  assert.equal(finalAnswerOf(m), "直接回答");
});

test("a steer chip is never folded into the turn process", () => {
  const m = { role: "assistant", content: "", parts: [], steps: [], ts: 0 };
  appendTool(m, "read_file", JSON.stringify({ file_path: "a.py" }), "c1");
  finishTool(m, "", undefined, "c1");
  appendSteerPart(m, "顺便看 b.py");
  appendText(m, "好的");
  appendTool(m, "read_file", JSON.stringify({ file_path: "b.py" }), "c2");
  finishTool(m, "", undefined, "c2");
  appendText(m, "完成。");

  const segs = groupParts(partsOf(m));
  assert.deepEqual(
    segs.map((s) => (s.type === "work" ? "work" : s.part.kind)),
    ["work", "steer", "text", "work", "text"],
  );
  assert.deepEqual(
    segs.map((_, i) => isProcessSegment(segs, i)),
    [true, false, true, true, false],
  );
});
