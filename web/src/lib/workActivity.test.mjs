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
const A = await vite.ssrLoadModule("/src/lib/workActivity.ts");
// Same shape as i18n's chat strings (i18n.ts itself needs a browser store).
const ZH = {
  actDone: { thinking: "已完成思考", read: "已读取文件", search: "已搜索代码", commands: "执行了命令", webSearch: "已搜索网页", write: "已写入文件" },
  actJoin2: (a, b) => `${a}并${b.replace(/^已/, "")}`,
  actJoinMore: (labels, more) => labels.join("，") + (more ? "等" : ""),
};
const EN = {
  actDone: { read: "Read files", commands: "Ran commands" },
  actJoin2: (a, b) => `${a} and ${b.charAt(0).toLowerCase()}${b.slice(1)}`,
  actJoinMore: (l, more) => l.join(", ") + (more ? ", etc." : ""),
};
const getStrings = (lang) => ({ chat: lang === "zh" ? ZH : EN });
after(() => vite.close());

const tool = (name, extra = {}) => ({ kind: "tool", name, argsStr: "{}", result: "ok", ...extra });

test("tools map to categories; unknown tools are generic", () => {
  assert.equal(A.toolCategory("execute"), "commands");
  assert.equal(A.toolCategory("apply_patch"), "edit");
  assert.equal(A.toolCategory("task"), "subagents");
  assert.equal(A.toolCategory("mcp_foo"), "tools");
});

test("done title names the most used categories, top three, no numbers", () => {
  const S = getStrings("zh").chat;
  const items = [tool("read_file"), tool("execute"), tool("execute"), tool("grep")];
  assert.deepEqual(A.groupCategories(items), ["commands", "read", "search"]);
  assert.equal(A.doneTitle(items, S), "执行了命令，已读取文件，已搜索代码");
  assert.equal(A.doneTitle([tool("execute"), tool("read_file")], S), "执行了命令并读取文件");
  assert.equal(A.doneTitle([...items, tool("web_search"), tool("write_file")], S), "执行了命令，已读取文件，已搜索代码等");
  assert.equal(A.doneTitle([{ kind: "think", text: "x" }], S), "已完成思考");
  const E = getStrings("en").chat;
  assert.equal(A.doneTitle([tool("execute"), tool("read_file")], E), "Ran commands and read files");
});

test("running tool is the newest unfinished one", () => {
  const a = tool("read_file");
  const b = tool("execute", { result: undefined });
  assert.equal(A.runningTool([a, b]), b);
  assert.equal(A.runningTool([a]), null);
});

test("diff stats skip file headers", () => {
  assert.deepEqual(A.diffStats("--- a/x\n+++ b/x\n@@\n-old\n+new\n+more\n ctx"), { added: 2, removed: 1 });
  assert.equal(A.diffStats(""), null);
});

test("task result JSON parses; anything else is null", () => {
  const r = A.parseTaskResult(JSON.stringify({
    success: false, status: "max_turns", error: "stopped", subagent_name: "Code Agent", subagent_type: "code",
    tool_count: 2, tool_calls_summary: [{ name: "glob", info: "*.cs" }], execution_time: 1.25, result: "partial",
  }));
  assert.equal(r.status, "max_turns");
  assert.equal(r.subagentName, "Code Agent");
  assert.equal(r.executionMs, 1250);
  assert.deepEqual(r.steps, [{ name: "glob", info: "*.cs" }]);
  assert.equal(A.parseTaskResult("Error: boom"), null);
  assert.equal(A.parseTaskResult('{"a":1}'), null);
});

test("latest todos come from the newest successful write_todos", () => {
  const todos = (list) => ({ kind: "tool", name: "write_todos", argsStr: JSON.stringify({ todos: list }), result: "ok" });
  const msgs = [
    { parts: [todos([{ content: "a", status: "pending" }])] },
    { parts: [todos([{ content: "a", status: "completed" }, { content: "b", status: "in_progress" }]),
              { ...todos([{ content: "bad" }]), result: "Error: x" }] },
  ];
  const got = A.latestTodos(msgs);
  assert.deepEqual(got.map((t) => t.status), ["completed", "in_progress"]);
  assert.deepEqual(A.todoProgress(got), { done: 1, total: 2, active: got[1] });
  assert.deepEqual(A.latestTodos([{ parts: [] }]), []);
});

test("an unfinished plan stays with its turn, not with later questions", () => {
  const todos = (list) => ({ kind: "tool", name: "write_todos", argsStr: JSON.stringify({ todos: list }), result: "ok" });
  const plan = [{ content: "a", status: "in_progress" }, { content: "b", status: "pending" }];
  const interrupted = [
    { role: "user", content: "q1" },
    { role: "assistant", parts: [todos(plan)] },
  ];
  assert.equal(A.latestTodos(interrupted).length, 2);
  assert.deepEqual(A.latestTodos([...interrupted, { role: "user", content: "q2" }, { role: "assistant", parts: [] }]), []);
  // a steer chip or an automated report is not a new question: the plan stays
  assert.equal(A.latestTodos([...interrupted, { role: "user", content: "hurry", steer: true }]).length, 2);
  assert.equal(A.latestTodos([...interrupted, { role: "user", content: "[Background]", automated: true }]).length, 2);
  // a new plan written in the new turn shows
  const next = [...interrupted, { role: "user", content: "q2" }, { role: "assistant", parts: [todos([{ content: "c", status: "pending" }])] }];
  assert.deepEqual(A.latestTodos(next).map((t) => t.content), ["c"]);
});

test("the dock shows the plan only while the turn runs; a finished turn leaves it", () => {
  const todos = (list) => ({ kind: "tool", name: "write_todos", argsStr: JSON.stringify({ todos: list }), result: "ok" });
  // The model finished the work but never wrote the last step as completed.
  const msgs = [
    { role: "user", content: "do it" },
    { role: "assistant", parts: [todos([{ content: "a", status: "completed" }, { content: "b", status: "in_progress" }])] },
  ];
  assert.deepEqual(A.dockTodos(msgs, true).map((t) => t.status), ["completed", "in_progress"]);
  assert.deepEqual(A.dockTodos(msgs, false), []);
});

test("toolResultPreview keeps execute output for inline arrow display and copy/paste", () => {
  const out = "d727b86 benchmarks: adapt runner\n1 file changed, 12 insertions(+)\n[Exit code: 0]";
  assert.equal(A.toolResultPreview("execute", out), "d727b86 benchmarks: adapt runner 1 file changed, 12 insertions(+)");
  assert.equal(A.toolResultPreview("execute", "(no output)\n[Exit code: 0]"), "");
  assert.equal(A.toolResultPreview("execute", "\n[Exit code: 0]"), "");
  assert.deepEqual(A.parseShellOutput("boom\n[Exit code: 2]"), { output: "boom", exitCode: 2 });
  assert.equal(A.toolResultPreview("read_file", "line 1\nline 2"), "");
  assert.equal(A.toolResultPreview("apply_patch", "patched", true), "");
});

test("parseTaskResult extracts model and modelTier for trace and chat cards", () => {
  const parsed = A.parseTaskResult(JSON.stringify({
    success: true,
    subagent_type: "code",
    subagent_name: "Code Agent",
    model: "openai/claude-opus-5-5",
    model_tier: "main",
    result: "done",
    execution_time: 3.4,
    tool_count: 2,
  }));
  assert.equal(parsed?.model, "openai/claude-opus-5-5");
  assert.equal(parsed?.modelTier, "main");
});
