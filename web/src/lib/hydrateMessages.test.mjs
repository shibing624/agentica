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
const { eventsToChatMsgs } = await vite.ssrLoadModule("/src/lib/hydrateMessages.ts");
after(() => vite.close());

function parts(m) {
  return m.parts || [];
}

test("a turn with pictures shows them again and hides the media description", () => {
  const a = "a".repeat(64) + ".png";
  const b = "b".repeat(64) + ".jpg";
  const msgs = eventsToChatMsgs([
    {
      type: "user",
      content: `看下这个\n\n[图片: ${a}]\n\n[图片: ${b}]\n\n[图片内容]\n一张截图\n\n[图片内容]\n另一张`,
      timestamp: "2026-08-28T01:00:00.000Z",
    },
    { type: "assistant", content: "ok", timestamp: "2026-08-28T01:00:02.000Z" },
    { type: "user", content: `[图片: ${a}]\n\n[图片内容]\n只有图片`, timestamp: "2026-08-28T01:01:00.000Z" },
    { type: "user", content: "文中提到 [图片: x] 不是附件", timestamp: "2026-08-28T01:02:00.000Z" },
  ]);
  assert.equal(msgs[0].content, "看下这个");
  assert.deepEqual(msgs[0].previews, [`/api/chat-images/${a}`, `/api/chat-images/${b}`]);
  assert.equal(msgs[2].content, "");
  assert.deepEqual(msgs[2].previews, [`/api/chat-images/${a}`]);
  assert.equal(msgs[3].content, "文中提到 [图片: x] 不是附件");
  assert.equal(msgs[3].previews, undefined);
});

test("plain user/assistant text still hydrates as two bubbles", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "hello", timestamp: "2026-08-28T01:00:00.000Z" },
    { type: "event", name: "request_begin" },
    { type: "event", name: "text" },
    { type: "assistant", content: "hi", timestamp: "2026-08-28T01:00:02.000Z" },
  ]);
  assert.equal(msgs.length, 2);
  assert.equal(msgs[0].role, "user");
  assert.equal(msgs[0].content, "hello");
  assert.equal(msgs[1].role, "assistant");
  assert.equal(msgs[1].content, "hi");
  assert.equal(parts(msgs[1]).some((p) => p.kind === "tool"), false);
});

test("one user turn folds tool_calls + tool rows into the assistant work group", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "where is persist?", timestamp: "2026-08-28T01:00:00.000Z" },
    { type: "event", name: "tool_call", tool_call_id: "c1", tool_name: "grep" },
    {
      type: "assistant",
      content: "",
      timestamp: "2026-08-28T01:00:01.000Z",
      reasoning_content: "search the repo",
      tool_calls: [{
        id: "c1",
        type: "function",
        function: { name: "grep", arguments: '{"pattern":"persist","path":"."}' },
      }],
    },
    {
      type: "tool",
      tool_name: "grep",
      tool_call_id: "c1",
      content: "agentica/runner/persist.py:1",
      timestamp: "2026-08-28T01:00:01.400Z",
    },
    { type: "assistant", content: "It is in persist.py.", timestamp: "2026-08-28T01:00:02.000Z" },
  ]);
  assert.equal(msgs.length, 2);
  assert.equal(msgs[0].content, "where is persist?");
  const kinds = parts(msgs[1]).map((p) => p.kind);
  assert.deepEqual(kinds, ["think", "tool", "text"]);
  const tool = parts(msgs[1]).find((p) => p.kind === "tool");
  assert.equal(tool.name, "grep");
  assert.equal(tool.toolCallId, "c1");
  assert.match(tool.argsStr, /persist/);
  assert.equal(tool.result, "agentica/runner/persist.py:1");
  assert.equal(tool.ms, 400);
  assert.equal(msgs[1].content, "It is in persist.py.");
});

test("parallel tool_calls pair results by tool_call_id", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "read both" },
    {
      type: "assistant",
      content: "",
      tool_calls: [
        { id: "a", type: "function", function: { name: "read_file", arguments: '{"file_path":"a.py"}' } },
        { id: "b", type: "function", function: { name: "read_file", arguments: '{"file_path":"b.py"}' } },
      ],
    },
    { type: "tool", tool_call_id: "b", tool_name: "read_file", content: "bbb" },
    { type: "tool", tool_call_id: "a", tool_name: "read_file", content: "aaa" },
    { type: "assistant", content: "done" },
  ]);
  const tools = parts(msgs[1]).filter((p) => p.kind === "tool");
  assert.equal(tools.length, 2);
  assert.equal(tools[0].toolCallId, "a");
  assert.equal(tools[0].result, "aaa");
  assert.equal(tools[1].toolCallId, "b");
  assert.equal(tools[1].result, "bbb");
});

test("tool errors keep the Error: prefix the live stream uses", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "run it" },
    {
      type: "assistant",
      content: "",
      tool_calls: [{ id: "x", type: "function", function: { name: "execute", arguments: '{"command":"false"}' } }],
    },
    { type: "tool", tool_call_id: "x", tool_name: "execute", content: "exit 1", is_error: true },
    { type: "assistant", content: "failed" },
  ]);
  const tool = parts(msgs[1]).find((p) => p.kind === "tool");
  assert.equal(tool.result, "Error: exit 1");
});

test("a write tool row's diff comes back on reload, on the part and the step", () => {
  const diff = "diff -- a.py\n@@ -1 +1 @@\n-x\n+y";
  const msgs = eventsToChatMsgs([
    { type: "user", content: "edit" },
    {
      type: "assistant",
      content: "",
      tool_calls: [{ id: "w", type: "function", function: { name: "apply_patch", arguments: "{}" } }],
    },
    { type: "tool", tool_call_id: "w", tool_name: "apply_patch", content: "ok", diff },
    { type: "assistant", content: "done" },
  ]);
  assert.equal(parts(msgs[1]).find((p) => p.kind === "tool").diff, diff);
  assert.equal(msgs[1].steps.find((s) => s.type === "tool").diff, diff);
});

const PREAMBLE = [
  "<context_window>\nCurrent context window 2.\n</context_window>",
  '<session_notes path="/h/p/s1.notes.md">\n- goal: ship\n</session_notes>',
  "<dropped_span>\n- user: hi\n</dropped_span>",
  "New context window started. Continue from session notes and search_session.",
].join("\n\n");

const view = (msgs) => msgs.map((m) => (m.compact ? "card" + (m.compact.auto ? ":auto" : "") : m.role + ":" + m.content));

test("manual /compact replays as the user's command, then the card with notes", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "hi" },
    { type: "assistant", content: "hello" },
    { type: "compact_boundary", trigger: "manual", timestamp: "2026-11-05T05:25:42.691Z" },
    { type: "user", content: PREAMBLE },
    { type: "user", content: "next" },
  ]);
  assert.deepEqual(view(msgs), ["user:hi", "assistant:hello", "user:/compact", "card", "user:next"]);
  assert.deepEqual(msgs[3].compact, { notes: "- goal: ship", notesPath: "/h/p/s1.notes.md" });
  assert.equal(msgs[2].ts, Date.parse("2026-11-05T05:25:42.691Z"));
});

test("auto compact keeps the pending question and shows no /compact", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "old" },
    { type: "assistant", content: "before" },
    { type: "compact_boundary", trigger: "auto" },
    { type: "user", content: PREAMBLE + "\n\n" + PREAMBLE + "\n\nwhat next?" },
    { type: "assistant", content: "this" },
  ]);
  assert.deepEqual(view(msgs), ["user:old", "assistant:before", "card:auto", "user:what next?", "assistant:this"]);
  assert.equal(msgs[2].compact.notes, "- goal: ship");
});

test("logs without trigger: bare preamble means /compact, a folded question means auto", () => {
  const idle = eventsToChatMsgs([
    { type: "user", content: "hi" },
    { type: "assistant", content: "hello" },
    { type: "compact_boundary" },
    { type: "user", content: PREAMBLE },
  ]);
  assert.deepEqual(view(idle), ["user:hi", "assistant:hello", "user:/compact", "card"]);
  const auto = eventsToChatMsgs([
    { type: "compact_boundary" },
    { type: "user", content: PREAMBLE + "\n\nq" },
  ]);
  assert.deepEqual(view(auto), ["card:auto", "user:q"]);
});

test("legacy boundary with no preamble after it is still a card between the windows", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "old" },
    { type: "assistant", content: "before compact" },
    { type: "compact_boundary", summary: "..." },
    { type: "user", content: "new" },
    { type: "assistant", content: "after compact" },
  ]);
  assert.deepEqual(view(msgs), ["user:old", "assistant:before compact", "card", "user:new", "assistant:after compact"]);
});

test("two user turns stay two assistant bubbles", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "q1" },
    {
      type: "assistant",
      content: "",
      tool_calls: [{ id: "t1", function: { name: "glob", arguments: '{"pattern":"*.py"}' } }],
    },
    { type: "tool", tool_call_id: "t1", tool_name: "glob", content: "a.py" },
    { type: "assistant", content: "a.py" },
    { type: "user", content: "q2" },
    { type: "assistant", content: "ok" },
  ]);
  assert.equal(msgs.map((m) => m.role).join(","), "user,assistant,user,assistant");
  assert.equal(parts(msgs[1]).filter((p) => p.kind === "tool").length, 1);
  assert.equal(parts(msgs[3]).some((p) => p.kind === "tool"), false);
});

test("an automated row is a notice, not the user's query", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "fix the bug", timestamp: "2026-11-06T01:00:00.000Z" },
    { type: "assistant", content: "started it", timestamp: "2026-11-06T01:00:02.000Z" },
    {
      type: "user", automated: true,
      content: "[Background terminal #4 (term_4) failed: exit -15 after 35s]\nCommand: x",
      timestamp: "2026-11-06T01:05:00.000Z",
    },
    { type: "assistant", content: "it was me stopping it", timestamp: "2026-11-06T01:05:03.000Z" },
  ]);
  assert.deepEqual(msgs.map((m) => [m.role, !!m.automated]), [
    ["user", false], ["assistant", false], ["user", true], ["assistant", false],
  ]);
  assert.match(msgs[2].content, /^\[Background terminal #4/);
});
