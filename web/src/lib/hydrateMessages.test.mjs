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
