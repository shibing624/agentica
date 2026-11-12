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
const { eventsToChatMsgs, includeShadowedQuestion } = await vite.ssrLoadModule("/src/lib/hydrateMessages.ts");
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

test("a stopped or failed turn shows its partial answer, not the model-facing marker", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "q1", timestamp: "2026-08-28T01:00:00.000Z" },
    {
      type: "assistant",
      content: "half\n\n[User interrupted the response]",
      partial_content: "half",
      finish_reason: "cancelled",
      timestamp: "2026-08-28T01:00:02.000Z",
    },
    { type: "user", content: "q2", timestamp: "2026-08-28T01:01:00.000Z" },
    {
      type: "assistant",
      content: "[Run failed: boom]",
      partial_content: "",
      finish_reason: "error",
      error: "boom",
      timestamp: "2026-08-28T01:01:02.000Z",
    },
  ]);
  assert.equal(msgs.length, 4);
  assert.equal(msgs[1].content, "half");
  assert.equal(msgs[1].aborted, true);
  assert.equal(msgs[3].content, "");
  assert.equal(msgs[3].error, "boom");
  assert.equal(msgs[3].aborted, undefined);
});

test("a turn sealed after a crash shows as stopped, not the seal marker", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "q1", timestamp: "2026-08-28T01:00:00.000Z" },
    {
      type: "assistant",
      content: "[Session ended before the assistant replied]",
      partial_content: "",
      finish_reason: "interrupted",
      timestamp: "2026-08-28T01:00:02.000Z",
    },
    { type: "user", content: "q2", timestamp: "2026-08-28T01:01:00.000Z" },
    {
      // Rows sealed before ``partial_content`` existed.
      type: "assistant",
      content: "[Session ended before the assistant replied]",
      finish_reason: "interrupted",
      timestamp: "2026-08-28T01:01:02.000Z",
    },
  ]);
  assert.equal(msgs.length, 4);
  assert.equal(msgs[1].content, "");
  assert.equal(msgs[1].aborted, true);
  assert.equal(msgs[3].content, "");
  assert.equal(msgs[3].aborted, true);
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
  assert.equal(msgs[1].durationMs, 2000);
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

test("auto compact as a turn starts shows the question once, after the card", () => {
  // The question is logged, the window rolls over, and the new window re-logs
  // it behind the preamble: the copy the turn actually runs on.
  const msgs = eventsToChatMsgs([
    { type: "user", content: "old" },
    { type: "assistant", content: "before" },
    { type: "user", content: "batch1 怎么算？" },
    { type: "compact_boundary", trigger: "auto" },
    { type: "user", content: PREAMBLE + "\n\nbatch1 怎么算？" },
    { type: "assistant", content: "这样算" },
  ]);
  assert.deepEqual(view(msgs), ["user:old", "assistant:before", "card:auto", "user:batch1 怎么算？", "assistant:这样算"]);
});

test("a page cut keeps the unanswered question on the boundary's page", () => {
  const entries = [
    { type: "user", content: "old" },
    { type: "assistant", content: "before" },
    { type: "user", content: "batch1 怎么算？" },
    { type: "compact_boundary", trigger: "auto" },
    { type: "user", content: PREAMBLE + "\n\nbatch1 怎么算？" },
    { type: "assistant", content: "这样算" },
  ];
  assert.equal(includeShadowedQuestion(entries, 3), 2);
  const answered = [
    { type: "user", content: "q" },
    { type: "assistant", content: "a" },
    { type: "compact_boundary", trigger: "auto" },
    { type: "user", content: PREAMBLE + "\n\nq" },
  ];
  assert.equal(includeShadowedQuestion(answered, 2), 2);
});

test("a question asked again after it was answered is not a shadow", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "q" },
    { type: "assistant", content: "a" },
    { type: "compact_boundary", trigger: "auto" },
    { type: "user", content: PREAMBLE + "\n\nq" },
  ]);
  assert.deepEqual(view(msgs), ["user:q", "assistant:a", "card:auto", "user:q"]);
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

test("a scheduled run keeps the automation that sent it", () => {
  const msgs = eventsToChatMsgs([
    {
      type: "user", automated: true, cron_job: { id: "j1", name: "Daily digest" },
      content: "[Scheduled task “Daily digest” (j1)]\nsummarize", timestamp: "2026-11-06T01:00:00.000Z",
    },
    { type: "assistant", content: "here it is", timestamp: "2026-11-06T01:00:05.000Z" },
    { type: "user", automated: true, content: "[Background terminal #1]", timestamp: "2026-11-06T01:10:00.000Z" },
  ]);
  assert.deepEqual(msgs[0].cronJob, { id: "j1", name: "Daily digest" });
  assert.equal(msgs[0].automated, true);
  // Other automated turns stay plain notices.
  assert.equal(msgs[2].cronJob, undefined);
});

test("prepending turn-aligned pages matches hydrating the full transcript at once", () => {
  const olderPage = [
    { type: "user", content: "q1", uuid: "u1" },
    { type: "assistant", content: "a1", uuid: "a1" },
    { type: "user", content: "q2", uuid: "u2" },
    { type: "assistant", content: "a2", uuid: "a2" },
  ];
  const tailPage = [
    { type: "compact_boundary", trigger: "auto", summary: "" },
    {
      type: "user",
      content:
        '<context_window>\n<session_notes path="/tmp/n.md">notes</session_notes>\n' +
        "New context window started. Continue from session notes and search_session.\n</context_window>\n\nq3",
      uuid: "u3",
    },
    { type: "assistant", content: "a3", uuid: "a3" },
    { type: "user", content: "q4", uuid: "u4" },
    { type: "assistant", content: "a4", uuid: "a4" },
  ];
  const paged = [...eventsToChatMsgs(olderPage), ...eventsToChatMsgs(tailPage)];
  const whole = eventsToChatMsgs([...olderPage, ...tailPage]);
  assert.deepEqual(
    paged.map((m) => [m.role, m.content, m.uuid, m.compact]),
    whole.map((m) => [m.role, m.content, m.uuid, m.compact]),
  );
});

test("mid-turn user steer rows and folded tool steers hydrate as steer parts inside the assistant turn", () => {
  const msgs = eventsToChatMsgs([
    { type: "user", content: "fix the bug", timestamp: "2026-11-08T04:16:00.000Z" },
    {
      type: "assistant",
      content: "",
      timestamp: "2026-11-08T04:16:05.000Z",
      tool_calls: [{ id: "tc1", type: "function", function: { name: "grep", arguments: '{"pattern":"steer"}' } }],
    },
    {
      type: "tool",
      tool_call_id: "tc1",
      tool_name: "grep",
      content: "web/src/api.ts:280",
      timestamp: "2026-11-08T04:16:06.000Z",
    },
    {
      type: "user",
      steer: true,
      content: "优化展示效果",
      timestamp: "2026-11-08T04:16:07.000Z",
    },
    {
      type: "assistant",
      content: "",
      timestamp: "2026-11-08T04:16:10.000Z",
      tool_calls: [{ id: "tc2", type: "function", function: { name: "read_file", arguments: '{"file_path":"a.ts"}' } }],
    },
    {
      type: "tool",
      tool_call_id: "tc2",
      tool_name: "read_file",
      content: "line 1\n\n[User guidance received while you were working]\n前面输入的效果是ok的。",
      timestamp: "2026-11-08T04:16:12.000Z",
    },
    { type: "assistant", content: "all fixed", timestamp: "2026-11-08T04:16:15.000Z" },
  ]);
  assert.equal(msgs.length, 2);
  assert.equal(msgs[0].role, "user");
  assert.equal(msgs[0].content, "fix the bug");
  assert.equal(msgs[1].role, "assistant");
  assert.equal(msgs[1].content, "all fixed");
  assert.deepEqual(
    parts(msgs[1]).map((p) => [p.kind, p.kind === "tool" ? p.name : p.text]),
    [
      ["tool", "grep"],
      ["steer", "优化展示效果"],
      ["tool", "read_file"],
      ["steer", "前面输入的效果是ok的。"],
      ["text", "all fixed"],
    ],
  );
  // Folded steer text is stripped from the tool result itself
  const tc2 = parts(msgs[1]).find((p) => p.kind === "tool" && p.toolCallId === "tc2");
  assert.equal(tc2.result, "line 1");
});

test("window preamble mentioning __RELAYED__ or closing tags in notes/dropped_span stays hidden inside compact card", () => {
  const preambleWithMarkers = [
    "<context_window>\nCurrent context window 1.\nYou have 512000 tokens left in this context window.\n</context_window>",
    '<session_notes path="/tmp/s.notes.md">\n- Parsed `steer: true`, `__RELAYED__`, and `</session_notes>` in `[User guidance received while you were working]`\n</session_notes>',
    '<dropped_span>\nconst WINDOW_CLOSERS = ["</context_window>", "</session_notes>", "</dropped_span>"];\nconst WINDOW_CONTINUATION = "New context window started. Continue from session notes and search_session.";\n</dropped_span>',
    "New context window started. Continue from session notes and search_session.",
  ].join("\n\n");

  const msgs = eventsToChatMsgs([
    { type: "user", content: "first question" },
    { type: "assistant", content: "first answer" },
    { type: "compact_boundary", trigger: "manual", timestamp: "2026-11-08T05:00:00.000Z" },
    { type: "user", content: preambleWithMarkers, timestamp: "2026-11-08T05:00:01.000Z" },
    { type: "user", content: "next question after compact", timestamp: "2026-11-08T05:01:00.000Z" },
    { type: "assistant", content: "next answer", timestamp: "2026-11-08T05:01:05.000Z" },
  ]);

  assert.deepEqual(view(msgs), [
    "user:first question",
    "assistant:first answer",
    "user:/compact",
    "card",
    "user:next question after compact",
    "assistant:next answer",
  ]);
  // No steer chip or user bubble may leak `<context_window>`
  for (const m of msgs) {
    assert.equal(String(m.content || "").includes("<context_window>"), false);
    for (const p of parts(m)) {
      assert.equal(String(p.text || "").includes("<context_window>"), false);
    }
  }
  assert.match(msgs[3].compact.notes, /__RELAYED__/);
  assert.equal(msgs[3].compact.notesPath, "/tmp/s.notes.md");
});

test("compact card takes notes_path from the boundary row when the preamble carries no <session_notes>", () => {
  const bare = "<context_window>\nCurrent context window 2.\nYou have 1000 tokens left in this context window.\n</context_window>";
  const msgs = eventsToChatMsgs([
    { type: "user", content: "hi" },
    { type: "compact_boundary", trigger: "manual", notes_path: "~/.agentica/projects/p/s1.notes.md" },
    { type: "user", content: bare },
  ]);
  const card = msgs.find((m) => m.compact);
  assert.equal(card.compact.notesPath, "~/.agentica/projects/p/s1.notes.md");
  assert.equal(card.compact.notes, "");
});

test("auto-rollover question logged with the folded remaining-tokens reminder shows only the question", () => {
  const preamble = "<context_window>\nCurrent context window 1.\nYou have 512000 tokens left in this context window.\n</context_window>";
  const question = "换个模型后输入框多了东西，fix";
  const msgs = eventsToChatMsgs([
    { type: "user", content: "earlier" },
    { type: "assistant", content: "earlier answer" },
    { type: "compact_boundary", trigger: "auto" },
    {
      type: "user",
      content: `${preamble}\n\n${question}\n\n<context_window>\nYou have 13572 tokens left in this context window.\n</context_window>`,
    },
    { type: "assistant", content: "fixed" },
  ]);
  const users = msgs.filter((m) => m.role === "user" && !m.compact).map((m) => m.content);
  assert.equal(users.at(-1), question);
});
