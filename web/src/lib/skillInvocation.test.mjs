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
const { parseSkillTurn } = await vite.ssrLoadModule("/src/lib/skillInvocation.ts");
const { fileEditsOf, patchEdits } = await vite.ssrLoadModule("/src/lib/fileEdits.ts");
after(() => vite.close());

// Same text Skill.render_invocation() produces (agentica/skills/skill.py).
const BODY = "# Review\n\nSteps:\n- read the diff\n\nArgument: this line is part of the skill body";
const expanded = (args) => {
  const head = `Loading: code-review\nBase directory: /Users/me/.agentica/skills/code-review\n\n${BODY}`;
  if (!args) return head;
  return [head, "", "---", "", "## Skill invocation", "",
    "The user invoked this skill as: `/review " + args + "`", "", `Argument: ${args}`,
    "Expected argument: a path", "",
    "The argument names the target or scope this skill applies to. Read it as input."].join("\n");
};

test("the expanded prompt in the log becomes a chip with its argument", () => {
  assert.deepEqual(parseSkillTurn(expanded("src/app.py 的改动")), { name: "code-review", args: "src/app.py 的改动" });
});

test("no argument, and a body line that looks like one, is not taken as the argument", () => {
  // No trailer, but the header names the skill's own directory: replay (which
  // has no skills list) still recognises it.
  assert.deepEqual(parseSkillTurn(expanded("")), { name: "code-review", args: "" });
  assert.deepEqual(
    parseSkillTurn(expanded(""), [{ cmd: "/review", name: "code-review" }]),
    { name: "code-review", args: "" },
  );
});

test("a multi-line argument is kept whole", () => {
  assert.equal(parseSkillTurn(expanded("line one\nline two")).args, "line one\nline two");
});

test("a pasted document with a skill-like header is not hidden behind a chip", () => {
  const pasted = [
    "Loading: whatever",
    "Base directory: /tmp/skills/notes",
    "",
    "这是一段我粘贴的普通内容，讨论技能格式。",
    "后面还有很长的正文，不能被吞掉。",
  ].join("\n");
  assert.equal(parseSkillTurn(pasted, [{ cmd: "/review", name: "code-review" }]), null);
});

test("an unterminated invocation does not swallow the remaining body", () => {
  const text = [
    "Loading: code-review",
    "Base directory: /Users/me/.agentica/skills/code-review",
    "",
    "---",
    "",
    "## Skill invocation",
    "",
    "Argument: src/app.py",
    "",
    "这里是很长的一段后续正文，原本没有终止标记",
  ].join("\n");
  assert.deepEqual(parseSkillTurn(text), { name: "code-review", args: "" });
});

test("the typed /trigger form maps through the known skills only", () => {
  const skills = [{ cmd: "/review", name: "code-review" }];
  assert.deepEqual(parseSkillTurn("/review  fix the bug", skills), { name: "code-review", args: "fix the bug" });
  assert.deepEqual(parseSkillTurn("/Review", skills), { name: "code-review", args: "" });
  assert.equal(parseSkillTurn("/goal ship it", skills), null);
  assert.equal(parseSkillTurn("please /review this", skills), null);
  assert.equal(parseSkillTurn("an ordinary question"), null);
});

test("apply_patch counts +/- per file, including add, delete and move", () => {
  const patch = [
    "*** Begin Patch",
    "*** Update File: app.py",
    "@@",
    " def run():",
    "-    timeout = 10",
    "+    timeout = 30",
    "+    retries = 3",
    "*** Add File: docs/new.md",
    "+# New",
    "+text",
    "*** Delete File: old.txt",
    "*** Update File: a.py",
    "*** Move to: b.py",
    "@@",
    "-x = 1",
    "*** End Patch",
  ].join("\n");
  assert.deepEqual(patchEdits(patch), [
    { path: "app.py", added: 2, removed: 1 },
    { path: "docs/new.md", added: 2, removed: 0 },
    { path: "old.txt", added: 0, removed: 0, deleted: true },
    { path: "b.py", added: 0, removed: 1 },
  ]);
});

test("steps: write_file counts new lines, failed calls are skipped, one row per file", () => {
  const steps = [
    { type: "tool", name: "write_file", argsStr: JSON.stringify({ file_path: "README.md", content: "a\nb\n" }), result: "Wrote" },
    { type: "tool", name: "apply_patch", argsStr: JSON.stringify({ patch: "*** Update File: README.md\n@@\n-a\n+c" }), result: "ok" },
    { type: "tool", name: "apply_patch", argsStr: JSON.stringify({ patch: "*** Update File: x.py\n@@\n+y" }), result: "Error: context not found" },
    { type: "tool", name: "read_file", argsStr: JSON.stringify({ file_path: "z.py" }), result: "..." },
  ];
  assert.deepEqual(fileEditsOf(steps), [{ path: "README.md", added: 3, removed: 1 }]);
});

