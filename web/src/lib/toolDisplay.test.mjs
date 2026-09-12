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
const { formatToolDisplay, layoutToolDisplay } = await vite.ssrLoadModule("/src/lib/toolDisplay.ts");
after(() => vite.close());

test("execute keeps the complete command in the expanded call body", () => {
  const command = `python -c '${"print(1);".repeat(80)}' --output /tmp/a-very-long-result.json`;

  assert.deepEqual(layoutToolDisplay("execute", command), {
    header: "",
    body: command,
    bodyKind: "call",
  });
});

test("ordinary one-line tool arguments stay in the summary", () => {
  assert.deepEqual(layoutToolDisplay("get_skill_info", "skill_name='agentica'"), {
    header: "skill_name='agentica'",
    body: "",
    bodyKind: "args",
  });
});

test("headless multi-line displays remain in the expanded body", () => {
  const display = "◐ Locate the renderer\n    ○ Update the UI";

  assert.deepEqual(layoutToolDisplay("write_todos", display), {
    header: "",
    body: display,
    bodyKind: "args",
  });
});

test("handoff rows show the model argument the caller passed", () => {
  // Unlike the CLI, no resolved ``model=`` label is added: that needs the live
  // session, which the chat row does not have. An explicit argument still
  // shows, and an omitted one stays omitted rather than being invented.
  const withModel = formatToolDisplay("delegate", {
    task: "port the parser",
    label: "parser port",
    model: "zhipuai/glm-4.7",
  });
  assert.equal(withModel.split("\n")[0], "label='parser port', model='zhipuai/glm-4.7'");

  const inherited = formatToolDisplay("task", { subagent_type: "explore", description: "find it" });
  assert.equal(inherited.split("\n")[0], "subagent_type='explore'");
  assert.ok(!inherited.includes("model="));
});
