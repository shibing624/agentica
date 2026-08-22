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
const { layoutToolDisplay } = await vite.ssrLoadModule("/src/lib/toolDisplay.ts");
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
