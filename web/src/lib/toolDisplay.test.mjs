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
const { formatToolDisplay, layoutToolDisplay, toolImageSource } = await vite.ssrLoadModule("/src/lib/toolDisplay.ts");
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

test("task rows show subagent_type without inventing a model", () => {
  // The chat row has no live session to resolve the subagent's model from,
  // so no ``model=`` label is added.
  const inherited = formatToolDisplay("task", { description: "Find it", prompt: "find it in src/", subagent_type: "explore" });
  assert.equal(inherited.split("\n")[0], "description='Find it', subagent_type='explore'");
  assert.equal(inherited.split("\n")[1].trim(), "find it in src/");
  assert.ok(!inherited.includes("model="));
});

test("an image analysis call names the image it looks at", () => {
  assert.equal(toolImageSource("analyze_image", { image_path_or_url: " /tmp/a.png " }), "/tmp/a.png");
  assert.equal(toolImageSource("analyze_image", { image_path_or_url: "https://x/y.png", question: "?" }), "https://x/y.png");
  assert.equal(toolImageSource("analyze_image", { image_path_or_url: "  " }), null);
  assert.equal(toolImageSource("analyze_image", {}), null);
  assert.equal(toolImageSource("read_file", { image_path_or_url: "/tmp/a.png" }), null);
});
