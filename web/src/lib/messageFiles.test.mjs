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
const M = await vite.ssrLoadModule("/src/lib/messageFiles.ts");
after(() => vite.close());

test("candidates list edits first, then named and tool paths, each once and workspace-relative", () => {
  const ws = "/repo";
  const steps = [
    { type: "tool", name: "write_file", argsStr: JSON.stringify({ file_path: "/repo/src/a.ts", content: "x\n" }), result: "ok" },
    { type: "tool", name: "read_file", argsStr: JSON.stringify({ file_path: "/repo/README.md" }), result: "..." },
  ];
  const got = M.messageFileCandidates({ content: "see `src/a.ts` and `docs/b.md`", steps, files: ["up.csv"] }, ws);
  assert.deepEqual(got, ["src/a.ts", "docs/b.md", "README.md", "up.csv"]);
});

test("a path that climbs out of the workspace is not a candidate", () => {
  assert.deepEqual(M.messageFileCandidates({ content: "`../secret.txt`" }, "/repo"), []);
});
