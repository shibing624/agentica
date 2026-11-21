import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const M = await vite.ssrLoadModule("/src/lib/messageFiles.ts");
globalThis.localStorage = { getItem: () => null, setItem() {} };
const { MessageFilesCard } = await vite.ssrLoadModule("/src/workspace/MessageFilesCard.tsx");
after(() => vite.close());

const edit = (path, content = "first\nsecond\n") => ({
  type: "tool", name: "write_file", argsStr: JSON.stringify({ file_path: path, content }), result: "ok",
});
const card = (steps) => renderToStaticMarkup(createElement(MessageFilesCard, {
  text: "", steps, workspace: "/repo", sessionId: "s1", mentionedFiles: [],
  onOpenFile() {}, onOpenDiff() {}, role: "assistant",
}));

test("a recorded edit renders on the first paint without waiting for a workspace stat", () => {
  const html = card([edit("/repo/src/main.js")]);
  assert.match(html, /Edited main\.js/);
  assert.match(html, /<span class="add">\+2<\/span>/);
  assert.doesNotMatch(html, /mfc-row/, "one file is the header, not a duplicate row");
});

test("the edit summary shows total changes and folds after four files", () => {
  const html = card(Array.from({ length: 5 }, (_, i) => edit(`src/file${i}.js`)));
  assert.match(html, /Edited 5 files/);
  assert.match(html, /<span class="add">\+10<\/span>/);
  assert.equal(html.match(/class="mfc-row"/g)?.length, 4);
  assert.match(html, /All 5 files/);
});

test("unfinished and failed edits do not create an edited-file card", () => {
  assert.equal(card([{ ...edit("pending.js"), result: undefined }, { ...edit("failed.js"), result: "Error: failed" }]), "");
});

test("a recorded comparison provides removals that write_file arguments cannot report", () => {
  const html = card([{ ...edit("src/main.js"), diff: "diff -- src/main.js\n@@ -1 +1 @@\n-old\n+new" }]);
  assert.match(html, /<span class="add">\+1<\/span><span class="del">-1<\/span>/);
});

test("mention candidates exclude recorded edits but keep named, read and uploaded paths", () => {
  const ws = "/repo";
  const steps = [
    { type: "tool", name: "write_file", argsStr: JSON.stringify({ file_path: "/repo/src/a.ts", content: "x\n" }), result: "ok" },
    { type: "tool", name: "read_file", argsStr: JSON.stringify({ file_path: "/repo/README.md" }), result: "..." },
  ];
  const got = M.mentionedFileCandidates({ content: "see `src/a.ts` and `docs/b.md`", steps, files: ["up.csv"] }, ws);
  assert.deepEqual(got, ["docs/b.md", "README.md", "up.csv"]);
});

test("a path that climbs out of the workspace is not a candidate", () => {
  assert.deepEqual(M.mentionedFileCandidates({ content: "`../secret.txt`" }, "/repo"), []);
});
