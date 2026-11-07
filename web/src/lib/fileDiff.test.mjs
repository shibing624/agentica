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
const D = await vite.ssrLoadModule("/src/lib/fileDiff.ts");
after(() => vite.close());

// The exact shape agentica.tools.helpers.files_unified_diff writes.
const TWO_FILES = [
  "diff -- a.py",
  "@@ -34,4 +34,4 @@",
  " ctx1",
  "-old",
  "+new",
  " ctx2",
  "",
  "diff -- new.md",
  "@@ -0,0 +1 @@",
  "+hello",
].join("\n");

test("parseFileDiffs splits files and reads hunk ranges, defaulting a missing count to 1", () => {
  const files = D.parseFileDiffs(TWO_FILES);
  assert.deepEqual(files.map((f) => f.path), ["a.py", "new.md"]);
  assert.deepEqual(files[0].hunks[0], {
    oldStart: 34, oldLines: 4, newStart: 34, newLines: 4, lines: [" ctx1", "-old", "+new", " ctx2"],
  });
  assert.deepEqual(
    [files[1].hunks[0].oldStart, files[1].hunks[0].oldLines, files[1].hunks[0].newStart, files[1].hunks[0].newLines],
    [0, 0, 1, 1],
  );
});

test("parseFileDiffs drops the gateway's clip marker instead of drawing it as a line", () => {
  const files = D.parseFileDiffs("diff -- a.py\n@@ -1 +1 @@\n-x\n+y\n... (120 more chars)");
  assert.deepEqual(files[0].hunks[0].lines, ["-x", "+y"]);
});

test("hunkRows numbers context on both sides, deletions old-only, additions new-only", () => {
  const [h] = D.parseFileDiffs(TWO_FILES)[0].hunks;
  assert.deepEqual(D.hunkRows(h), [
    { kind: "context", old: 34, new: 34, text: "ctx1" },
    { kind: "del", old: 35, text: "old" },
    { kind: "add", new: 35, text: "new" },
    { kind: "context", old: 36, new: 36, text: "ctx2" },
  ]);
});

test("splitRows pairs a deletion run with the addition run after it", () => {
  const h = { oldStart: 1, oldLines: 3, newStart: 1, newLines: 2, lines: ["-a", "-b", "+A", " c"] };
  assert.deepEqual(D.splitRows(h), [
    { left: { no: 1, text: "a", kind: "del" }, right: { no: 1, text: "A", kind: "add" } },
    { left: { no: 2, text: "b", kind: "del" } },
    { left: { no: 3, text: "c", kind: "context" }, right: { no: 2, text: "c", kind: "context" } },
  ]);
});

test("fileDiffsOf groups edits per workspace-relative path in call order and skips other tools", () => {
  const steps = [
    { type: "tool", name: "apply_patch", diff: "diff -- /ws/a.py\n@@ -1 +1 @@\n-x\n+y" },
    { type: "tool", name: "execute", result: "ok" },
    { type: "tool", name: "write_file", diff: "diff -- a.py\n@@ -1 +1,2 @@\n y\n+z" },
    { type: "tool", name: "write_file", diff: "diff -- /elsewhere/b.py\n@@ -0,0 +1 @@\n+q" },
  ];
  const byPath = D.fileDiffsOf(steps, "/ws");
  assert.deepEqual([...byPath.keys()], ["a.py"]);
  const a = byPath.get("a.py");
  assert.equal(a.edits.length, 2);
  assert.deepEqual(D.diffCounts(a.edits), { added: 2, removed: 1 });
  assert.equal(D.oneSided(a.edits), false);
  assert.equal(D.oneSided([[{ oldStart: 0, oldLines: 0, newStart: 1, newLines: 1, lines: ["+q"] }]]), true);
});

test("renderedEdits cuts at the line budget and says so", () => {
  const h = { oldStart: 1, oldLines: 3, newStart: 1, newLines: 3, lines: [" a", " b", " c"] };
  const { edits, truncated } = D.renderedEdits([[h], [h]], 4);
  assert.equal(truncated, true);
  assert.deepEqual(edits.map((hs) => hs.map((x) => x.lines.length)), [[3], [1]]);
  assert.equal(D.renderedEdits([[h]], 3).truncated, false);
});
