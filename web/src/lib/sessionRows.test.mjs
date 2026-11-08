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
const {
  collapseSessionRows, findQueryRanges, searchEventsConversation, searchMsgsConversation, sessionMatches,
} = await vite.ssrLoadModule("/src/lib/sessionRows.ts");
after(() => vite.close());

const rows = Array.from({ length: 8 }, (_, i) => ({
  id: `s${i}`,
  session: { title: `Session ${i}`, preview: `preview ${i}` },
}));

test("search covers title, project and first message", () => {
  assert.equal(sessionMatches(rows[0], "agentica", "session 0"), true);
  assert.equal(sessionMatches(rows[0], "agentica", "agent"), true);
  assert.equal(sessionMatches(rows[0], "agentica", "preview 0"), true);
  assert.equal(sessionMatches(rows[0], "agentica", "missing"), false);
});

test("collapsed groups show five idle rows in their existing order", () => {
  const result = collapseSessionRows(rows, null, false);
  assert.deepEqual(result.visible.map((row) => row.id), ["s0", "s1", "s2", "s3", "s4"]);
  assert.equal(result.hiddenCount, 3);
});

test("current and running rows never disappear behind the fold", () => {
  const changed = rows.map((row) => ({ ...row, session: { ...row.session } }));
  changed[6].session.running = true;
  const result = collapseSessionRows(changed, "s7", false);
  assert.deepEqual(result.visible.map((row) => row.id), ["s0", "s1", "s2", "s3", "s4", "s6", "s7"]);
  assert.equal(result.hiddenCount, 1);
});

test("expanded groups retain every row", () => {
  const result = collapseSessionRows(rows, null, true);
  assert.equal(result.visible.length, rows.length);
  assert.equal(result.hiddenCount, 0);
});
test("pinned rows never disappear behind the fold", () => {
  const changed = rows.map((row) => ({ ...row, session: { ...row.session } }));
  changed[7].session.pinned = true;
  const result = collapseSessionRows(changed, null, false);
  assert.ok(result.visible.some((row) => row.id === "s7"));
  assert.equal(result.visible.filter((row) => !row.session.pinned).length, 5);
});

test("conversation search matches both user queries and assistant answers with turn numbers", () => {
  const events = [
    { type: "user", content: "how do I configure sglang on gpu?" },
    { type: "assistant", content: "Run tmux on 29.206.1.234." },
    { type: "user", content: "what about vllm?" },
    { type: "assistant", content: "Use gpu4 at 28.8.205.124." },
  ];
  const qHits = searchEventsConversation(events, "sglang");
  assert.equal(qHits.length, 1);
  assert.equal(qHits[0].turn, 1);
  assert.equal(qHits[0].role, "user");
  const aHits = searchEventsConversation(events, "28.8.205.124");
  assert.equal(aHits.length, 1);
  assert.equal(aHits[0].turn, 2);
  assert.equal(aHits[0].role, "assistant");
  assert.equal(sessionMatches(rows[0], "agentica", "28.8.205.124", aHits), true);

  const msgHits = searchMsgsConversation(
    [
      { role: "user", content: "what about vllm?" },
      { role: "assistant", content: "Use gpu4 at 28.8.205.124." },
    ],
    1,
    "gpu4",
  );
  assert.equal(msgHits.length, 1);
  assert.equal(msgHits[0].turn, 2);
  assert.equal(msgHits[0].role, "assistant");
});

test("findQueryRanges locates all case-insensitive keyword occurrences for in-page highlighting", () => {
  assert.deepEqual(
    findQueryRanges("Use GPU4 on gpu4.local with SGLang", "gpu4"),
    [[4, 8], [12, 16]],
  );
  assert.deepEqual(findQueryRanges("nothing here", ""), []);
});
