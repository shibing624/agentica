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
const { collapseSessionRows, sessionMatches } = await vite.ssrLoadModule("/src/lib/sessionRows.ts");
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
