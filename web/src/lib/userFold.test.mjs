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
const { USER_FOLD_LINES, measureUserBubbleFold } = await vite.ssrLoadModule("/src/lib/userFold.ts");
after(() => vite.close());

test("folds only when scrollHeight exceeds 9 lines plus 1.3x slack", () => {
  // 20px line-height * 9 lines = 180px limit; threshold = 180 * 1.3 = 234px
  const short = measureUserBubbleFold(220, "20px", "14px");
  assert.equal(short.limit, 20 * USER_FOLD_LINES);
  assert.equal(short.overflows, false);

  const tall = measureUserBubbleFold(260, "20px", "14px");
  assert.equal(tall.limit, 180);
  assert.equal(tall.overflows, true);
});

test("falls back to 1.5 * fontSize when computed line-height is 'normal'", () => {
  // "normal" parses to NaN; fallback = 16 * 1.5 = 24px -> limit = 216px, threshold = 280.8px
  const res = measureUserBubbleFold(300, "normal", "16px");
  assert.equal(res.limit, 216);
  assert.equal(res.overflows, true);
});
