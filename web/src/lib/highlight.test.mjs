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
const H = await vite.ssrLoadModule("/src/lib/highlight.ts");
after(() => vite.close());

test("python fences tagged python3 / py3 are highlighted as python", () => {
  for (const tag of ["python", "py", "python3", "py3"]) {
    assert.equal(H.resolveLanguage(tag), "python", tag);
    assert.match(H.highlightToHtml("def f():\n    return 1\n", tag), /hljs-keyword">def</, tag);
  }
});

test("a large source file still gets colors: the cap covers the 256 KB preview", () => {
  const body = "def f(x):\n    return x  # n\n".repeat(4000); // ~120 KB, like runner/loop.py
  assert.ok(body.length > 64 * 1024 && body.length <= H.HIGHLIGHT_LIMIT);
  assert.match(H.highlightToHtml(body, "python", body.length <= H.HIGHLIGHT_LIMIT), /hljs-comment/);
});
