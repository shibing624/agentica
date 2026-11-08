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
const FP = await vite.ssrLoadModule("/src/lib/filePath.ts");
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

test("LaTeX files (.tex, .sty, .cls, .bib) resolve to latex highlighting and code preview", () => {
  for (const name of ["paper/main.tex", "style.sty", "neurips.cls", "refs.bib"]) {
    assert.equal(H.languageFromFilename(name), "latex", name);
    assert.equal(FP.classifyPreviewTarget(name), "code", name);
    assert.equal(FP.isFilePathLike(name), true, name);
  }
  const html = H.highlightToHtml("\\documentclass{article}\n\\section{Intro} % comment\n", "tex");
  assert.match(html, /hljs-keyword">\\documentclass</);
  assert.match(html, /hljs-comment">% comment/);
});

test("unknown extensions and dotless files default to plain-text preview while known binaries stay unsupported", () => {
  for (const name of ["notes.txt", "README", "LICENSE", "output.custom", "run.out", "foo.constructor"]) {
    assert.equal(FP.classifyPreviewTarget(name), "text", name);
  }
  assert.equal(H.languageFromFilename("foo.constructor"), "");
  assert.equal(H.resolveLanguage("constructor"), "");
  for (const bin of ["archive.zip", "video.mp4", "weights.safetensors", "module.pyc", "sheet.xlsx"]) {
    assert.equal(FP.classifyPreviewTarget(bin), "unsupported", bin);
  }
  assert.equal(FP.classifyPreviewTarget("figure.png"), "image");
  assert.equal(FP.classifyPreviewTarget("paper.pdf"), "pdf");
  assert.equal(FP.classifyPreviewTarget("index.html"), "html");
  assert.equal(FP.classifyPreviewTarget("README.md"), "md");
});

test("home-relative ~/.agentica session notes resolve against their own parent directory and extract from bare text", () => {
  const notes = "~/.agentica/projects/default/-Users-xuming-Documents-Codes-agentica-2037c358/s_25vv1np1mw6iz4iy.notes.md";
  const target = FP.splitExternalTarget(notes, "/Users/xuming/Documents/Codes/agentica");
  assert.equal(target.isExternal, true);
  assert.equal(target.targetRoot, "~/.agentica/projects/default/-Users-xuming-Documents-Codes-agentica-2037c358");
  assert.equal(target.targetRel, "s_25vv1np1mw6iz4iy.notes.md");
  assert.equal(target.absPath, notes);
  assert.equal(FP.classifyPreviewTarget(notes), "md");
  assert.equal(FP.normalizeMentionedPath(notes, "/Users/xuming/Documents/Codes/agentica"), notes);
  assert.deepEqual(
    FP.extractFilePaths(`${notes} 在web下侧边栏无法显示内容`),
    [notes],
  );
});

test("bare absolute paths are extracted regardless of their top-level directory", () => {
  assert.deepEqual(
    FP.extractFilePaths("见 /data/runs/out.json 和 /mnt/x/report.md，以及 ~/a/b.py。"),
    ["/data/runs/out.json", "/mnt/x/report.md", "~/a/b.py"],
  );
  assert.deepEqual(FP.extractFilePaths("ratio 1/2 and a/b"), []);
});
