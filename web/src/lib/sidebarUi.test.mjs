import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../..", import.meta.url));
const read = (path) => readFile(new URL(path, `file://${root}`), "utf8");

test("sidebar timestamp uses dsh's smaller 10px step", async () => {
  const css = await read("web/src/style.css");
  assert.match(css, /\.s-item \.mt\{[^}]*font-size:10px/);
});

test("selected session has a stronger fill than an unselected hover", async () => {
  const css = await read("web/src/style.css");
  assert.match(css, /:root\{[\s\S]*?--selected:[^;]+;/);
  assert.match(css, /\[data-theme="dark"\]\{[\s\S]*?--selected:[^;]+;/);
  assert.match(css, /\.s-item\.active\{[^}]*background:var\(--selected\)/);
  assert.match(css, /\.s-item:hover:not\(\.active\)\{background:var\(--hover\)\}/);
});

test("the active workspace uses an open folder glyph with an accent color", async () => {
  const [css, icons, tree] = await Promise.all([
    read("web/src/style.css"),
    read("web/src/icons.tsx"),
    read("web/src/components/SessionTree.tsx"),
  ]);
  assert.match(css, /\.p-head\.workspace-active \.p-icon\{color:var\(--accent\)\}/);
  assert.match(icons, /export const IconFolderOpen/);
  assert.match(tree, /className=\{"p-head"/);
  assert.match(tree, /workspace-active/);
  assert.match(tree, /<IconFolderOpen \/>/);
  assert.match(tree, /<IconFolder \/>/);
});

test("the workspace menu offers Reveal in Finder on macOS only", async () => {
  const [tree, i18n, store, data, settings] = await Promise.all([
    read("web/src/components/SessionTree.tsx"),
    read("web/src/i18n.ts"),
    read("web/src/store.ts"),
    read("web/src/data.ts"),
    read("gateway/src/agentica_gateway/routes/settings.py"),
  ]);
  assert.match(tree, /serverPlatform === "darwin"/);
  assert.match(tree, /S\.workspace\.revealInFinder/);
  assert.match(tree, /api\.revealPathApi\(g\.dir\)/);
  assert.match(tree, /IconFinder/);
  assert.match(i18n, /revealInFinder: "Show in Finder"/);
  assert.match(i18n, /revealInFinder: "在 Finder 中显示"/);
  assert.match(store, /serverPlatform: string/);
  assert.match(data, /serverPlatform: data\.platform/);
  assert.match(settings, /"platform": sys\.platform/);
});
