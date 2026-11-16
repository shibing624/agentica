import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// `.main` is the app's content column (flex: 1, height: 100vh). A component
// that borrowed it as a modifier ("task-model main") turned a model chip into a
// viewport-high purple slab. Layout-owning class names must not double as
// modifiers: here, any selector that puts `.main` on something other than itself.
const css = readFileSync(new URL("../style.css", import.meta.url), "utf8");

test("`.main` is never used as a modifier on another class", () => {
  const compound = css.match(/[A-Za-z0-9_\]\)-]\.main\b/g) ?? [];
  assert.deepEqual(compound, []);
});

test("no component hands `main` to the content column's class as a bare modifier", () => {
  const files = ["components/TaskCard.tsx", "pages/TracesPage.tsx"];
  for (const f of files) {
    const src = readFileSync(new URL(`../${f}`, import.meta.url), "utf8");
    assert.ok(!/\?\s*" main"/.test(src), `${f} adds a bare "main" class`);
  }
});
