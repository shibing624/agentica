import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../..", import.meta.url));
const read = (path) => readFile(new URL(path, `file://${root}`), "utf8");

// dsh renders reasoning through `MarkdownText variant="compact"`: the body sans
// face at the secondary tier (13/20) in the tertiary colour — never the mono,
// italic, pre-wrapped quote the old `.think-text` rule carried.
test("reasoning body reads as compact sans Markdown, not a mono italic quote", async () => {
  const css = await read("web/src/style.css");
  const rule = /\.wr-body\.think-text\{([^}]*)\}/.exec(css);
  assert.ok(rule, "the reasoning body rule is missing");
  const body = rule[1];
  assert.match(body, /font-family:var\(--sans\)/);
  assert.match(body, /font-size:13px/);
  assert.match(body, /line-height:20px/);
  assert.match(body, /font-style:normal/);
  assert.match(body, /color:var\(--fg3\)/);
  assert.doesNotMatch(body, /font-family:var\(--mono\)/);
  // The base class that leaked mono + italic is gone.
  assert.doesNotMatch(css, /^\.think-text\{/m);
});

test("an open work group owns its scrolling and follows its tail", async () => {
  const [group, sticky, css] = await Promise.all([
    read("web/src/components/WorkGroup.tsx"),
    read("web/src/lib/stickyBody.ts"),
    read("web/src/style.css"),
  ]);
  // The capped body is the scroller, and its content is what growth is measured on.
  assert.match(group, /useStickyBody\(open, running\)/);
  assert.match(group, /ref=\{body\.bodyRef\}/);
  assert.match(group, /ref=\{body\.contentRef\}/);
  assert.match(group, /\.\.\.body\.handlers/);
  // Growing content keeps the tail in view while the reader has not scrolled away.
  assert.match(sticky, /new ResizeObserver/);
  assert.match(sticky, /if \(follow\.stick\) stickToBottom\(el, follow\)/);
  // An upward gesture leaves the tail; the shared rule resumes within 80px.
  assert.match(sticky, /follow\.wheel\(e\.deltaY\)/);
  assert.match(sticky, /follow\.touchMove\(/);
  // The body is the only scroller inside the group: the text block never is.
  assert.doesNotMatch(css, /\.wr-body\.think-text\{[^}]*overflow-y:auto/);
});
