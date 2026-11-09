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
const { createStreamFollow, stickToBottom } = await vite.ssrLoadModule("/src/lib/streamFollow.ts");
after(() => vite.close());

test("switching from a long session to a shorter one keeps stick true on stickToBottom", () => {
  const follow = createStreamFollow();
  const longEl = { scrollTop: 0, scrollHeight: 10000, clientHeight: 800 };
  stickToBottom(longEl, follow);
  assert.equal(longEl.scrollTop, 10000);
  assert.equal(follow.stick, true);

  // Switch to a shorter session and scroll up in it, then switch back.
  follow.wheel(-40);
  assert.equal(follow.stick, false);

  follow.resume();
  const shortEl = { scrollTop: 400, scrollHeight: 1200, clientHeight: 800 };
  stickToBottom(shortEl, follow);
  assert.equal(follow.stick, true);
});

test("a delayed scroll event after resume() does not clear stick before layout settles", () => {
  const follow = createStreamFollow();
  follow.wheel(-20);
  assert.equal(follow.stick, false);

  // Session switch calls resume(); if a scroll event fires while scrollTop is still
  // clamped from the previous shorter session, stick must stay true.
  follow.resume();
  follow.scrolled({ scrollTop: 0, scrollHeight: 8000, clientHeight: 800 });
  assert.equal(follow.stick, true);
});

test("content height changes that clamp scrollTop do not unstick, while same-height upward scroll does", () => {
  const follow = createStreamFollow();
  follow.scrolled({ scrollTop: 9200, scrollHeight: 10000, clientHeight: 800 });
  assert.equal(follow.stick, true);

  // Layout shrink/reflow clamps scrollTop while changing scrollHeight: stay stuck.
  follow.scrolled({ scrollTop: 3000, scrollHeight: 5000, clientHeight: 800 });
  assert.equal(follow.stick, true);

  // User drags scrollbar upward at the same scrollHeight: unstick.
  follow.scrolled({ scrollTop: 2600, scrollHeight: 5000, clientHeight: 800 });
  assert.equal(follow.stick, false);

  // User scrolls back within 80px of the bottom: re-stick.
  follow.scrolled({ scrollTop: 4150, scrollHeight: 5000, clientHeight: 800 });
  assert.equal(follow.stick, true);
});
