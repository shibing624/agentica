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
const { normalizeBrowserUrl, browserTabLabel } = await vite.ssrLoadModule("/src/lib/browserUrl.ts");
const T = await vite.ssrLoadModule("/src/lib/panelTabs.ts");
after(() => vite.close());

test("an address gets the scheme its host implies", () => {
  assert.equal(normalizeBrowserUrl("localhost:3000"), "http://localhost:3000/");
  assert.equal(normalizeBrowserUrl("127.0.0.1:8000/docs"), "http://127.0.0.1:8000/docs");
  assert.equal(normalizeBrowserUrl("example.com/a?b=1"), "https://example.com/a?b=1");
  assert.equal(normalizeBrowserUrl(" https://example.com "), "https://example.com/");
  assert.equal(normalizeBrowserUrl("http://example.com"), "http://example.com/");
});

test("only http(s) addresses load; text that is not an address is refused, never searched", () => {
  for (const bad of ["", "   ", "javascript:alert(1)", "file:///etc/passwd", "data:text/html,x", "chrome://gpu", "how do I fix this", "intranet", "ftp://x.com"]) {
    assert.equal(normalizeBrowserUrl(bad), null, bad);
  }
});

test("a tab is labelled by its title, else by its host", () => {
  assert.equal(browserTabLabel("https://example.com/a", "Example"), "Example");
  assert.equal(browserTabLabel("http://localhost:3000/x"), "localhost:3000");
});

test("browser tabs survive a reload and a stored non-web url is dropped", () => {
  const record = { tabs: [T.FILES_TAB, { key: "browser:b1", kind: "browser", id: "b1", url: "http://localhost:3000/", title: "App" }], active: "browser:b1", open: true, dirPath: "" };
  const back = T.restoreSessionPanel(T.serializeSessionPanel(record));
  assert.deepEqual(back.tabs[1], record.tabs[1]);
  assert.equal(back.active, "browser:b1");
  const tampered = T.restoreSessionPanel(JSON.stringify({ open: true, active: "files", tabs: [{ kind: "browser", id: "x", url: "javascript:alert(1)", title: "t" }] }));
  assert.equal(tampered.tabs[1].url, "");
});
