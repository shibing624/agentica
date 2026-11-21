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
const { normalizeBrowserUrl, browserTabLabel, browserNavigationUrl } = await vite.ssrLoadModule("/src/lib/browserUrl.ts");
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

test("navigation events accept only absolute web URLs outside the app origin without URL credentials", () => {
  const origin = "http://127.0.0.1:8881";
  assert.equal(browserNavigationUrl("http://localhost:3100/demo?q=1#section", origin), "http://localhost:3100/demo?q=1#section");
  assert.equal(browserNavigationUrl("https://intranet/demo", origin), "https://intranet/demo");
  for (const value of [null, { url: "https://example.com" }, "example.com", "/chat", "about:blank",
    "file:///tmp/note", "https://user:password@example.com/", `${origin}/chat`]) {
    assert.equal(browserNavigationUrl(value, origin), null, String(value));
  }
});

test("an agent URL is displayed, and only a user confirmation loads it", () => {
  const manual = { key: "browser:manual", kind: "browser", id: "manual", url: "https://manual.test/", title: "Manual" };
  const record = { ...T.initialSessionPanel(), tabs: [T.FILES_TAB, manual], active: manual.key };
  const first = T.requestBrowserNavigation(record, "https://example.test/start");
  assert.equal(first.open, true);
  const agentKey = first.active;
  const agentTab = first.tabs.find((tab) => tab.key === agentKey);
  assert.equal(agentTab.url, "");
  assert.equal(agentTab.pendingUrl, "https://example.test/start");
  assert.deepEqual(first.tabs.find((tab) => tab.key === manual.key), manual);
  assert.equal(first.tabs.length, 3);

  const background = { ...first, active: manual.key, open: false };
  const next = T.requestBrowserNavigation(background, "https://example.test/done");
  assert.equal(next.open, false);
  assert.equal(next.active, manual.key);
  assert.equal(next.tabs.find((tab) => tab.key === agentKey).url, "");
  assert.equal(next.tabs.find((tab) => tab.key === agentKey).pendingUrl, "https://example.test/done");
  assert.equal(T.requestBrowserNavigation(next, "https://example.test/done"), next);

  const dismissed = T.dismissBrowserNavigation(next, agentKey);
  const dismissedTab = dismissed.tabs.find((tab) => tab.key === agentKey);
  assert.equal(dismissedTab.pendingUrl, "");
  assert.equal(dismissedTab.url, "");

  const confirmed = T.confirmBrowserNavigation(next, agentKey);
  const confirmedTab = confirmed.tabs.find((tab) => tab.key === agentKey);
  assert.equal(confirmedTab.url, "https://example.test/done");
  assert.equal(confirmedTab.pendingUrl, "");
  assert.equal(confirmed.active, manual.key);
});

test("a proposed URL is never persisted or restored from localStorage", () => {
  const proposed = { key: "browser:agent", kind: "browser", id: "agent", url: "", title: "", pendingUrl: "https://example.test/x" };
  const record = { ...T.initialSessionPanel(), tabs: [T.FILES_TAB, proposed] };
  const raw = T.serializeSessionPanel(record);
  assert.ok(!raw.includes("example.test"));
  assert.equal(T.restoreSessionPanel(raw).tabs[1].pendingUrl, undefined);
  const tampered = T.restoreSessionPanel(JSON.stringify({
    open: true, tabs: [{ kind: "browser", id: "x", url: "", title: "", pendingUrl: "javascript:alert(1)" }],
  }));
  assert.equal(tampered.tabs[1].pendingUrl, undefined);
});

test("an SSE browser_navigate payload stays a proposal end to end", () => {
  const origin = "http://127.0.0.1:8881";
  const evt = { event: "browser_navigate", data: { url: "https://example.test/done?q=1" } };
  const url = browserNavigationUrl(evt.data.url, origin);
  assert.equal(url, "https://example.test/done?q=1");
  const proposed = T.requestBrowserNavigation(T.initialSessionPanel(), url);
  const tab = proposed.tabs.find((t) => t.kind === "browser");
  assert.equal(tab.url, "");
  assert.equal(T.confirmBrowserNavigation(proposed, tab.key).tabs.find((t) => t.key === tab.key).url, url);
  assert.equal(browserNavigationUrl(`${origin}/chat`, origin), null);
});
