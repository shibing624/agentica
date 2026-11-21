import { useCallback, useEffect, useRef, useState } from "react";
import { IconChevronRight, IconClose, IconExternal, IconRefresh } from "../icons";
import { useStrings } from "../i18n";
import { showToast } from "../store";
import { BROWSER_PARTITION, inDesktopShell, normalizeBrowserUrl } from "../lib/browserUrl";

/** The Electron `<webview>` methods this tab uses; they throw before `dom-ready`. */
type Guest = HTMLElement & {
  goBack(): void;
  goForward(): void;
  reload(): void;
  loadURL(url: string): Promise<void>;
  canGoBack(): boolean;
  canGoForward(): boolean;
  getURL(): string;
  getTitle(): string;
};

/**
 * A browser tab: an address bar over the page. Inside the desktop shell the
 * page is a `<webview>` (its own storage, no access to the app); in a plain
 * browser it is an iframe, which many sites refuse to be shown in — the note
 * and the open-externally button are the way out. An incoming URL navigates
 * the page, but the parent's echo of a guest's own navigation must not reload it.
 * An agent's `pendingUrl` is only displayed: the user decides whether to load it.
 */
export function BrowserView({
  url, pendingUrl, visible, onChange, onOpen, onDismiss,
}: {
  url: string;
  pendingUrl: string;
  visible: boolean;
  onChange: (patch: { url?: string; title?: string }) => void;
  onOpen: () => void;
  onDismiss: () => void;
}) {
  const S = useStrings();
  const desktop = inDesktopShell();
  const guest = useRef<Guest | null>(null);
  const ready = useRef(false);
  const initial = useRef(url);
  const currentUrl = useRef(url);
  const report = useRef(onChange);
  report.current = onChange;
  const [current, setCurrent] = useState(url);
  const [draft, setDraft] = useState(url);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState({ back: false, forward: false });
  const [error, setError] = useState(false);
  const [frameKey, setFrameKey] = useState(0);
  const hasPage = Boolean(current);
  if (!initial.current && current) initial.current = current;

  useEffect(() => {
    const el = guest.current;
    if (!desktop || !el) return;
    const sync = () => {
      if (!ready.current) return;
      const href = el.getURL();
      currentUrl.current = href;
      setCurrent(href);
      setDraft(href);
      setHistory({ back: el.canGoBack(), forward: el.canGoForward() });
      report.current({ url: href });
    };
    const on = (name: string, fn: (e: any) => void) => {
      el.addEventListener(name, fn);
      return () => el.removeEventListener(name, fn);
    };
    const offs = [
      on("dom-ready", () => { ready.current = true; sync(); }),
      on("did-start-loading", () => setLoading(true)),
      on("did-stop-loading", () => setLoading(false)),
      on("did-navigate", sync),
      on("did-navigate-in-page", sync),
      on("page-title-updated", (e) => report.current({ title: String(e.title || "") })),
    ];
    return () => { ready.current = false; offs.forEach((off) => off()); };
  }, [desktop, hasPage]);

  const navSeq = useRef(0);
  const loadFailed = S.browser.loadFailed;
  const navigate = useCallback((target: string) => {
    const previous = currentUrl.current;
    const seq = ++navSeq.current;
    currentUrl.current = target;
    setError(false);
    setCurrent(target);
    setDraft(target);
    const el = guest.current;
    if (!desktop || !el || !target) return;
    if (!ready.current) {
      el.setAttribute("src", target);
      return;
    }
    // A superseding navigation can cancel loadURL before Electron emits
    // stop-loading; only the latest attempt may roll the view back.
    void el.loadURL(target).catch(() => {
      if (seq !== navSeq.current) return;
      // Restoring the tab URL too: leaving it at the failed target would make
      // the [url] effect retry the navigation forever.
      currentUrl.current = previous;
      setCurrent(previous);
      setDraft(previous);
      setLoading(false);
      report.current({ url: previous });
      showToast(loadFailed, 2500);
    });
  }, [desktop, loadFailed]);

  useEffect(() => {
    if (url !== currentUrl.current) navigate(url);
  }, [url, navigate]);

  const go = () => {
    const target = normalizeBrowserUrl(draft);
    if (!target) { setError(true); return; }
    navigate(target);
    onChange({ url: target });
  };
  const reload = () => {
    if (desktop) { if (ready.current) guest.current?.reload(); } else setFrameKey((k) => k + 1);
  };

  return (
    <div className="bv">
      <form className="bv-bar" onSubmit={(e) => { e.preventDefault(); go(); }}>
        {desktop && (
          <>
            <button type="button" className="ib bv-back" disabled={!history.back} title={S.browser.back} aria-label={S.browser.back}
                    onClick={() => ready.current && guest.current?.goBack()}><IconChevronRight /></button>
            <button type="button" className="ib" disabled={!history.forward} title={S.browser.forward} aria-label={S.browser.forward}
                    onClick={() => ready.current && guest.current?.goForward()}><IconChevronRight /></button>
          </>
        )}
        <button type="button" className={"ib" + (loading ? " bv-loading" : "")} disabled={!current} title={S.browser.reload}
                aria-label={S.browser.reload} onClick={reload}><IconRefresh /></button>
        <input className={"bv-url" + (error ? " bad" : "")} value={draft} spellCheck={false} autoCapitalize="off"
               placeholder={S.browser.addressPlaceholder} aria-label={S.browser.addressPlaceholder}
               autoFocus={!url} title={error ? S.browser.notAnAddress : undefined}
               onChange={(e) => { setDraft(e.target.value); setError(false); }} />
        <a className={"ib" + (current ? "" : " disabled")} href={current || undefined} target="_blank" rel="noreferrer noopener"
           title={S.browser.openExternal} aria-label={S.browser.openExternal}><IconExternal /></a>
      </form>
      {pendingUrl && (
        <div className="bv-pending" role="group" aria-label={S.browser.agentWantsToOpen}>
          <span className="bv-pending-label">{S.browser.agentWantsToOpen}</span>
          <span className="bv-pending-url" title={pendingUrl}>{pendingUrl}</span>
          <button type="button" className="bv-pending-open" onClick={onOpen}>{S.browser.openHere}</button>
          <button type="button" className="ib" onClick={onDismiss} title={S.browser.dismiss} aria-label={S.browser.dismiss}>
            <IconClose />
          </button>
        </div>
      )}
      {!desktop && current && <div className="bv-note">{S.browser.embedNote}</div>}
      <div className="bv-page">
        {!current && <div className="bv-empty">{S.browser.empty}</div>}
        {current && desktop && (
          <webview ref={guest as any} className="bv-guest" partition={BROWSER_PARTITION} src={initial.current} style={{ display: visible ? "flex" : "none" }} />
        )}
        {current && !desktop && (
          <iframe key={frameKey} className="bv-guest" src={current} title={current}
                  sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads" />
        )}
      </div>
    </div>
  );
}
