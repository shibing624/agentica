import { useEffect, useRef, useState } from "react";
import { IconChevronRight, IconExternal, IconRefresh } from "../icons";
import { useStrings } from "../i18n";
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
 * and the open-externally button are the way out. The page is loaded once from
 * `url`; moving around inside it is the page's own business, and the tab only
 * remembers where it ended up.
 */
export function BrowserView({
  url, visible, onChange,
}: {
  url: string;
  visible: boolean;
  onChange: (patch: { url?: string; title?: string }) => void;
}) {
  const S = useStrings();
  const desktop = inDesktopShell();
  const guest = useRef<Guest | null>(null);
  const ready = useRef(false);
  const initial = useRef(url);
  const [current, setCurrent] = useState(url);
  const [draft, setDraft] = useState(url);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState({ back: false, forward: false });
  const [error, setError] = useState(false);
  const [frameKey, setFrameKey] = useState(0);

  useEffect(() => {
    const el = guest.current;
    if (!desktop || !el) return;
    const sync = () => {
      const href = el.getURL();
      setCurrent(href);
      setDraft(href);
      setHistory({ back: el.canGoBack(), forward: el.canGoForward() });
      onChange({ url: href });
    };
    const on = (name: string, fn: (e: any) => void) => {
      el.addEventListener(name, fn);
      return () => el.removeEventListener(name, fn);
    };
    const offs = [
      on("dom-ready", () => { ready.current = true; }),
      on("did-start-loading", () => setLoading(true)),
      on("did-stop-loading", () => setLoading(false)),
      on("did-navigate", sync),
      on("did-navigate-in-page", sync),
      on("page-title-updated", (e) => onChange({ title: String(e.title || "") })),
    ];
    return () => offs.forEach((off) => off());
    // `onChange` is a fresh closure each render; the listeners only need the latest at call time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [desktop]);

  const go = () => {
    const target = normalizeBrowserUrl(draft);
    if (!target) { setError(true); return; }
    setError(false);
    setCurrent(target);
    setDraft(target);
    onChange({ url: target });
    if (desktop) {
      if (ready.current) void guest.current?.loadURL(target);
      else guest.current?.setAttribute("src", target);
    }
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
      {!desktop && current && <div className="bv-note">{S.browser.embedNote}</div>}
      <div className="bv-page">
        {!current && <div className="bv-empty">{S.browser.empty}</div>}
        {current && desktop && (
          <webview ref={guest as any} className="bv-guest" partition={BROWSER_PARTITION} src={initial.current || current} style={{ display: visible ? "flex" : "none" }} />
        )}
        {current && !desktop && (
          <iframe key={frameKey} className="bv-guest" src={current} title={current}
                  sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads" />
        )}
      </div>
    </div>
  );
}
