/**
 * One gateway terminal in a panel tab (dsh's sidebar terminal): xterm.js over a
 * socket to /api/terminals/{id}/stream. The shell lives on the gateway, so
 * unmounting this only detaches; the next mount replays the screen.
 */
import { useEffect, useRef, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";
import { useStrings } from "../i18n";

const FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

function streamUrl(id: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/api/terminals/${encodeURIComponent(id)}/stream`;
}

/** The page's own ink and ground, so the screen follows the light/dark theme. */
function pageTheme(node: HTMLElement) {
  const css = getComputedStyle(node);
  const fg = css.color;
  const bg = css.backgroundColor;
  return { foreground: fg, background: bg, cursor: fg, cursorAccent: bg, selectionBackground: "rgba(120,140,170,.35)" };
}

export function TerminalView({
  id, visible, onGone,
}: {
  id: string;
  visible: boolean;
  /** The gateway no longer has this shell. An exit is not this: the tab stays
   *  to show the exit code until the user closes it. */
  onGone: () => void;
}) {
  const S = useStrings();
  const host = useRef<HTMLDivElement>(null);
  const term = useRef<Terminal | null>(null);
  const fit = useRef<FitAddon | null>(null);
  const sock = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<"connecting" | "live" | "lost" | "exited">("connecting");
  const [attempt, setAttempt] = useState(0);
  const onGoneRef = useRef(onGone);
  onGoneRef.current = onGone;
  const visibleRef = useRef(visible);
  visibleRef.current = visible;

  useEffect(() => {
    const node = host.current!;
    const xterm = new Terminal({
      cursorBlink: true, fontFamily: FONT, fontSize: 12, scrollback: 5000,
      minimumContrastRatio: 4.5, theme: pageTheme(node), allowProposedApi: false,
    });
    const addon = new FitAddon();
    xterm.loadAddon(addon);
    xterm.open(node);
    term.current = xterm;
    fit.current = addon;
    return () => { xterm.dispose(); term.current = null; fit.current = null; };
  }, []);

  // `data-theme` on <html> is the resolved light/dark, including an "auto"
  // that follows the system; repaint whenever it flips.
  useEffect(() => {
    const repaint = () => { if (host.current && term.current) term.current.options.theme = pageTheme(host.current); };
    const observer = new MutationObserver(repaint);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const xterm = term.current!;
    const ws = new WebSocket(streamUrl(id));
    sock.current = ws;
    setStatus("connecting");
    const sendSize = () => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "resize", cols: xterm.cols, rows: xterm.rows }));
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "snapshot") {
        // Replay at the size the shell drew it at (dsh does the same): a
        // line filled to 80 columns replayed at 120 would not wrap, and zsh's
        // end-of-line marker shows up as a stray "%". Fit after it is drawn.
        xterm.reset();
        if (msg.info?.cols && msg.info?.rows) xterm.resize(msg.info.cols, msg.info.rows);
        xterm.write(msg.data, () => {
          setStatus("live");
          if (visibleRef.current && host.current?.clientWidth) { fit.current?.fit(); sendSize(); }
        });
      } else if (msg.type === "output") {
        xterm.write(msg.data);
      } else if (msg.type === "exit") {
        setStatus("exited");
        xterm.write(`\r\n\x1b[2m[${S.terminal.exited(msg.code)}]\x1b[0m\r\n`);
      } else if (msg.type === "closed") {
        onGoneRef.current();
      }
    };
    ws.onclose = (ev) => {
      if (sock.current !== ws) return;
      if (ev.code === 4404) { onGoneRef.current(); return; }
      setStatus((s) => (s === "exited" ? s : "lost"));
    };
    const input = xterm.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "input", data }));
    });
    const resized = xterm.onResize(sendSize);
    return () => {
      input.dispose();
      resized.dispose();
      sock.current = null;
      ws.close();
    };
  }, [id, attempt, S.terminal]);

  useEffect(() => {
    const node = host.current!;
    const observer = new ResizeObserver(() => {
      if (!visibleRef.current || node.clientWidth === 0 || node.clientHeight === 0) return;
      fit.current?.fit();
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!visible) return;
    const node = host.current;
    if (node?.clientWidth && node.clientHeight) fit.current?.fit();
    term.current?.focus();
  }, [visible]);

  return (
    <div className="term-pane">
      <div className="term-screen" ref={host} />
      {status === "lost" && (
        <div className="term-banner">
          <span>{S.terminal.disconnected}</span>
          <button type="button" className="ws-tool-btn" onClick={() => setAttempt((n) => n + 1)}>{S.terminal.reconnect}</button>
        </div>
      )}
    </div>
  );
}
