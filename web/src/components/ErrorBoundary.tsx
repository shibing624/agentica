import { Component, type ErrorInfo, type ReactNode } from "react";
import { getStrings } from "../i18n";
import { shouldAutoReload } from "../lib/staleBundle";

/**
 * A lazy page that fails to load (the gateway was upgraded and the page still
 * holds the old build's file names) or any error thrown while rendering.
 *
 * Without a boundary React unmounts the whole tree on an uncaught render error:
 * a blank window with no sidebar to click, which on the desktop shell, with no
 * address bar and no reload shortcut, reads as a freeze. Here the user gets a
 * message and a Reload button; a stale-bundle failure just reloads once.
 */

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: unknown }> {
  state = { error: null as unknown };

  static getDerivedStateFromError(error: unknown) {
    return { error: error ?? new Error("unknown") };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
    if (shouldAutoReload(error)) window.location.reload();
  }

  render() {
    if (this.state.error === null) return this.props.children;
    const S = getStrings();
    return (
      <div className="crash" role="alert">
        <h2>{S.common.crashTitle}</h2>
        <p>{S.common.crashHint}</p>
        <code>{String((this.state.error as { message?: string })?.message ?? this.state.error).slice(0, 240)}</code>
        <button type="button" className="dp-btn primary" onClick={() => window.location.reload()}>
          {S.common.reload}
        </button>
      </div>
    );
  }
}
