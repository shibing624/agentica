import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "katex/dist/katex.min.css";
import "./style.css";
import "./trace.css";

// The desktop shell appends AgenticaDesktop to its user agent (desktop/main.js).
// On macOS it hides the title bar, so the page has to leave room for the traffic
// lights and mark its own header as the drag area.
if (/AgenticaDesktop/.test(navigator.userAgent) && /Macintosh/.test(navigator.userAgent)) {
  document.documentElement.dataset.desktop = "mac";
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
