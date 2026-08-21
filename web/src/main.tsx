import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "katex/dist/katex.min.css";
import "./style.css";
import "./trace.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
