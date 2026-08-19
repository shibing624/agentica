import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { LOGO_SRC } from "./icons";
import "./style.css";
import "./trace.css";

const icon = document.createElement("link");
icon.rel = "icon";
icon.type = "image/svg+xml";
icon.href = LOGO_SRC;
document.head.appendChild(icon);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
