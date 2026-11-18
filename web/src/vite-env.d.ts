/// <reference types="vite/client" />

// Electron's <webview>, which only exists inside the desktop shell.
declare namespace React.JSX {
  interface IntrinsicElements {
    webview: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
      src?: string;
      partition?: string;
    };
  }
}
