import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8881", changeOrigin: false },
      "/ws": { target: "ws://127.0.0.1:8881", ws: true },
      "/health": { target: "http://127.0.0.1:8881", changeOrigin: false },
    },
  },
  build: {
    outDir: path.resolve(__dirname, "dist"),
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (/node_modules\/(react|react-dom|scheduler)\//.test(id)) return "react";
          if (id.includes("node_modules/katex/")) return "math";
          if (/node_modules\/(react-markdown|remark-|rehype-|micromark|mdast-|hast-|unified|vfile|unist-)/.test(id)) {
            return "markdown";
          }
        },
      },
    },
  },
});
