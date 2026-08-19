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
    outDir: path.resolve(__dirname, "../agentica/gateway/ui"),
    emptyOutDir: true,
    sourcemap: false,
  },
});
