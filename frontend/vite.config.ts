import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built assets are served by FastAPI (satquery/api/app.py) from satquery/web/dist.
// `npm run dev` proxies the API to a backend on :8000.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../satquery/web/dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: { manualChunks: { three: ["three", "@react-three/fiber"] } },
    },
  },
  server: { proxy: { "/api": "http://localhost:8000" } },
});
