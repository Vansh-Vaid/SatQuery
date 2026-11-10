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
      output: {
        // keep the 3D runtime in its own cacheable chunk, shared by the landing page and the console
        manualChunks: (id) => (/node_modules[\\/](three|@react-three)[\\/]/.test(id) ? "three" : undefined),
      },
    },
  },
  server: { proxy: { "/api": "http://localhost:8000" } },
});
