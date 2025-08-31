// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api/video": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
      "/api/auth": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
    port: 5173,
  },
});
