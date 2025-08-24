// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { API_URL } from "./api";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Your HTTP API
      "/video": {
        target: API_URL,
        changeOrigin: true,
      },
    },
  },
});