import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/video": {
        target: "https://backend-app-101856457372.us-central1.run.app",
        changeOrigin: true,
      },
    },
  },
});
