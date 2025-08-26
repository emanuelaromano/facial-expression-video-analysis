const status = "dev";
export const API_URL =
  status === "prod"
    ? "https://backend-app-916307297241.us-central1.run.app"
    : "http://localhost:8080";
export const WS_URL =
  status === "prod"
    ? "https://backend-app-916307297241.us-central1.run.app"
    : "ws://localhost:8080";
