const status = "prod";
export const API_URL =
  status === "prod"
    ? "https://hireview-prep-916307297241.us-central1.run.app"
    : "http://localhost:8080";
export const WS_URL =
  status === "prod"
    ? "https://hireview-prep-916307297241.us-central1.run.app"
    : "ws://localhost:8080";
