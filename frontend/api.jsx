const status = "prod";

export const API_URL =
  status === "prod"
    ? "https://backend-app-916307297241.us-central1.run.app"
    : "http://localhost:8080";
