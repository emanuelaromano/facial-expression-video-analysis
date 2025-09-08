let status = "prod";

export const API_URL =
  status !== "dev"
    ? "https://backend-app-916307297241.us-central1.run.app/api"
    : "http://localhost:8080/api";
