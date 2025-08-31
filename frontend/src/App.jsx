import {
  Routes,
  Route,
  BrowserRouter,
  Navigate,
  Outlet,
} from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";
import Layout from "./layout/layout";
import LandingPage from "./views/landingPage";
import VideoUpload from "./views/videoUpload";
import NotFound from "./views/notFound";
import AuthPage from "./views/authPage";
import { API_URL } from "../api";

function RequireAuth({ status, children }) {
  return status === "validated" ? children : <Navigate to="/" replace />;
}

function App() {
  const [authStatus, setAuthStatus] = useState("checking");

  useEffect(() => {
    const token = localStorage.getItem("token");
    const run = async () => {
      if (!token) {
        setAuthStatus("not-validated");
        return;
      }
      try {
        const res = await axios.get(`${API_URL}/auth/validate`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.data?.validated) {
          setAuthStatus("validated");
        } else {
          localStorage.removeItem("token");
          setAuthStatus("not-validated");
        }
      } catch {
        localStorage.removeItem("token");
        setAuthStatus("not-validated");
      }
    };
    run();
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route
            index
            element={
              authStatus === "validated" ? (
                <LandingPage />
              ) : authStatus === "checking" ? (
                <div className="flex justify-center items-center h-screen">
                  <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-white" />
                </div>
              ) : (
                <AuthPage setAuthStatus={setAuthStatus} />
              )
            }
          />

          <Route
            path="upload"
            element={
              <RequireAuth status={authStatus}>
                <VideoUpload />
              </RequireAuth>
            }
          />

          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
