import { useNavigate } from "react-router-dom";

const NotFound = () => {
  const navigate = useNavigate();

  return (
    <div
      style={{
        backgroundColor: "rgba(255, 255, 255, 0.95)",
        borderRadius: "0.25em",
        boxShadow: "0 0 0.25em rgba(0, 0, 0, 0.25)",
        boxSizing: "border-box",
        left: "50%",
        padding: "8vmin",
        paddingLeft: "12vmin",
        paddingRight: "12vmin",
        position: "fixed",
        textAlign: "center",
        top: "50%",
        transform: "translate(-50%, -50%)",
        display: "flex",
        flexDirection: "column",
        gap: "2rem",
        minWidth: "min(100%, 300px)",
      }}
    >
      <div className="flex flex-col items-center justify-center gap-3">
        <div className="text-lg font-bold">Page Not Found</div>
        <div>The page you are looking for does not exist.</div>
      </div>
      <button className="primary-button" onClick={() => navigate("/")}>
        Go to Homepage
      </button>
    </div>
  );
};

export default NotFound;
