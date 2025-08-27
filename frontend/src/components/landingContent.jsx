import { useNavigate } from "react-router-dom";

const LandingContent = () => {
  const navigate = useNavigate();

  const handleTryDemo = () => {
    navigate("/video");
  };

  return (
    <div
      style={{
        backgroundColor: "rgba(255, 255, 255, 0.8)",
        borderRadius: "0.25em",
        boxShadow: "0 0 0.25em rgba(0, 0, 0, 0.25)",
        boxSizing: "border-box",
        left: "50%",
        padding: "10vmin",
        position: "fixed",
        textAlign: "center",
        top: "50%",
        transform: "translate(-50%, -50%)",
        display: "flex",
        flexDirection: "column",
        gap: "2rem",
      }}
    >
      <div>Learn to Speak Like a Leader</div>
      <button className="primary-button" onClick={handleTryDemo}>
        Try the Demo
      </button>
    </div>
  );
};

export default LandingContent;
