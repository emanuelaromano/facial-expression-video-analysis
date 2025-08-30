import johnKennedy from "../assets/john-francis-kennedy.jpg";
import martinLutherKing from "../assets/martin-luther-king.png";
import winstonChurchill from "../assets/winston-churchill.jpg";
import steveJobs from "../assets/steve-jobs.jpg";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const images = [martinLutherKing, johnKennedy, winstonChurchill, steveJobs];

const LandingPage = () => {
  const navigate = useNavigate();

  const [imageOrder, setImageOrder] = useState(images.map((_, index) => index));
  const [isMounted, setIsMounted] = useState({
    status: false,
    setInterval: 3000,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      if (!isMounted.status) {
        setIsMounted({
          status: true,
          setInterval: 6000,
        });
      }
      const newImageOrder = [...imageOrder];
      const temp = newImageOrder[0];
      newImageOrder[0] = newImageOrder[1];
      newImageOrder[1] = newImageOrder[2];
      newImageOrder[2] = newImageOrder[3];
      newImageOrder[3] = temp;
      setImageOrder(newImageOrder);
    }, isMounted.setInterval);

    return () => clearInterval(interval);
  }, [imageOrder, isMounted]);

  const handleTryDemo = () => {
    navigate("/video-upload");
  };

  return (
    <div className="h-screen w-screen overflow-hidden">
      <div className="fixed top-0 left-0 w-full h-full">
        <img
          src={images[imageOrder[0]]}
          alt="Image 1"
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      </div>
      <div className="bg1"></div>
      <div className="bg2"></div>
      <div className="bg3"></div>
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
          <div className="text-lg font-bold">Get Started</div>
          <div>Learn to Speak Like a Leader</div>
        </div>
        <button className="primary-button" onClick={handleTryDemo}>
          Try the Demo
        </button>
      </div>
    </div>
  );
};

export default LandingPage;
