import johnKennedy from "../assets/john-francis-kennedy.jpg";
import martinLutherKing from "../assets/martin-luther-king.png";
import winstonChurchill from "../assets/winston-churchill.jpg";
import steveJobs from "../assets/steve-jobs.jpg";
import { useState, useEffect } from "react";
import LandingContent from "../components/landingContent";

const images = [johnKennedy, martinLutherKing, winstonChurchill, steveJobs];

const LandingPage = () => {
  const [imageOrder, setImageOrder] = useState(images.map((_, index) => index));

  useEffect(() => {
    const interval = setInterval(() => {
      const newImageOrder = [...imageOrder];
      const temp = newImageOrder[0];
      newImageOrder[0] = newImageOrder[1];
      newImageOrder[1] = newImageOrder[2];
      newImageOrder[2] = newImageOrder[3];
      newImageOrder[3] = temp;
      setImageOrder(newImageOrder);
    }, 3000);

    return () => clearInterval(interval);
  }, [imageOrder]);

  return (
    <div>
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
      <LandingContent />
    </div>
  );
};

export default LandingPage;
