import johnKennedy from "../assets/john-francis-kennedy.jpg";
import martinLutherKing from "../assets/martin-luther-king.png";
import winstonChurchill from "../assets/winston-churchill.jpg";
import steveJobs from "../assets/steve-jobs.jpg";
import { useState, useEffect } from "react";
import axios from "axios";
import { useDispatch } from "react-redux";
import { setBannerThunk } from "../redux/slices/videoSlice";
import { API_URL } from "../../api";

const images = [martinLutherKing, johnKennedy, winstonChurchill, steveJobs];

const AuthPage = ({ setAuthStatus }) => {
  const dispatch = useDispatch();
  const [accessCode, setAccessCode] = useState("");
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.get(`${API_URL}/auth/validate`, {
        headers: { Authorization: `Bearer ${accessCode}` },
      });
      console.log({ response: res });
      if (res.data.validated) {
        localStorage.setItem("token", accessCode);
        setAuthStatus("validated");
      }
    } catch (error) {
      console.log(error);
      dispatch(setBannerThunk("Access code was not validated", "error"));
    }
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
          paddingLeft: "5vmin",
          paddingRight: "5vmin",
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
          <div className="text-lg font-bold">Coming Soon</div>
          <div>
            If you have an access code for testing, please enter it below.
          </div>
          <div>
            To request one, please{" "}
            <a
              href="mailto:emanuela.romano.1998@gmail.com"
              className="text-blue-600 underline"
            >
              contact us
            </a>
            .
          </div>
        </div>
        <form
          onSubmit={handleSubmit}
          className="flex flex-col items-center justify-center gap-3"
        >
          <input
            type="password"
            className="primary-textbox text-sm w-full h-[3rem] p-1 text-left resize-none flex items-center justify-center "
            placeholder="Access Code"
            value={accessCode}
            onChange={(e) => setAccessCode(e.target.value)}
            spellCheck={false}
            autoComplete="off"
          />
          <button type="submit" className="primary-button w-full">
            Submit
          </button>
        </form>
      </div>
    </div>
  );
};

export default AuthPage;
