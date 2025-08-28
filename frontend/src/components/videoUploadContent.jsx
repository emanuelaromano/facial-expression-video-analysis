import { useState } from "react";
import { setBannerThunk } from "../redux/slices/videoSlice";
import { useDispatch } from "react-redux";

const VideoUploadContent = () => {
  const [scenario, setScenario] = useState("");
  const [currentSectionIndex, setCurrentSectionIndex] = useState(0);
  const dispatch = useDispatch();
  return (
    <div
      className="flex justify-center flex-col items-center relative overflow-hidden"
      style={{
        backgroundColor: "rgba(255, 255, 255)",
        borderRadius: "0.25em",
        boxShadow: "0 0 0.25em rgba(0, 0, 0, 0.25)",
        position: "fixed",
        textAlign: "center",
        top: "0",
        right: "0",
        bottom: "0",
        left: "50%",
        transform: "translateX(-50%)",
        marginTop: "calc(max(20vmin, 3rem) + 2rem)",
        marginBottom: "calc(max(20vmin, 3rem) + 2rem)",
        gap: "2rem",
      }}
    >
      {currentSectionIndex > 0 && (
        <div
          className="absolute top-6 left-6 cursor-pointer underline text-sm hover:translate-y-[-1px] hover:translate-x-[-1px] transition-all duration-300 ease-in-out"
          onClick={() => setCurrentSectionIndex(currentSectionIndex - 1)}
        >
          Back
        </div>
      )}
      <div className="flex flex-col w-full p-20">
        <div
          className={`flex w-[70%] absolute top-[50%] flex-col justify-center items-center gap-5 transition-all duration-500 ease-in-out 
              ${currentSectionIndex === 0 ? "left-[50%] translate-x-[-50%] translate-y-[-50%]" : "left-[-50%] translate-x-[-50%] translate-y-[-50%]"}`}
        >
          <div>Enter the scenario you want to practice</div>
          <div className="flex flex-col w-full gap-3 justify-center items-center">
            <textarea
              className="primary-textbox w-full h-[10rem]"
              type="text"
              placeholder="Scenario"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
            />
            <button
              onClick={() => {
                if (!scenario.length) {
                  dispatch(
                    setBannerThunk("Scenario description is required", "error"),
                  );
                  return;
                }
                setCurrentSectionIndex(1);
              }}
              className="primary-button w-full"
            >
              Submit and continue
            </button>
          </div>
        </div>
        <div
          className={`flex w-[70%] absolute top-[50%] flex-col justify-center items-center gap-2 transition-all duration-500 ease-in-out 
              ${currentSectionIndex === 1 ? "left-[50%] translate-x-[-50%] translate-y-[-50%]" : "left-[150%] translate-x-[-50%] translate-y-[-50%]"}`}
        >
          <div className="py-5 flex flex-col items-center gap-3">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              viewBox="0 0 48 48"
            >
              <path
                stroke="currentColor"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
              />
            </svg>
            <p className="text-sm text-gray-600">
              Click the button below to record your video
            </p>
          </div>
          <div className="flex flex-col gap-6 w-full">
            <button className="primary-button w-full">Record Video</button>
            <p className="text-sm text-gray-600">
              or{" "}
              <button className="cursor-pointer">
                <span className="underline font-bold">upload</span>
              </button>{" "}
              a video
            </p>
          </div>
        </div>
        {/* <div className="w-[50%]"></div> */}
      </div>
    </div>
  );
};

export default VideoUploadContent;
