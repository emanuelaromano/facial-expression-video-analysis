import { useDispatch } from "react-redux";
import { setBannerThunk } from "../redux/slices/videoSlice";
import { useRef, useState } from "react";
import { CircleX } from "lucide-react";

function VideoUpload() {
  const dispatch = useDispatch();
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      dispatch(setBannerThunk("Video uploaded successfully", "success"));
    } else {
      dispatch(setBannerThunk("No video file selected", "error"));
    }
  };

  const handleUploadVideo = () => {
    fileInputRef.current.click();
  };

  const handleRecordVideo = async () => {
    alert("Record video clicked");
  };

  const handleClearVideo = () => {
    setSelectedFile(null);
  };

  return (
    <>
      <div className="flex flex-col items-center">
        <h1 className="text-3xl font-bold mb-8 text-gray-800">
          Upload Your Video
        </h1>
        <div className="bg-white p-8 rounded-lg shadow-lg border-2 border-dashed border-gray-300 hover:border-[var(--pink-500)] transition-colors">
          <div className="text-center">
            <div className="mb-4">
              <svg
                className="mx-auto h-12 w-12 text-gray-400"
                stroke="currentColor"
                fill="none"
                viewBox="0 0 48 48"
              >
                <path
                  d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            <p className="text-lg text-gray-600 mb-4">
              Click the button below to select your video file
            </p>
            <div className="flex flex-row gap-2 px-20 justify-center items-center">
              <button
                onClick={handleUploadVideo}
                className="bg-[var(--pink-500)] hover:bg-[var(--pink-700)] text-white font-bold py-3 px-6 rounded-full transition-colors duration-200 shadow-lg hover:shadow-xl"
              >
                Select Video
              </button>
              <p className="text-lg text-gray-600 px-4">OR</p>
              <button
                onClick={handleRecordVideo}
                className="border-2 border-gray-300 hover:border-[var(--pink-500)] text-[var(--pink-500)] font-bold py-3 px-6 rounded-full transition-colors duration-200 shadow-lg hover:shadow-xl"
              >
                Record Video
              </button>
            </div>

            <p className="text-sm text-gray-500 mt-10">
              {selectedFile
                ? selectedFile.name +
                  " (" +
                  (selectedFile.size / (1024 * 1024)).toFixed(2) +
                  " MB)"
                : "Supported formats: MP4, AVI, MOV, WMV, and more"}
            </p>
          </div>
          {selectedFile && (
            <div className="mt-4 relative bg-black border-2 border-gray-300 rounded-lg px-6">
              <div className="flex flex-col items-center">
                <video
                  controls
                  className="max-w-auto max-h-96 rounded-lg shadow-md"
                  src={URL.createObjectURL(selectedFile)}
                >
                  Your browser does not support the video tag.
                </video>
              </div>
              <div
                onClick={handleClearVideo}
                className="absolute mt-1 mr-1 top-0 right-0"
              >
                <CircleX className="w-6 h-6 text-white bg-[var(--pink-500)] hover:bg-[var(--pink-700)] rounded-full p-1" />
              </div>
            </div>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={handleVideoUpload}
        />
      </div>
    </>
  );
}

export default VideoUpload;
