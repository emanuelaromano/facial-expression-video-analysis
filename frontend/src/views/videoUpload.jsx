import { useDispatch } from "react-redux";
import { setBannerThunk } from "../redux/slices/videoSlice";
import { useRef, useState, useEffect, useCallback } from "react";
import { CircleX, RotateCcw } from "lucide-react";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import JSZip from "jszip";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function VideoUpload() {
  const dispatch = useDispatch();
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [recordVideo, setRecordVideo] = useState(false);
  const [recording, setRecording] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [mediaStream, setMediaStream] = useState(null);
  const [wasRecorded, setWasRecorded] = useState(false);
  const [transcriptAnalysis, setTranscriptAnalysis] = useState(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const videoIdRef = useRef(uuidv4());
  const videoId = videoIdRef.current;
  // eslint-disable-next-line no-unused-vars
  const [progress, setProgress] = useState(50);

  const stopMediaStream = useCallback(() => {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      setMediaStream(null);
    }
  }, [mediaStream]);

  const resetVideoState = useCallback(() => {
    setSelectedFile(null);
    setWasRecorded(false);
    setRecordVideo(false);
    setRecording(false);
    setProcessedVideoUrl(null);
    setTranscriptAnalysis(null);
    stopMediaStream();
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.load();
    }
  }, [stopMediaStream]);

  const handleVideoUpload = useCallback(
    (e) => {
      resetVideoState();
      const file = e.target.files[0];
      if (file) {
        setSelectedFile(file);
        setWasRecorded(false);
        dispatch(setBannerThunk("Video uploaded successfully", "success"));
      } else {
        dispatch(setBannerThunk("No video file selected", "error"));
      }
    },
    [resetVideoState, dispatch],
  );

  const handleUploadVideo = () => fileInputRef.current.click();

  const handleRecordVideo = useCallback(async () => {
    setSelectedFile(null);
    setRecordVideo(true);
    setCameraLoading(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      });

      setMediaStream(stream);
      setRecordVideo(true);
    } catch (err) {
      console.error("Camera access failed:", err);
      dispatch(setBannerThunk("Unable to access camera", "error"));
    } finally {
      setCameraLoading(false);
    }
  }, [dispatch]);

  const handleStartRecording = () => {
    if (!mediaStream) return;
    // Add a quick audio noise notification
    const AudioContextClass =
      window.AudioContext || window["webkitAudioContext"];
    const audioContext = new AudioContextClass();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(
      0.01,
      audioContext.currentTime + 0.1,
    );

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.1);

    const options = { mimeType: "video/webm;codecs=vp8,opus" };
    let recorder;

    try {
      recorder = new MediaRecorder(mediaStream, options);
    } catch {
      try {
        recorder = new MediaRecorder(mediaStream, { mimeType: "video/webm" });
      } catch {
        recorder = new MediaRecorder(mediaStream);
      }
    }

    mediaRecorderRef.current = recorder;
    const chunks = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      const file = new File([blob], "recorded-video.webm", {
        type: "video/webm",
      });
      setSelectedFile(file);
      setWasRecorded(true);
      setRecording(false);
      setRecordVideo(false);
      stopMediaStream();
      dispatch(setBannerThunk("Video recorded successfully", "success"));
    };

    recorder.start(1000);
    setRecording(true);
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
    }
  };

  const handleCancelAnalysis = () => {
    console.log("Cancel analysis");
  };

  const handleRunAnalysis = useCallback(async () => {
    if (!selectedFile) {
      dispatch(
        setBannerThunk("Please upload or record a video first", "error"),
      );
      return;
    }

    setAnalyzing(true);
    setProcessedVideoUrl(null);
    setTranscriptAnalysis(null);

    try {
      const formData = new FormData();
      formData.append("uuid", videoId);
      formData.append("original_video", selectedFile);

      // Expect a ZIP back (binary)
      const response = await axios.post(`/video/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: "blob",
      });

      if (response.status < 200 || response.status >= 300) {
        throw new Error(`Server returned ${response.status}`);
      }

      // Unzip the blob
      const zip = await JSZip.loadAsync(response.data);

      const analysisEntry = zip.file("spoken_content_analysis.txt");
      const videoEntry = zip.file(`${videoId}.mp4`);

      if (!analysisEntry || !videoEntry) {
        throw new Error("ZIP missing analysis or video");
      }

      // Load analysis text
      const analysisText = await analysisEntry.async("text");
      setTranscriptAnalysis(analysisText);

      // Load processed video
      const videoBlob = await videoEntry.async("blob");
      const url = URL.createObjectURL(videoBlob);
      setProcessedVideoUrl(url);

      dispatch(setBannerThunk("Analysis complete", "success"));
    } catch (err) {
      console.error("Analysis failed:", err);
      dispatch(setBannerThunk("Analysis failed.", "error"));
    } finally {
      setAnalyzing(false);
    }
  }, [selectedFile, videoId, dispatch]);

  useEffect(() => {
    if (mediaStream && videoRef.current) {
      videoRef.current.srcObject = mediaStream;
    }
  }, [mediaStream]);

  useEffect(() => () => stopMediaStream(), [stopMediaStream]);

  // Cleanup processed video URL on unmount
  useEffect(() => {
    return () => {
      if (processedVideoUrl) {
        URL.revokeObjectURL(processedVideoUrl);
      }
    };
  }, [processedVideoUrl]);

  return (
    <div className="flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-8 text-gray-800">
        Upload Your Video
      </h1>
      <div className="bg-white p-8 rounded-lg shadow-lg border-2 border-dashed border-gray-300 hover:border-[var(--pink-500)] transition-colors">
        <div className="text-center w-160 mx-auto">
          <div className="mb-4">
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
          </div>

          <p className="text-lg text-gray-600 mb-4">
            Click the button below to select your video file
          </p>
          <div className="flex flex-row gap-2 px-10 justify-center items-center">
            <button
              onClick={handleUploadVideo}
              className="bg-[var(--pink-500)] hover:bg-[var(--pink-700)] text-white font-bold py-3 px-6 rounded-full transition-colors duration-200 shadow-lg hover:shadow-xl"
            >
              Upload Video
            </button>
            <p className="text-lg text-gray-600 px-4">OR</p>
            <button
              disabled={wasRecorded || recordVideo}
              onClick={handleRecordVideo}
              className={`disabled:opacity-50 border-2 border-gray-300 text-[var(--pink-500)] font-bold py-3 px-6 rounded-full transition-colors duration-200 shadow-lg ${!(wasRecorded || recordVideo) ? "hover:border-[var(--pink-500)] hover:shadow-xl" : ""}`}
            >
              Record Video
            </button>
          </div>

          <p className="text-sm text-gray-500 mt-10">
            {selectedFile
              ? `${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)`
              : "Supported formats: MP4, AVI, MOV, WMV, and more"}
          </p>
        </div>

        {selectedFile && (
          <div className="flex flex-col items-center gap-4">
            <div className="mt-4 relative bg-black border-2 border-gray-300 flex justify-center items-center rounded-lg px-6">
              <video
                controls
                className="min-w-[170] max-h-96 shadow-md"
                src={processedVideoUrl ?? URL.createObjectURL(selectedFile)}
              />
              {!processedVideoUrl && !analyzing && (
                <div
                  onClick={resetVideoState}
                  className="absolute mt-2 mr-2 top-0 right-0"
                >
                  <CircleX className="w-8 h-8 text-white bg-[var(--pink-500)] hover:bg-[var(--pink-700)] rounded-lg p-1" />
                </div>
              )}
              {wasRecorded && !analyzing && !processedVideoUrl && (
                <div
                  onClick={handleRecordVideo}
                  className="absolute mt-2 mr-2 top-10 right-0"
                >
                  <RotateCcw className="w-8 h-8 text-white bg-[var(--pink-500)] hover:bg-[var(--pink-700)] rounded-lg p-1" />
                </div>
              )}
            </div>
            <div className="w-160 flex flex-col gap-4 justify-center mt-4 px-10">
              {analyzing && (
                <div className="flex flex-row gap-2 items-center">
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div
                      className="h-full bg-[var(--pink-500)] rounded-full"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-sm text-gray-500">{progress}%</p>
                </div>
              )}
              {transcriptAnalysis && (
                <div className="w-full flex flex-col justify-center ">
                  <div className="prose prose-pink max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {transcriptAnalysis}
                    </ReactMarkdown>
                  </div>
                </div>
              )}
              {!processedVideoUrl && !analyzing && !transcriptAnalysis ? (
                <button
                  onClick={handleRunAnalysis}
                  className="bg-white border-2 border-gray-300 hover:border-[var(--pink-500)] disabled:opacity-50 text-[var(--pink-500)] font-bold py-2 w-full rounded-full transition-colors duration-200 shadow-lg hover:shadow-xl z-10"
                >
                  Run Analysis
                </button>
              ) : analyzing ? (
                <button
                  onClick={handleCancelAnalysis}
                  className="bg-[var(--pink-500)] hover:bg-[var(--pink-700)] text-white font-bold py-3 px-6 rounded-full transition-colors duration-200 shadow-lg hover:shadow-xl"
                >
                  Cancel
                </button>
              ) : (
                <button
                  onClick={resetVideoState}
                  className="w-full bg-[var(--pink-500)] hover:bg-[var(--pink-700)] text-white font-bold py-3 rounded-full transition-colors duration-200 shadow-lg hover:shadow-xl"
                >
                  Reset
                </button>
              )}
            </div>
          </div>
        )}

        {recordVideo && (
          <div className="mt-4 relative bg-black border-2 border-gray-300 rounded-lg">
            {cameraLoading ? (
              <div
                className="w-full max-h-96 bg-gray-800 flex items-center justify-center"
                style={{ minHeight: "300px" }}
              >
                <div className="text-white text-center">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mx-auto mb-2" />
                  <p>Accessing camera...</p>
                </div>
              </div>
            ) : (
              <div className="relative">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full max-h-96 object-contain bg-gray-800"
                  style={{ minHeight: "300px" }}
                />
                {recording && (
                  <div className="absolute top-4 left-4 bg-red-500 text-white px-3 py-1 rounded-full flex items-center gap-2">
                    <div className="w-3 h-3 bg-white rounded-full animate-pulse" />{" "}
                    REC
                  </div>
                )}
              </div>
            )}
            <div
              onClick={resetVideoState}
              className="absolute mt-2 mr-2 top-0 right-0"
            >
              <CircleX className="w-8 h-8 text-white bg-[var(--pink-500)] hover:bg-[var(--pink-700)] rounded-lg p-1" />
            </div>
            <button
              onClick={recording ? handleStopRecording : handleStartRecording}
              className="absolute bottom-4 left-4 bg-[var(--pink-500)] hover:bg-[var(--pink-700)] text-white font-bold py-2 px-4 rounded-lg transition-colors duration-200 shadow-lg hover:shadow-xl z-10"
            >
              {recording ? "Stop" : "Start"}
            </button>
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
  );
}

export default VideoUpload;
