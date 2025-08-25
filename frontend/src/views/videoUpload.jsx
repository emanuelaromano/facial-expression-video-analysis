import { useDispatch } from "react-redux";
import { setBannerThunk } from "../redux/slices/videoSlice";
import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { CircleX, RotateCcw, CirclePlay, CircleStop } from "lucide-react";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import JSZip from "jszip";
import { API_URL } from "../../api";
import ExpressionStats from "../components/expressionStats";
import TranscriptAnalysis from "../components/transcriptAnalysis";

function VideoUpload() {
  const dispatch = useDispatch();
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const analysisAbortRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [recordVideo, setRecordVideo] = useState(false);
  const [recording, setRecording] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [mediaStream, setMediaStream] = useState(null);
  const [wasRecorded, setWasRecorded] = useState(false);
  const [transcriptAnalysis, setTranscriptAnalysis] = useState(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  const [expressionStats, setExpressionStats] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const videoIdRef = useRef(uuidv4());
  const videoId = videoIdRef.current;
  const [progress, setProgress] = useState({
    progress: 0,
    state: "initializing",
  });

  // Memoize the video URL to prevent recreation on every render
  const videoUrl = useMemo(() => {
    if (processedVideoUrl) return processedVideoUrl;
    if (selectedFile) return URL.createObjectURL(selectedFile);
    return null;
  }, [processedVideoUrl, selectedFile]);

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
    setExpressionStats(null);
    setProgress({ progress: 0, state: "initializing" });
    setAnalyzing(false);
    setCameraLoading(false);
    stopMediaStream();
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.load();
    }
    // Reset the video ID for a fresh analysis
    videoIdRef.current = uuidv4();
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

  const handleUploadVideo = () => {
    // If analysis is running, abort it first
    if (analyzing && analysisAbortRef.current) {
      analysisAbortRef.current.abort();
      setAnalyzing(false);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    fileInputRef.current.click();
  };

  const handleRecordVideo = useCallback(async () => {
    if (analyzing && analysisAbortRef.current) {
      analysisAbortRef.current.abort();
      setAnalyzing(false);
    }
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
  }, [dispatch, analyzing]);

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

  useEffect(() => {
    if (expressionStats) {
      console.log(expressionStats);
    }
  }, [expressionStats]);

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
    }
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

    // Create EventSource for streaming updates
    const eventSource = new EventSource(`${API_URL}/video/stream/${videoId}`);

    eventSource.onmessage = (event) => {
      console.log("Stream update:", event.data);
      const data = JSON.parse(event.data);
      // Add a small delay to make progress updates smoother
      setTimeout(
        () => setProgress({ progress: data.progress, state: data.state }),
        100,
      );
    };

    eventSource.onerror = (error) => {
      console.error("EventSource error:", error);
      eventSource.close();
    };

    try {
      const controller = new AbortController();
      analysisAbortRef.current = controller;

      const formData = new FormData();
      formData.append("uuid", videoId);
      formData.append("original_video", selectedFile);

      // Expect a ZIP back (binary)
      const response = await axios.post(`${API_URL}/video/analyze`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Accept: "application/zip, application/octet-stream",
        },
        responseType: "blob",
        signal: controller.signal,
      });

      if (response.status < 200 || response.status >= 300) {
        throw new Error(`Server returned ${response.status}`);
      }

      // Unzip the blob
      const zip = await JSZip.loadAsync(response.data);

      const analysisEntry = zip.file("spoken_content_analysis.txt");
      const videoEntry = zip.file("processed_video.mp4");
      const expressionStatsEntry = zip.file("expression_stats.json");

      if (!analysisEntry || !videoEntry || !expressionStatsEntry) {
        throw new Error("ZIP missing analysis or video or expression stats");
      }

      // Load analysis text
      const analysisText = await analysisEntry.async("text");
      setTranscriptAnalysis(analysisText);

      // Load processed video
      const videoBlob = await videoEntry.async("blob");
      const url = URL.createObjectURL(videoBlob);
      setProcessedVideoUrl(url);

      // Load expression stats
      const expressionStatsText = await expressionStatsEntry.async("text");
      const expressionStatsDict = JSON.parse(expressionStatsText);
      setExpressionStats(expressionStatsDict);

      dispatch(setBannerThunk("Analysis complete", "success"));
    } catch (err) {
      if (err?.code === "ERR_CANCELED") {
        console.log("Analysis canceled by user");
        dispatch(setBannerThunk("Analysis canceled", "info"));
      } else {
        console.error("Analysis failed:", err);
        dispatch(setBannerThunk("Analysis failed.", "error"));
      }
      // Close EventSource on error
      eventSource.close();
    } finally {
      setAnalyzing(false);
      analysisAbortRef.current = null;
      // Close the EventSource
      eventSource.close();
    }
  }, [selectedFile, videoId, dispatch]);

  useEffect(() => {
    if (mediaStream && videoRef.current) {
      videoRef.current.srcObject = mediaStream;
    }
  }, [mediaStream]);

  // Clean up the memoized video URL when selectedFile changes
  useEffect(() => {
    return () => {
      // This will run when selectedFile changes or component unmounts
      // The URL will be automatically cleaned up by the browser when the component unmounts
    };
  }, [selectedFile]);

  useEffect(() => () => stopMediaStream(), [stopMediaStream]);

  const handleInterruptAnalysis = () => {
    if (analysisAbortRef.current) {
      analysisAbortRef.current.abort();
    }
    // Note: EventSource will be closed in the finally block when analysis completes
  };

  // Cleanup URLs on unmount
  useEffect(() => {
    return () => {
      analysisAbortRef.current?.abort();
      // Clean up processed video URL
      if (processedVideoUrl) {
        URL.revokeObjectURL(processedVideoUrl);
      }
      // Clean up original video URL if it exists and is different
      if (selectedFile && !processedVideoUrl) {
        // We can't directly revoke the memoized URL here, but it will be cleaned up
        // when the component unmounts or when selectedFile changes
      }
    };
  }, [processedVideoUrl, selectedFile]);

  return (
    <div className="flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-8 text-gray-800">
        Upload Your Video
      </h1>
      <div className="flex flex-col justify-center items-center bg-white p-8 rounded-lg shadow-lg border-2 border-dashed border-gray-300 hover:border-[var(--pink-500)] transition-colors">
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
          <div className="flex w-150 flex-col justify-center items-center gap-4">
            <div className="mt-4 relative bg-black border-2 border-gray-300 flex justify-center items-center rounded-lg overflow-hidden">
              <video
                key={videoUrl}
                controls
                className="w-full h-fit shadow-md"
                src={videoUrl}
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
                  <RotateCcw className="w-8 h-8 text-white bg-orange-500 hover:bg-orange-700 rounded-lg p-1" />
                </div>
              )}
            </div>
            <div className="w-150 flex flex-col gap-4 justify-center">
              {expressionStats && (
                <ExpressionStats
                  expressionStats={expressionStats}
                  title="Expression Analysis"
                />
              )}
              {transcriptAnalysis && (
                <TranscriptAnalysis
                  transcriptAnalysis={transcriptAnalysis}
                  title="Transcript Analysis"
                />
              )}
              <div className="w-full my-4">
                {!processedVideoUrl &&
                !analyzing &&
                !transcriptAnalysis &&
                !expressionStats ? (
                  <button
                    onClick={handleRunAnalysis}
                    className="bg-white border-2 border-gray-300 hover:border-[var(--pink-500)] disabled:opacity-50 text-[var(--pink-500)] font-bold py-2 w-full rounded-full transition-colors duration-200 shadow-lg hover:shadow-xl z-10"
                  >
                    Run Analysis
                  </button>
                ) : analyzing ? (
                  <div>
                    <div className="flex flex-col gap-3 items-center mb-4 px-4">
                      <div className="flex items-center gap-3">
                        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-[var(--pink-500)]"></div>
                        <p className="text-sm font-medium text-[var(--pink-500)] capitalize">
                          {progress.state.replace(/_/g, " ")}
                        </p>
                      </div>
                      <div className="w-full flex flex-row gap-2 items-center">
                        <div className="w-full h-2 bg-gray-200 rounded-full">
                          <div
                            className="h-full bg-[var(--pink-500)] rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${progress.progress}%` }}
                          />
                        </div>
                        <p className="text-sm text-gray-500">
                          {progress.progress}%
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={handleInterruptAnalysis}
                      className="bg-white border-2 border-gray-300 hover:border-[var(--pink-500)] disabled:opacity-50 text-[var(--pink-500)] font-bold py-2 w-full rounded-full transition-colors duration-200 shadow-lg hover:shadow-xl z-10"
                    >
                      Interrupt
                    </button>
                  </div>
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
          </div>
        )}

        {recordVideo && (
          <div className="mt-4 relative mx-auto bg-black border-2 border-gray-300 rounded-lg overflow-hidden">
            {cameraLoading ? (
              <div
                className="w-150 h-fit bg-gray-800 flex items-center justify-center"
                style={{ minHeight: "300px" }}
              >
                <div className="text-white text-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mx-auto mb-2" />
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
                  className="w-full h-fit object-contain bg-gray-800"
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
            <div
              onClick={recording ? handleStopRecording : handleStartRecording}
              className="absolute top-12 right-2"
            >
              {recording ? (
                <CircleStop className="w-8 h-8 text-white bg-red-500 hover:bg-red-700 rounded-lg p-1" />
              ) : (
                <CirclePlay className="w-8 h-8 text-white bg-green-500 hover:bg-green-700 rounded-lg p-1" />
              )}
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
  );
}

export default VideoUpload;
