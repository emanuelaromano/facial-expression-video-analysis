/* eslint-disable no-unused-vars */
import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { setBannerThunk } from "../redux/slices/videoSlice";
import { useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import { API_URL } from "../../api";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import JSZip from "jszip";
import { CircleX, CirclePlay } from "lucide-react";

const VideoUploadContent = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const questionAllowedSeconds = 5;

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const analysisAbortRef = useRef(null);
  const questionTimeoutRef = useRef(null);
  const questionIntervalRef = useRef(null);
  const videoIdRef = useRef(uuidv4());
  const videoId = videoIdRef.current;

  const [scenario, setScenario] = useState("");
  const [currentSectionIndex, setCurrentSectionIndex] = useState(0);

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
  const [question, setQuestion] = useState(null);
  const [seconds, setSeconds] = useState(5);
  const [generatingQuestion, setGeneratingQuestion] = useState(false);
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
    setCurrentSectionIndex(currentSectionIndex - 1);
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
  }, [stopMediaStream, currentSectionIndex]);

  const handleVideoUpload = useCallback(
    (e) => {
      // If analysis is running, abort and cancel on server
      if (analyzing && analysisAbortRef.current) {
        analysisAbortRef.current.abort();
        // Fire-and-forget server-side cancel
        fetch(`${API_URL}/video/cancel/${videoId}`, { method: "POST" }).catch(
          () => {},
        );
        setAnalyzing(false);
      }
      resetVideoState();
      const file = e.target.files[0];
      if (file) {
        dispatch(setBannerThunk("Video uploaded successfully", "success"));
        setSelectedFile(file);
        setCurrentSectionIndex(currentSectionIndex + 1);
        setWasRecorded(false);
      } else {
        dispatch(setBannerThunk("No video file selected", "error"));
      }
    },
    [resetVideoState, dispatch, analyzing, videoId, currentSectionIndex],
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
    // If analysis is running, abort it first
    if (analyzing && analysisAbortRef.current) {
      analysisAbortRef.current.abort();
      // Fire-and-forget server-side cancel
      fetch(`${API_URL}/video/cancel/${videoId}`, { method: "POST" }).catch(
        () => {},
      );
      setAnalyzing(false);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    resetVideoState();
    setRecordVideo(true);
    setCameraLoading(true);
    setCurrentSectionIndex(2);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { min: 1280, ideal: 1920, max: 3840 },
          height: { min: 720, ideal: 1080, max: 2160 },
          frameRate: { ideal: 30, max: 30 },
          facingMode: "user",
        },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 48000,
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
  }, [dispatch, analyzing, resetVideoState, videoId]);

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

    const preferredTypes = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
    ];
    let mimeType =
      preferredTypes.find((t) => MediaRecorder.isTypeSupported(t)) || "";

    let recorder;
    try {
      recorder = new MediaRecorder(mediaStream, {
        mimeType,
        videoBitsPerSecond: 6_000_000,
        audioBitsPerSecond: 128_000,
      });
    } catch {
      recorder = new MediaRecorder(mediaStream);
    }

    mediaRecorderRef.current = recorder;
    const chunks = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, {
        type: recorder.mimeType || "video/webm",
      });
      const fileExt = (recorder.mimeType || "").includes("mp4")
        ? "mp4"
        : "webm";
      const file = new File([blob], `recorded-video.${fileExt}`, {
        type: blob.type,
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
      const data = JSON.parse(event.data);
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

      // Get signed upload URL
      const uploadResponse = await axios.get(`${API_URL}/video/upload`, {
        params: { uuid: videoId, filename: selectedFile.name },
        signal: controller.signal,
      });

      const { url: uploadUrl, gcs_path, headers } = uploadResponse.data;

      if (!uploadUrl || !gcs_path) {
        dispatch(setBannerThunk("Error uploading video", "error"));
        throw new Error("Upload URL or GCS path not found");
      }

      console.log("uploadUrl", uploadUrl);
      console.log("gcs_path", gcs_path);

      // Upload file directly to GCS
      const uploadResult = await fetch(uploadUrl, {
        method: "PUT",
        headers: headers,
        body: selectedFile,
        signal: controller.signal,
      });

      if (!uploadResult.ok) {
        dispatch(setBannerThunk("Video upload failed", "error"));
        throw new Error(`GCS upload failed: ${uploadResult.status}`);
      }

      // Start processing with GCS path (no file upload)
      const formData = new FormData();
      formData.append("uuid", videoId);
      formData.append("gcs_path", gcs_path);
      formData.append("scenario_description", scenario);

      const response = await axios.post(`${API_URL}/video/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
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
  }, [selectedFile, videoId, dispatch, scenario]);

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
    // Also tell backend to stop work for this uuid
    fetch(`${API_URL}/video/cancel/${videoId}`, { method: "POST" }).catch(
      () => {},
    );
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

  const handleGenerateQuestion = async () => {
    setGeneratingQuestion(true);
    if (questionTimeoutRef.current) {
      clearTimeout(questionTimeoutRef.current);
    }
    if (questionIntervalRef.current) {
      clearInterval(questionIntervalRef.current);
    }
    const formData = new FormData();
    formData.append("scenario_description", scenario);
    try {
      const response = await axios.post(`${API_URL}/video/question`, formData);
      if (
        response.status < 200 ||
        response.status >= 300 ||
        !response.data.question
      ) {
        throw new Error(`Server returned ${response.status}`);
      }
      const question = response.data.question;
      setQuestion(question);
      setSeconds(questionAllowedSeconds);
    } catch (err) {
      console.error("Failed to generate question:", err);
      dispatch(setBannerThunk("Failed to generate question", "error"));
      return;
    }
    questionIntervalRef.current = setInterval(() => {
      setSeconds((prev) => prev - 1);
    }, 1000);
    questionTimeoutRef.current = setTimeout(() => {
      setQuestion(null);
      if (questionIntervalRef.current) {
        clearInterval(questionIntervalRef.current);
      }
      handleStartRecording();
    }, questionAllowedSeconds * 1000);
    setGeneratingQuestion(false);
  };

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (questionTimeoutRef.current) {
        clearTimeout(questionTimeoutRef.current);
      }
      if (questionIntervalRef.current) {
        clearInterval(questionIntervalRef.current);
      }
    };
  }, []);

  return (
    <>
      {currentSectionIndex < 2 && (
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
          <div
            className="absolute top-6 left-6 cursor-pointer underline text-sm hover:translate-y-[-1px] hover:translate-x-[-1px] transition-all duration-300 ease-in-out"
            onClick={() => {
              if (currentSectionIndex === 0) {
                navigate("/");
              } else {
                setCurrentSectionIndex(currentSectionIndex - 1);
              }
            }}
          >
            {currentSectionIndex > 0 && <span>Back</span>}
            {currentSectionIndex === 0 && <span>Home</span>}
          </div>
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
                        setBannerThunk(
                          "Scenario description is required",
                          "error",
                        ),
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
              className={`flex w-[70%] absolute top-[50%] transition-all duration-500 ease-in-out flex-col justify-center items-center gap-2
              ${currentSectionIndex === 1 ? "left-[50%] translate-x-[-50%] translate-y-[-50%]" : currentSectionIndex < 1 ? "left-[150%] translate-x-[-50%] translate-y-[-50%]" : "left-[-150%] translate-x-[-50%] translate-y-[-50%]"}`}
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
                <button
                  onClick={handleRecordVideo}
                  className="primary-button w-full"
                >
                  Record Video
                </button>
                <p className="text-sm text-gray-600">
                  or{" "}
                  <button
                    onClick={handleUploadVideo}
                    className="cursor-pointer hover:translate-y-[-0.5px] hover:translate-x-[-0.5px] transition-all duration-300 ease-in-out"
                  >
                    <span className="underline font-bold">upload</span>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="video/*"
                      className="hidden"
                      onChange={handleVideoUpload}
                    />
                  </button>{" "}
                  a video
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
      {selectedFile && currentSectionIndex === 2 && (
        <div
          className={`flex h-[80%] absolute top-[50%] flex-col justify-center items-center gap-4 transition-all duration-500 ease-in-out
          ${currentSectionIndex === 2 ? "left-[50%] translate-x-[-50%] translate-y-[-50%]" : "left-[150%] translate-x-[-50%] translate-y-[-50%]"}`}
        >
          <div className="h-full w-full relative bg-black border-gray-300 flex justify-center items-center rounded-[0.25rem] overflow-hidden">
            <video
              key={videoUrl}
              controls
              className="h-full w-full object-contain"
              src={videoUrl}
            />
            {!processedVideoUrl && !analyzing && (
              <div className="absolute top-2 right-2 flex flex-col gap-2">
                <div className="relative flex group">
                  <CircleX
                    onClick={resetVideoState}
                    className="w-8 h-8 cursor-pointer font-mono text-white bg-[var(--pink-500)] hover:bg-[var(--pink-700)] rounded-[0.25rem] p-1"
                  />
                  <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100 whitespace-nowrap">
                    Close
                  </p>
                </div>

                {wasRecorded && (
                  <div className="relative flex group">
                    <RotateCcw
                      onClick={handleRecordVideo}
                      className="w-8 h-8 text-white bg-[var(--pink-500)] hover:bg-[var(--pink-700)] rounded-[0.25rem] p-1"
                    />
                    <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100 whitespace-nowrap">
                      Record Again
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
          {analyzing && (
            <button
              onClick={handleInterruptAnalysis}
              className="secondary-button w-full"
            >
              Interrupt
            </button>
          )}
          {!analyzing && (
            <button
              onClick={handleRunAnalysis}
              className="secondary-button w-full"
            >
              Run Analysis
            </button>
          )}
        </div>
      )}
      {!selectedFile && currentSectionIndex === 2 && recordVideo && (
        <div
          className={`flex h-[80%] absolute rounded-[0.25rem] top-[50%] flex-col justify-center items-center gap-4 transition-all duration-500 ease-in-out
                ${currentSectionIndex === 2 ? "left-[50%] translate-x-[-50%] translate-y-[-50%]" : "left-[150%] translate-x-[-50%] translate-y-[-50%]"}`}
        >
          {cameraLoading ? (
            <div className="h-[80%] bg-gray-800 flex items-center justify-center rounded-[0.25rem]">
              <div className="text-white text-center ">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mx-auto mb-2" />
                <p>Accessing camera...</p>
              </div>
            </div>
          ) : (
            <div className="relative h-full">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full object-contain bg-gray-800 rounded-[0.25rem]"
                style={{ minHeight: "300px", transform: "scaleX(-1)" }}
              />
              {recording && (
                <div className="absolute top-4 left-4 bg-red-500 text-white px-3 py-1 rounded-full flex items-center gap-2">
                  <div className="w-3 h-3 bg-white rounded-full animate-pulse" />{" "}
                  REC
                </div>
              )}
              {question && !recording && (
                <div className="absolute justify-between items-center flex gap-2 bg-black/80 text-white px-3 py-2 rounded-[0.25rem] left-4 right-4 bottom-16 text-left">
                  <p className="text-sm text-left">{question}</p>
                  <p className="text-sm border-l-2 border-white pl-2 min-w-20 self-stretch flex items-center justify-center">
                    {seconds}s
                  </p>
                </div>
              )}
            </div>
          )}
          <div className="absolute top-2 right-2 flex flex-col gap-2">
            <div className="relative flex group">
              <CircleX
                onClick={resetVideoState}
                className="w-8 h-8 text-white bg-[var(--pink-500)] hover:bg-[var(--pink-700)] rounded-[0.25rem] p-1"
              />
              <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100">
                Close
              </p>
            </div>
            {recording ? (
              <div className="relative flex group">
                <CircleStop
                  onClick={handleStopRecording}
                  className="w-8 h-8 text-white bg-red-500 hover:bg-red-700 rounded-[0.25rem] p-1"
                />
                <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100 whitespace-nowrap">
                  Stop
                </p>
              </div>
            ) : (
              <div className="relative flex group">
                <CirclePlay
                  onClick={handleStartRecording}
                  className="w-8 h-8 text-white bg-green-500 hover:bg-green-700 rounded-[0.25rem] p-1"
                />
                <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100 whitespace-nowrap">
                  Record
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default VideoUploadContent;
