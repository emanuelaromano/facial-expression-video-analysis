/* eslint-disable no-unused-vars */
import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { setBannerThunk } from "../redux/slices/videoSlice";
import { useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import JSZip from "jszip";
import {
  CircleX,
  CirclePlay,
  CircleStop,
  RotateCcw,
  ChevronLeft,
  X,
} from "lucide-react";
import ExpressionStats from "../components/expressionStats";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const VideoUpload = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const analysisAbortRef = useRef(null);
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
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [transcript, setTranscript] = useState(null);
  const [generatingTranscript, setGeneratingTranscript] = useState(false);
  const [progress, setProgress] = useState({
    percent: 0,
    state: "initializing",
  });
  const [transcriptAnalysis, setTranscriptAnalysis] = useState(null);
  const [expressionStats, setExpressionStats] = useState(null);

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
    if (analysisAbortRef.current) {
      // If analysis is running, abort it first
      analysisAbortRef.current.abort();
      fetch(`/api/video/cancel/${videoId}`, { method: "POST" }).catch(() => {});
    }

    stopMediaStream();
    setSelectedFile(null);
    setWasRecorded(false);
    setTranscript(null);
    setGeneratingTranscript(false);
    setRecordVideo(false);
    setRecording(false);
    setProcessedVideoUrl(null);
    setTranscriptAnalysis(null);
    setExpressionStats(null);
    setProgress({ percent: 0, state: "initializing" });
    setAnalyzing(false);
    setCameraLoading(false);
    setCurrentSectionIndex(1);

    // Reset video element
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.load();
    }

    // Reset the video ID for a fresh analysis
    videoIdRef.current = uuidv4();
  }, [stopMediaStream, videoId, currentSectionIndex]);

  const handleVideoUpload = useCallback(
    (e) => {
      // If analysis is running, abort and cancel on server
      if (analyzing && analysisAbortRef.current) {
        analysisAbortRef.current.abort();
        // Fire-and-forget server-side cancel
        fetch(`/api/video/cancel/${videoId}`, { method: "POST" }).catch(
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

        // // TODO: Remove this
        // setCurrentSectionIndex(3);
        // setExpressionStats({
        //   neutral: 0.4,
        //   happy: 0.2,
        //   sad: 0.2,
        //   angry: 0.1,
        //   surprised: 0.1,
        // });
        // setTranscriptAnalysis(
        //   `You are very clear in your speech and you are easy to understand.

        //   You are also very engaging and you are able to keep the audience's attention. The context of the interview is about the product you are selling and the benefits of the product. You are a very good salesperson and you are able to sell the product to the audience. You are also very confident and you are able to sell the product to the audience. You are also very persuasive and you are able to sell the product to the audience. You are also very charismatic and you are able to sell the product to the audience. You are also very likable and you are able to sell the product to the audience.
        //   You are also very trustworthy and you are able to sell the product to the audience. You are also very knowledgeable and you are able to sell the product to the audience.`,
        // );
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
      fetch(`/api/video/cancel/${videoId}`, { method: "POST" }).catch(() => {});
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
    const eventSource = new EventSource(`/api/video/stream/${videoId}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("SSE message received:", data);
      setTimeout(
        () => setProgress({ percent: data.progress, state: data.state }),
        100,
      );
    };

    eventSource.onerror = () => {
      console.error("EventSource error:", eventSource.error);
      eventSource.close();
    };

    eventSource.onerror = (error) => {
      console.error("EventSource error:", error);
      eventSource.close();
    };

    try {
      const controller = new AbortController();
      analysisAbortRef.current = controller;

      // Get signed upload URL
      const uploadResponse = await axios.get(`/api/video/upload`, {
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

      const response = await axios.post(`/api/video/analyze`, formData, {
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
      setCurrentSectionIndex(3);
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

  const handleGenerateTranscript = async () => {
    setGeneratingTranscript(true);
    setTranscript(null);
    const formData = new FormData();
    formData.append("scenario_description", scenario);

    try {
      const response = await axios.post(`/api/video/transcript`, formData);
      if (
        response.status < 200 ||
        response.status >= 300 ||
        !response.data.transcript
      ) {
        throw new Error(`Server returned ${response.status}`);
      }
      setTranscript(response.data.transcript || "No transcript generated");
    } catch (err) {
      console.error("Failed to generate transcript:", err);
      dispatch(setBannerThunk("Failed to generate transcript", "error"));
    } finally {
      setGeneratingTranscript(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center">
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
            width: "60%",
            transform: "translateX(-50%)",
            marginTop: "calc(max(20vmin, 3rem))",
            marginBottom: "calc(max(20vmin, 3rem))",
            gap: "2rem",
          }}
        >
          <X
            onClick={() => {
              navigate("/");
              resetVideoState();
            }}
            className="w-8 h-8 cursor-pointer absolute top-2 right-2 font-mono text-[var(--pink-500)] bg-white hover:text-[var(--pink-700)] p-1"
          />
          {currentSectionIndex > 0 && (
            <ChevronLeft
              onClick={() => {
                setCurrentSectionIndex(currentSectionIndex - 1);
              }}
              className="w-8 h-8 cursor-pointer absolute top-2 left-2 font-mono text-[var(--pink-500)] bg-white hover:text-[var(--pink-700)] p-1"
            />
          )}
          <div className="flex flex-col w-full p-20">
            <div
              className={`flex w-[80%] absolute top-[50%] flex-col justify-center items-center gap-5 transition-all duration-500 ease-in-out 
              ${currentSectionIndex === 0 ? "left-[50%] translate-x-[-50%] translate-y-[-50%]" : "left-[-50%] translate-x-[-50%] translate-y-[-50%]"}`}
            >
              <div className="text-sm text-gray-600">
                Enter the scenario you want to practice for.
              </div>
              <div className="flex flex-col w-full gap-2 justify-center items-center">
                <div className="relative w-full pb-5">
                  <textarea
                    className="primary-textbox w-full h-[10rem]"
                    type="text"
                    placeholder="Scenario"
                    value={scenario}
                    maxLength={500}
                    onChange={(e) => setScenario(e.target.value)}
                  />
                  <p className="text-[0.65rem] text-gray-600 absolute bottom-0.5 right-4">
                    {scenario.length} / 500
                  </p>
                </div>

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
              className={`flex w-[80%] absolute top-[50%] transition-all duration-500 ease-in-out flex-col justify-center items-center gap-2
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
          className={`flex w-[65%] max-h-[calc(80vh)] absolute top-[50%] flex-col justify-center items-center gap-4 transition-all duration-500 ease-in-out
          ${currentSectionIndex === 2 ? "left-[50%] translate-x-[-50%] translate-y-[-50%]" : "left-[150%] translate-x-[-50%] translate-y-[-50%]"}`}
        >
          <div className="w-full relative bg-transparent border-gray-300 flex justify-center items-center rounded-[0.25rem] max-h-[80vh]">
            <video
              key={videoUrl}
              controls
              className="max-h-[80vh] max-w-full object-contain rounded-[0.25rem]"
              src={videoUrl}
            />
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

              {!analyzing && wasRecorded && (
                <div className="relative flex group">
                  <RotateCcw
                    onClick={handleRecordVideo}
                    className="w-8 h-8 cursor-pointer text-white bg-orange-500 hover:bg-orange-700 rounded-[0.25rem] p-1"
                  />
                  <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100 whitespace-nowrap">
                    Record Again
                  </p>
                </div>
              )}
            </div>
          </div>
          {analyzing && (
            <div className="relative w-full">
              <button className="secondary-button w-full disabled cursor-not-allowed">
                {progress.state.slice(0, 1).toUpperCase() +
                  progress.state.slice(1)}{" "}
                ({progress.percent}%)
              </button>
              <div
                style={{
                  clipPath: `inset(0 ${100 - progress.percent}% 0 0)`,
                }}
                className={`absolute disabled top-0 text-center flex items-center justify-center text-white left-0 bottom-0 right-0 !bg-[var(--pink-700)]`}
              >
                {progress.state.slice(0, 1).toUpperCase() +
                  progress.state.slice(1)}{" "}
                ({progress.percent}%)
              </div>
            </div>
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
          className={`flex w-[65%] absolute aspect-video rounded-[0.25rem] top-[50%] flex-col justify-center items-center gap-4 transition-all duration-500 ease-in-out
                ${currentSectionIndex === 2 ? "left-[50%] translate-x-[-50%] translate-y-[-50%]" : "left-[150%] translate-x-[-50%] translate-y-[-50%]"}`}
        >
          {cameraLoading ? (
            <div className="relative w-full aspect-video bg-gray-800 flex items-center justify-center rounded-[0.25rem]">
              <div className="text-white text-center ">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mx-auto mb-2" />
                <p>Accessing camera...</p>
              </div>
            </div>
          ) : (
            <div className="relative aspect-video">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-contain bg-gray-800 rounded-[0.25rem]"
                style={{ transform: "scaleX(-1)" }}
              />
              {recording && (
                <div className="absolute top-4 left-4 bg-red-500 text-white px-3 py-1 rounded-full flex items-center gap-2">
                  <div className="w-3 h-3 bg-white rounded-full animate-pulse" />{" "}
                  REC
                </div>
              )}
              {transcript && (
                <div className="absolute justify-between items-center flex gap-2 bg-black/80 text-white px-3 py-2 rounded-[0.25rem] left-4 right-4 bottom-16 text-left z-10">
                  <p className="text-sm text-left">
                    Hello my name is Emanuela and I
                  </p>
                </div>
              )}
            </div>
          )}
          <div className="absolute aspect-video top-2 right-2 flex flex-col gap-2">
            <div className="relative flex group">
              <CircleX
                onClick={resetVideoState}
                className="w-8 h-8 cursor-pointer text-white bg-[var(--pink-500)] hover:bg-[var(--pink-700)] rounded-[0.25rem] p-1"
              />
              <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100">
                Close
              </p>
            </div>
            {recording ? (
              <div className="relative flex group">
                <CircleStop
                  onClick={handleStopRecording}
                  className="w-8 h-8 cursor-pointer text-white bg-red-500 hover:bg-red-700 rounded-[0.25rem] p-1"
                />
                <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100 whitespace-nowrap">
                  Stop
                </p>
              </div>
            ) : (
              <div className="relative flex group">
                <CirclePlay
                  onClick={handleStartRecording}
                  className="w-8 h-8 cursor-pointer text-white bg-green-500 hover:bg-green-700 rounded-[0.25rem] p-1"
                />
                <p className="pointer-events-none bg-black/50 transition-opacity duration-150 p-2 py-1 rounded-[0.25rem] text-white text-sm absolute top-1/2 -translate-y-1/2 right-full mr-2 opacity-0 group-hover:opacity-100 whitespace-nowrap">
                  Record
                </p>
              </div>
            )}
          </div>
          <button
            onClick={handleGenerateTranscript}
            disabled={generatingTranscript}
            className="secondary-button w-full"
            style={{
              opacity: generatingTranscript ? 0.8 : 1,
              backgroundColor: generatingTranscript ? "white" : undefined,
              cursor: generatingTranscript ? "not-allowed" : undefined,
              transform: generatingTranscript
                ? "translateY(0) translateX(0)"
                : undefined,
              color: generatingTranscript ? "var(--pink-700)" : undefined,
            }}
          >
            {generatingTranscript
              ? "Generating Transcript..."
              : "Generate Transcript"}
          </button>
        </div>
      )}
      {currentSectionIndex === 3 && (
        <div
          className="flex justify-center flex-col items-center relative w-[65vw] gap-12"
          style={{
            backgroundColor: "rgba(255, 255, 255)",
            borderRadius: "0.25em",
            boxShadow: "0 0 0.25em rgba(0, 0, 0, 0.25)",
            padding: "10vh",
            marginTop: "15vh",
            marginBottom: "15vh",
          }}
        >
          <X
            onClick={resetVideoState}
            className="w-8 h-8 cursor-pointer absolute top-2 right-2 font-mono text-[var(--pink-500)] bg-white hover:text-[var(--pink-700)] p-1"
          />
          <div className="flex flex-col w-full items-center justify-center">
            <div className="text-lg font-bold mb-8">Expression Stats</div>
            <div className="w-full flex flex-col items-center justify-center">
              <ExpressionStats
                expressionStats={expressionStats}
                setExpressionStats={setExpressionStats}
              />
            </div>
          </div>
          <div className="flex flex-col w-full items-center justify-center">
            <div className="text-lg font-bold mb-8">Transcript Analysis</div>
            <div className="text-sm text-gray-700 text-left prose prose-sm max-w-none [&>p]:mb-4 [&>p:last-child]:mb-0">
              {transcriptAnalysis && (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {transcriptAnalysis}
                </ReactMarkdown>
              )}
            </div>
          </div>
          <div className="flex flex-col w-full items-center justify-center">
            <div className="text-lg font-bold mb-8">Processed Video</div>
            <div className="text-sm text-gray-700 text-left">
              <video
                key={videoUrl}
                controls
                className="max-h-[80vh] max-w-full object-contain rounded-[0.25rem]"
                src={videoUrl}
              />
            </div>
          </div>
          <button
            className="primary-button w-full"
            onClick={() => resetVideoState()}
          >
            Reset
          </button>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
