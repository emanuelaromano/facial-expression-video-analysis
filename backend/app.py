import os
import io
import cv2
import json
import math
import uuid as uuidlib
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
from openai import OpenAI

router = APIRouter(prefix="/video")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to your environment or .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Utils ----------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _sorted_frame_paths(frames_dir: Path) -> List[Path]:
    return sorted(frames_dir.glob("frame_*.png"))

def _video_meta(video_path: str) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, frame_count

def _draw_face_mesh(img: np.ndarray, landmarks_px: List[Tuple[int, int]]) -> None:
    # draw tiny circles; you could also draw predefined connections
    for (x, y) in landmarks_px:
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

# ---------- Facial analysis functions ----------

def split_video_into_frames(video_path: str, uuid: str) -> str:
    """Split video into PNG frames and return directory path."""
    try:
        fps, w, h, n = _video_meta(video_path)
        frames_dir = _ensure_dir(Path(tempfile.gettempdir()) / "video_pipeline" / uuid / "frames")
        # If frames already exist, avoid re-doing work.
        if any(frames_dir.iterdir()):
            return str(frames_dir)

        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_path = frames_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            idx += 1
        cap.release()
        if idx == 0:
            raise ValueError("No frames extracted; is the video readable?")
        return str(frames_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"split_video_into_frames failed: {e}")

def facial_expression_analysis(frame: str) -> str:
    """
    Use DeepFace to get dominant emotion.
    Returns a simple string like 'happy (0.93)' for type compatibility.
    """
    try:
        # DeepFace returns a list or dict depending on version; handle both
        analysis = DeepFace.analyze(img_path=frame, actions=["emotion"], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        dominant = analysis.get("dominant_emotion", "unknown")
        # Grab top score if available
        emotions = analysis.get("emotion", {})
        score = emotions.get(dominant, None)
        return f"{dominant} ({score:.2f})" if isinstance(score, (float, int)) else dominant
    except Exception:
        # Be robust—sometimes face not found, etc.
        return "unknown"

def face_mesh_analysis(frame: str) -> str:
    """
    Use MediaPipe FaceMesh. Returns a JSON string with {width, height, landmarks: [[x,y], ...]}
    Coordinates are pixel coords.
    """
    img = cv2.imread(frame)
    if img is None:
        return json.dumps({"width": 0, "height": 0, "landmarks": []})
    h, w = img.shape[:2]

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return json.dumps({"width": w, "height": h, "landmarks": []})
        lm = res.multi_face_landmarks[0]
        coords = []
        for p in lm.landmark:
            x = min(max(int(p.x * w), 0), w - 1)
            y = min(max(int(p.y * h), 0), h - 1)
            coords.append([x, y])
        return json.dumps({"width": w, "height": h, "landmarks": coords})

def overall_frame_analysis(original_frame: str, facial_expression: str, face_mesh: str) -> str:
    """
    Overlay the face mesh and emotion text on the frame.
    Returns processed frame path.
    """
    try:
        img = cv2.imread(original_frame)
        if img is None:
            return original_frame

        # draw mesh
        try:
            payload = json.loads(face_mesh)
            coords = payload.get("landmarks", [])
            _draw_face_mesh(img, [(int(x), int(y)) for x, y in coords])
        except Exception:
            pass

        # draw emotion text
        cv2.rectangle(img, (6, 6), (320, 46), (0, 0, 0), -1)
        cv2.putText(
            img,
            f"Emotion: {facial_expression}",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        processed_dir = _ensure_dir(Path(original_frame).parent.parent / "processed_frames")
        out_path = processed_dir / Path(original_frame).name
        cv2.imwrite(str(out_path), img)
        return str(out_path)
    except Exception as e:
        # On any failure, just pass original frame through
        return original_frame

def add_audio_to_video(processed_frames: List[str], audio_path: str, fps: float, out_path: Path) -> str:
    """
    Build an mp4 from frames and mux in the provided audio.
    """
    try:
        clip = ImageSequenceClip(processed_frames, fps=fps)
        if audio_path and Path(audio_path).exists():
            audio = AudioFileClip(audio_path)
            clip = clip.set_audio(audio)
        _ensure_dir(out_path.parent)
        clip.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(out_path.with_suffix(".temp_aac.m4a")),
            remove_temp=True,
            fps=fps,
            verbose=False,
            logger=None,
        )
        clip.close()
        if audio_path and Path(audio_path).exists():
            audio.close()
        return str(out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"add_audio_to_video failed: {e}")

# ---------- Transcript analysis functions ----------

def audio_transcription(video_path: str) -> str:
    """
    Transcribe with OpenAI STT.
    Prefers gpt-4o-transcribe if available; falls back to whisper-1.
    Docs: Speech-to-text & Transcriptions API. 
    """
    # Extract audio to a temp .wav for best results
    tmp_dir = _ensure_dir(Path(tempfile.gettempdir()) / "video_pipeline" / "audio")
    wav_path = tmp_dir / f"{uuidlib.uuid4().hex}.wav"

    try:
        # Extract with MoviePy
        vclip = VideoFileClip(video_path)
        if not vclip.audio:
            vclip.close()
            return ""
        vclip.audio.write_audiofile(str(wav_path), verbose=False, logger=None)
        vclip.close()

        # Try newer model first; fall back gracefully
        model_candidates = ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
        text = ""
        with open(wav_path, "rb") as f:
            last_err = None
            for model in model_candidates:
                try:
                    tx = client.audio.transcriptions.create(  # OpenAI API ref
                        model=model,
                        file=f,
                        response_format="text",
                    )
                    # .text usually present; when response_format="text", tx may already be a str
                    text = getattr(tx, "text", tx if isinstance(tx, str) else "")
                    if text:
                        break
                except Exception as e:
                    last_err = e
                    f.seek(0)  # rewind for the next attempt
            if not text and last_err:
                raise last_err
        return text
    finally:
        # Cleanup temp audio
        try:
            if wav_path.exists():
                wav_path.unlink()
        except Exception:
            pass

def analyze_transcription(transcription: str) -> str:
    """
    Use the Responses API to act as a career coach and critique the content.
    Docs: Responses API (text generation).
    Returns plain text with sections.
    """
    if not transcription.strip():
        return "No audio content detected."

    prompt = (
        "You are an expert interview/career coach. Review the following interview transcript "
        "and provide concise, actionable feedback.\n\n"
        "Requirements:\n"
        "- Keep it under 250 words.\n"
        "- Use sections: Summary, Strengths, Areas to Improve, Score (1–10).\n"
        "- Be specific and practical.\n\n"
        f"Transcript:\n{transcription}"
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",  # fast & capable for analysis
            input=prompt,
        )
        # Prefer the convenience field if available; otherwise reconstruct
        text = getattr(resp, "output_text", None)
        if not text:
            # Fallback: stitch from content parts
            try:
                parts = []
                for block in getattr(resp, "output", []):
                    for piece in getattr(block, "content", []):
                        if piece.type == "output_text" or piece.type == "text":
                            parts.append(piece.text)
                text = "\n".join(parts).strip()
            except Exception:
                text = str(resp)
        return text.strip()
    except Exception as e:
        return f"LLM analysis failed: {e}"

# ---------- Video processing orchestration ----------

@router.post("/analyze")
def analyze_video(uuid: str, original_video: str) -> JSONResponse:
    """
    Orchestrates:
      1) Split into frames
      2) Per-frame: DeepFace emotion + MediaPipe face mesh + overlay
      3) Extract original audio and rebuild processed video with audio
      4) Transcribe audio (OpenAI) and run 'career coach' analysis
    Returns JSON with output paths and analysis text.
    """
    if not Path(original_video).exists():
        raise HTTPException(status_code=400, detail=f"original_video not found: {original_video}")

    # Create working dirs
    uuid = uuid or uuidlib.uuid4().hex
    work_root = _ensure_dir(Path(tempfile.gettempdir()) / "video_pipeline" / uuid)
    processed_frames_dir = _ensure_dir(work_root / "processed_frames")

    # 1) Frame extraction
    frames_dir = Path(split_video_into_frames(original_video, uuid))
    frames = _sorted_frame_paths(frames_dir)
    if not frames:
        raise HTTPException(status_code=500, detail="No frames to process.")

    # Get video meta
    fps, vw, vh, _ = _video_meta(original_video)

    # 2) Per-frame analysis + overlay
    processed_paths: List[str] = []
    for f in frames:
        emotion = facial_expression_analysis(str(f))
        mesh_json = face_mesh_analysis(str(f))
        out = overall_frame_analysis(str(f), emotion, mesh_json)
        processed_paths.append(out)

    # 3) Extract audio and rebuild video with audio
    # Extract audio from the original video
    audio_tmp = work_root / "orig_audio.m4a"
    try:
        vclip = VideoFileClip(original_video)
        if vclip.audio:
            vclip.audio.write_audiofile(str(audio_tmp), verbose=False, logger=None)
        vclip.close()
    except Exception as e:
        # Continue without audio
        audio_tmp = Path("")

    output_video_path = work_root / "processed_video.mp4"
    final_video = add_audio_to_video(processed_paths, str(audio_tmp) if audio_tmp else "", fps, output_video_path)

    # 4) Transcribe and analyze
    transcript_text = audio_transcription(original_video)
    analysis_text = analyze_transcription(transcript_text)

    # Save transcript + analysis to files for convenience
    transcript_path = work_root / "transcript.txt"
    analysis_path = work_root / "analysis.txt"
    transcript_path.write_text(transcript_text or "", encoding="utf-8")
    analysis_path.write_text(analysis_text or "", encoding="utf-8")

    # Best-effort cleanup of temp audio
    try:
        if audio_tmp and audio_tmp.exists():
            audio_tmp.unlink()
    except Exception:
        pass

    payload = {
        "uuid": uuid,
        "processed_video": str(final_video),
        "transcript": str(transcript_path),
        "analysis_text_file": str(analysis_path),
        "analysis_preview": analysis_text[:400] + ("..." if len(analysis_text) > 400 else ""),
    }
    return JSONResponse(content=payload)
