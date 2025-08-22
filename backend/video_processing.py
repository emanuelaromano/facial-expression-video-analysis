import os
import shutil
import cv2
import logging
import subprocess
from dotenv import load_dotenv
from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from http import HTTPStatus
from deepface import DeepFace
from transformers import AutoImageProcessor, AutoModelForImageClassification  # Hugging Face
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import mediapipe as mp
from threading import Lock
from typing import Optional

router = APIRouter(prefix="/video")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger("hireview")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

########################################################
# Video processing globals
########################################################

FACE_MESH_EVERY = 2
EMOTION_EVERY = 10
DETECTOR_BACKEND = "opencv"
NUM_WORKERS = 4 
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
mesh_lock = Lock()
emotion_processor = AutoImageProcessor.from_pretrained("mo-thecreator/vit-Facial-Expression-Recognition")
emotion_model = AutoModelForImageClassification.from_pretrained("mo-thecreator/vit-Facial-Expression-Recognition").eval()
EMOTION_ID2LABEL = emotion_model.config.id2label

########################################################
# API endpoints
########################################################

@router.post("/analyze", status_code=HTTPStatus.OK)
def analyze_video(
    background_tasks: BackgroundTasks,
    uuid: str = Form(...),
    original_video: UploadFile = File(...),
) -> FileResponse:
    temp_dir = f"temp/{uuid}"
    temp_og_video_path = os.path.join(temp_dir, "video.mp4")
    temp_frames_dir = os.path.join(temp_dir, "frames")
    temp_processed_dir = os.path.join(temp_dir, "processed")
    temp_audio_path = os.path.join(temp_dir, "audio.m4a")
    temp_rebuilt_video_path = os.path.join(temp_dir, "rebuilt_video.mp4")
    logger.info(f"Analyzing video {uuid}")

    try:
        # Create temp folders
        os.makedirs(temp_frames_dir, exist_ok=True)
        os.makedirs(temp_processed_dir, exist_ok=True)

        # Save video and extract info
        save_original_video(original_video, temp_og_video_path)
        fps = get_fps(temp_og_video_path)
        audio_path = extract_audio_ffmpeg(temp_og_video_path, temp_audio_path)
        if not audio_path:
            logger.warning("Audio extraction failed - video will be silent")
        num_frames = split_video_into_frames(temp_og_video_path, temp_frames_dir)

        expressions = {}

        # Analyze only needed frames
        needed_idxs = sorted(
            set(range(0, num_frames, FACE_MESH_EVERY)) |
            set(range(0, num_frames, EMOTION_EVERY))
        )

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(analyze_frame, os.path.join(temp_frames_dir, f"{i}.jpg"), i): i
                for i in needed_idxs
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                expressions[str(idx)] = result or {}
                logger.info(f"Analyzed frame {idx}/{num_frames}")

        # Rebuild frames using most recent analyzed indices
        # Exponential smoothing for emotion confidence to reduce flicker
        alpha = 0.4
        last_label = None
        last_conf = None

        for i in range(num_frames):
            mesh_idx = (i // FACE_MESH_EVERY) * FACE_MESH_EVERY
            emo_idx = (i // EMOTION_EVERY) * EMOTION_EVERY
            mesh_idx = min(mesh_idx, num_frames - 1)
            emo_idx = min(emo_idx,  num_frames - 1)

            mesh_expression = expressions.get(str(mesh_idx), {}) or {}
            emotion_expression = expressions.get(str(emo_idx), {}) or {}

            label = emotion_expression.get("emotion")
            conf = emotion_expression.get("confidence")

            if label is None:
                label = last_label
            if isinstance(conf, (int, float)) and isinstance(last_conf, (int, float)):
                conf = alpha * conf + (1 - alpha) * last_conf
            last_label = label
            last_conf = conf

            final_expression = {
                "face_mesh": mesh_expression.get("face_mesh"),
                "emotion": label,
                "confidence": conf,
            }
            rebuild_frame(i, temp_frames_dir, temp_processed_dir, final_expression)

        rebuilt_video_path = rebuild_video_ffmpeg(temp_processed_dir, audio_path, temp_rebuilt_video_path, fps)

        if not os.path.exists(rebuilt_video_path):
            logger.error("rebuilt video not found at %s", rebuilt_video_path)
            raise HTTPException(status_code=500, detail="Failed to build output video")

        # Schedule cleanup AFTER the response is sent
        background_tasks.add_task(shutil.rmtree, temp_dir, True)

        # Return the mp4 file
        return FileResponse(
            path=rebuilt_video_path,
            media_type="video/mp4",
            filename=f"{uuid}.mp4"
        )
    except Exception as e:
        logger.exception("unexpected error in analyze_video")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

########################################################
# Video processing functions
########################################################

def save_original_video(original_video: UploadFile, path: str) -> None:
    with open(path, "wb") as f:
        f.write(original_video.file.read())


def split_video_into_frames(video_path: str, frames_dir: str) -> int:
    video = cv2.VideoCapture(video_path)
    count = 0
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_path = os.path.join(frames_dir, f"{count}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
    finally:
        video.release()
    return count


def analyze_frame(frame_path: str, frame_index: int) -> dict | None:
    try:
        face_mesh, emotion, confidence = None, None, None
        if frame_index % FACE_MESH_EVERY == 0:
           face_mesh = face_mesh_analysis(frame_path, frame_index)
        if frame_index % EMOTION_EVERY == 0:
            emotion, confidence = emotion_analysis(frame_path, frame_index)
        return {
            "face_mesh": face_mesh,
            "emotion": emotion,
            "confidence": confidence
        }
    except Exception as e:
        logger.warning(f"Unexpected error analyzing frame {frame_index}: {e}")
        # Return empty dict instead of None to avoid KeyError in rebuild loop
        return {
            "face_mesh": None,
            "emotion": None,
            "confidence": None
        }


def emotion_analysis(frame_path: str, frame_index: int):
    bgr = cv2.imread(frame_path)
    if bgr is None:
        logger.warning(f"Frame {frame_index} could not be read for emotion analysis.")
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        inputs = emotion_processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        label = EMOTION_ID2LABEL.get(idx, str(idx)) if EMOTION_ID2LABEL else str(idx)
        conf = float(probs[idx].item())
        return label, conf
    except Exception as e:
        logger.warning(f"ViT FER analysis failed for frame {frame_index}: {e}")
        return None, None

def face_mesh_analysis(frame_path: str, frame_index: int):
    frame_bgr = cv2.imread(frame_path)
    if frame_bgr is None:
        logger.warning(f"Frame {frame_index} could not be read for face mesh.")
        return None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        with mesh_lock:
            results = face_mesh_model.process(rgb)
    except Exception as e:
        logger.warning(f"FaceMesh processing failed for frame {frame_index}: {e}")
        return None

    if not results or not results.multi_face_landmarks:
        return None

    faces = []
    for face_landmarks in results.multi_face_landmarks:
        points = [
            {
                "x": float(round(lm.x, 5)),
                "y": float(round(lm.y, 5)),
                "z": float(round(lm.z, 5)),
            }
            for lm in face_landmarks.landmark
        ]
        faces.append(points)
    return faces


def rebuild_frame(
    frame_index: int,
    temp_frames_dir: str,
    temp_processed_dir: str,
    expression: Optional[dict] = None
) -> None:
    logger.info(f"Rebuilding frame {frame_index}")
    frame_path = os.path.join(temp_frames_dir, f"{frame_index}.jpg")
    frame_image = cv2.imread(frame_path)
    if frame_image is None:
        return

    height, width = frame_image.shape[:2]

    faces = None
    if isinstance(expression, dict):
        faces = expression.get("face_mesh")

    if faces:
        for face in faces:
            # Each point: {"x": float, "y": float, "z": float}
            for pt in face:
                # Clamp just in case normalized coords slightly leave [0,1]
                x = int(max(0, min(1, float(pt.get("x", 0.0)))) * width)
                y = int(max(0, min(1, float(pt.get("y", 0.0)))) * height)
                cv2.circle(
                    frame_image,
                    (x, y),
                    1,
                    (0, 255, 0),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    label_lines: list[str] = []
    if isinstance(expression, dict):
        emotion = expression.get("emotion")
        confidence = expression.get("confidence")
        # Only draw the box when BOTH are present
        if emotion is not None and confidence is not None:
            try:
                conf_val = float(confidence)
            except (TypeError, ValueError):
                conf_val = None

            label_lines.append(str(emotion).upper())
            if conf_val is not None:
                label_lines.append(f"Conf: {conf_val:.2f}")

    if label_lines:
        padding = 8
        line_gap = 18
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Compute max line width
        max_text_w = 0
        for line in label_lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            if text_width > max_text_w:
                max_text_w = text_width

        box_w = int(max_text_w + 2 * padding)
        box_h = int(line_gap * len(label_lines) + 2 * padding)

        top_left = (10, 10)
        bottom_right = (10 + box_w, 10 + box_h)
        cv2.rectangle(
            frame_image, top_left, bottom_right, (30, 30, 30), thickness=-1
        )

        # Text lines
        y = 10 + padding + 12
        for line in label_lines:
            cv2.putText(
                frame_image,
                line,
                (10 + padding, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y += line_gap

    os.makedirs(temp_processed_dir, exist_ok=True)
    out_path = os.path.join(temp_processed_dir, f"{frame_index}.jpg")
    cv2.imwrite(out_path, frame_image)

def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cap.release()
    return fps if fps > 0 else 25.0

def extract_audio_ffmpeg(input_video: str, output_audio: str) -> Optional[str]:
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)

    # Check for audio stream
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0", input_video
    ]
    try:
        subprocess.run(probe_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        logger.warning(f"No audio stream found in: {input_video}")
        return None

    ext = os.path.splitext(output_audio)[1].lower()
    if ext == ".mp3":
        acodec = "libmp3lame"
    else:
        acodec = "aac"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-map", "0:a:0",
        "-vn",
        "-c:a", acodec,
        output_audio
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(output_audio) and os.path.getsize(output_audio) > 1024:  # at least 1KB
            logger.info(f"Audio extracted: {output_audio} ({os.path.getsize(output_audio)} bytes)")
            return output_audio
        return None
    except subprocess.CalledProcessError as e:
        return None

def rebuild_video_ffmpeg(frames_dir: str, audio_path: Optional[str], output_path: str, fps: float) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base = [
        "ffmpeg", "-y",
        "-start_number", "0",
        "-framerate", str(int(round(fps))),
        "-i", os.path.join(frames_dir, "%d.jpg"),
    ]

    if audio_path and os.path.exists(audio_path):
        cmd = base + [
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",  
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-r", str(int(round(fps))),
            "-c:a", "copy",                    
            "-shortest",
            output_path
        ]
    else:
        cmd = base + [
            "-map", "0:v:0",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-r", str(int(round(fps))),
            output_path
        ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Output video not created: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg video rebuild failed: {e.stderr.decode(errors='ignore')[:400]}")
        raise


#################################
# Transcript analysis functions
#################################

def extract_audio(video_path: str) -> str:
    # TODO: Extract the audio from the video and return the audio path
    # This function is kept for future transcript analysis features
    pass

def audio_transcription(video_path: str) -> str:
    # TODO: Transcribe the audio of the video using OpenAI Whisper
    # This function is kept for future transcript analysis features
    pass

def analyze_transcription(transcription: str) -> str:
    # TODO: Analyze the transcription using OpenAI and return the analysis.
    # The LLM should act as a career coach, giving feedback to the candidate based on the quality of the content.
    # This function is kept for future transcript analysis features
    pass

