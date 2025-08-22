import os
import shutil
import cv2
import logging
import subprocess
from dotenv import load_dotenv
from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from http import HTTPStatus
from transformers import AutoImageProcessor, AutoModelForImageClassification  # Hugging Face
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import mediapipe as mp
from threading import Lock
from typing import Optional
from openai import OpenAI
from zipfile import ZipFile, ZIP_DEFLATED
from fractions import Fraction
import json

router = APIRouter()
client = OpenAI()
load_dotenv()

# Logging setup
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

FACE_MESH_EVERY = int(os.getenv("FACE_MESH_EVERY", "2"))
EMOTION_EVERY   = int(os.getenv("EMOTION_EVERY", "10"))
NUM_WORKERS     = int(os.getenv("NUM_WORKERS", "4"))

# Building the face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
mesh_lock = Lock()

# Face detection for emotion analysis
mp_fd = mp.solutions.face_detection
face_det = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
fd_lock = Lock()

# Emotion analysis
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
    job_description: Optional[str] = Form("Data Engineer"),
) -> FileResponse:
    temp_dir = f"temp/{uuid}"
    temp_og_video_path = os.path.join(temp_dir, "video.mp4")
    temp_frames_dir = os.path.join(temp_dir, "frames")
    temp_processed_dir = os.path.join(temp_dir, "processed")
    temp_audio_path = os.path.join(temp_dir, "audio.m4a")
    temp_rebuilt_video_path = os.path.join(temp_dir, "rebuilt_video.mp4")
    bundle_path = os.path.join(temp_dir, f"{uuid}.zip")
    analysis_path = os.path.join(temp_dir, "spoken_content_analysis.txt")

    logger.info(f"Analyzing video {uuid}")

    try:
        # Create temp folders
        os.makedirs(temp_frames_dir, exist_ok=True)
        os.makedirs(temp_processed_dir, exist_ok=True)

        # Save video and extract info
        save_original_video(original_video, temp_og_video_path)
        fps_num, fps_den = get_fps_fraction(temp_og_video_path)
        audio_path = extract_audio_ffmpeg(temp_og_video_path, temp_audio_path)
        if not audio_path:
            logger.warning("Audio extraction failed - video will be silent")
        num_frames = split_video_into_frames(temp_og_video_path, temp_frames_dir)

        # Analyze only needed frames
        expressions = {}
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
        alpha = 0.4
        last_label = None
        last_conf = None

        for i in range(num_frames):
            mesh_idx = min((i // FACE_MESH_EVERY) * FACE_MESH_EVERY, num_frames - 1)
            emo_idx  = min((i // EMOTION_EVERY)  * EMOTION_EVERY,  num_frames - 1)

            mesh_expression = expressions.get(str(mesh_idx), {}) or {}
            emotion_expression = expressions.get(str(emo_idx), {}) or {}

            label = emotion_expression.get("emotion")
            conf  = emotion_expression.get("confidence")

            if label is None:
                label = last_label
            if isinstance(conf, (int, float)) and isinstance(last_conf, (int, float)):
                conf = alpha * conf + (1 - alpha) * last_conf

            last_label = label
            last_conf  = conf

            final_expression = {
                "face_mesh": mesh_expression.get("face_mesh"),
                "emotion": label,
                "confidence": conf,
            }
            rebuild_frame(i, temp_frames_dir, temp_processed_dir, num_frames, final_expression)

        # Rebuild video
        rebuilt_video_path = rebuild_video_ffmpeg(
            temp_processed_dir,
            audio_path,
            temp_rebuilt_video_path,
            fps_num,
            fps_den,
        )
        if not os.path.exists(rebuilt_video_path):
            logger.error("rebuilt video not found at %s", rebuilt_video_path)
            raise HTTPException(status_code=500, detail="Failed to build output video")

        # Spoken-content analysis (guard for missing audio)
        analyze_spoken_content(audio_path, job_description, analysis_path)

        # Bundle video + analysis into a ZIP
        with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
            zf.write(rebuilt_video_path, arcname=f"{uuid}.mp4")
            zf.write(analysis_path, arcname="spoken_content_analysis.txt")

        # Cleanup AFTER response is sent
        background_tasks.add_task(delayed_rmtree, temp_dir, 30)

        return FileResponse(
            path=bundle_path,
            media_type="application/zip",
            filename=f"{uuid}.zip",
        )

    except Exception as e:
        logger.exception("unexpected error in analyze_video")
        # Best effort: if a partially created bundle exists, return a 500 instead of streaming it
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


def analyze_frame(frame_path: str, frame_index: int) -> dict:
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
        return {
            "face_mesh": None,
            "emotion": None,
            "confidence": None
        }

def detect_face_bbox(bgr):
    # Implement face detection using MediaPipe to improve emotion analysis
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with fd_lock:
        res = face_det.process(rgb)
    if not res or not res.detections: return None
    d = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width*d.location_data.relative_bounding_box.height)
    rb = d.location_data.relative_bounding_box
    x1 = max(0, int(rb.xmin * width)); y1 = max(0, int(rb.ymin * height))
    x2 = min(width, int((rb.xmin + rb.width) * width)); y2 = min(height, int((rb.ymin + rb.height) * height))
    mx = int(0.1 * (x2 - x1)); my = int(0.1 * (y2 - y1))
    return max(0, x1-mx), max(0, y1-my), min(width, x2+mx), min(height, y2+my)


def emotion_analysis(frame_path: str, frame_index: int):
    bgr = cv2.imread(frame_path)
    if bgr is None:
        logger.warning(f"Frame {frame_index} could not be read for emotion analysis.")
        return None, None
    bbox = detect_face_bbox(bgr)
    face = bgr if not bbox else bgr[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
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
    num_frames: int,
    expression: Optional[dict] = None
) -> None:
    logger.info(f"Rebuilding frame {frame_index}/{num_frames}")
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
    cv2.imwrite(out_path, frame_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

def get_fps_fraction(video_path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "json", video_path
    ]
    try:
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info = json.loads(out.stdout.decode("utf-8"))
        rate = info["streams"][0]["avg_frame_rate"]
        frac = Fraction(rate)
        return frac.numerator, frac.denominator
    except Exception:
        # Safe default if probing fails
        return 30000, 1001

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

def rebuild_video_ffmpeg(frames_dir: str, audio_path: Optional[str], output_path: str,
                         fps_num: int, fps_den: int, keep_all_frames: bool = True) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps_str = f"{fps_num}/{fps_den}"

    base = [
        "ffmpeg", "-y",
        "-start_number", "0",
        "-framerate", fps_str,          # exact input rate for the image sequence
        "-i", os.path.join(frames_dir, "%d.jpg"),
    ]

    if audio_path and os.path.exists(audio_path):
        cmd = base + [
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-vsync", "cfr",
            "-r", fps_str,
            "-c:a", "copy",
        ]
        if not keep_all_frames:
            cmd += ["-shortest"]
        cmd += [output_path]
    else:
        cmd = base + [
            "-map", "0:v:0",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-vsync", "cfr",
            "-r", fps_str,
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

def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )
    return (transcript.text or "").strip()

def analyze_spoken_content(audio_path: str, job_description: str, analysis_path: str) -> None:
    spoken_content_analysis = "No audio stream found."
    try:
        if audio_path:
            transcript = transcribe_audio(audio_path).strip()
            if not transcript:
                spoken_content_analysis = "No speech detected."
            else:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system",
                        "content": f"You are a career coach. Analyze the following interview answer and give concise, actionable feedback. The job description is: {job_description}"},
                        {"role": "user", "content": transcript},
                    ],
                    temperature=0.2,
                )
                spoken_content_analysis = resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Spoken content analysis skipped: {e}")
        spoken_content_analysis = "Analysis unavailable."

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(spoken_content_analysis)

def delayed_rmtree(path: str, delay_seconds: int = 30):
    import time
    time.sleep(delay_seconds)
    shutil.rmtree(path, ignore_errors=True)
