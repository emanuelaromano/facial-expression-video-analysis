import os
import shutil
import cv2
import logging
import subprocess
from dotenv import load_dotenv
from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from http import HTTPStatus
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import mediapipe as mp
from threading import Lock, Event
from typing import Optional
from openai import OpenAI
from zipfile import ZipFile, ZIP_DEFLATED
from fractions import Fraction
import json
import asyncio
from subprocess import Popen, PIPE
import time
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# Configure temp directory for Cloud Run compatibility
TEMP_ROOT = os.getenv("TEMP_ROOT", "/tmp")
TEMP_BASE = os.path.join(TEMP_ROOT, "hireview")
os.makedirs(TEMP_BASE, exist_ok=True)

# Delete our temp subtree at startup
shutil.rmtree(TEMP_BASE, ignore_errors=True)

router = APIRouter()

########################################################
# Logging setup
########################################################

# Logging setup
logger = logging.getLogger("hireview")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

########################################################
# Utility functions
########################################################

# Client disconnect handling
async def _abort_if_disconnected(request: Request, cancel_event: Event):
    await asyncio.sleep(0)
    if await request.is_disconnected():
        cancel_event.set()
        raise HTTPException(status_code=499, detail="Client Closed Request")

########################################################
# Video processing globals
########################################################

FACE_MESH_EVERY = int(os.getenv("FACE_MESH_EVERY", "2"))
EMOTION_EVERY   = int(os.getenv("EMOTION_EVERY", "10"))
NUM_WORKERS     = int(os.getenv("NUM_WORKERS", "4"))
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-proj-1234567890")

# Building the face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
mesh_lock = Lock()

# Face detection for emotion analysis
mp_fd = mp.solutions.face_detection
face_det = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
fd_lock = Lock()

# Emotion analysis
try:
    logger.info("Loading emotion analysis model...")
    emotion_processor = AutoImageProcessor.from_pretrained(
        "/opt/models/vit-fer",
        local_files_only=True
    )
    emotion_model = AutoModelForImageClassification.from_pretrained("/opt/models/vit-fer").eval()
    EMOTION_ID2LABEL = emotion_model.config.id2label
    logger.info("Emotion analysis model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load emotion analysis model: {e}")
    logger.error("Model files in /opt/models/vit-fer:")
    try:
        import os
        if os.path.exists("/opt/models/vit-fer"):
            for root, dirs, files in os.walk("/opt/models/vit-fer"):
                for file in files:
                    logger.error(f"  {os.path.join(root, file)}")
        else:
            logger.error("  /opt/models/vit-fer directory does not exist")
    except Exception as dir_error:
        logger.error(f"Could not list model directory: {dir_error}")
    
    # Set fallback values
    emotion_processor = None
    emotion_model = None
    EMOTION_ID2LABEL = {}

# Transcription client
client = OpenAI(api_key=OPENAI_API_KEY)

########################################################
# SSE Streaming endpoints
########################################################

class VideoStream:
    states = {}

    def __init__(self, uuid: str):
        pass

    def initialize_status(self, uuid: str):
        if uuid not in self.states.keys():
            self.states[uuid] = {
                "state": "initializing",
                "progress": 2
            }
        else:
            self.states[uuid]["state"] = "initializing"
            self.states[uuid]["progress"] = 2

    def update_status(self, uuid: str, state: str, progress: int):
        self.states[uuid] = {
            "state": state,
            "progress": progress
        }

    def get_video_stream(self, uuid: str):
        try:
            while self.states[uuid]["progress"] < 100:
                yield f"data: {json.dumps({'state': self.states[uuid]['state'], 'progress': self.states[uuid]['progress']})}\r\n\r\n".encode("utf-8")
                time.sleep(2)
            if self.states[uuid]["progress"] == 100:
                self.update_status(uuid, "completed", 100)
                yield f"data: {json.dumps({'state': self.states[uuid]['state'], 'progress': self.states[uuid]['progress']})}\r\n\r\n".encode("utf-8")
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error in video stream for {uuid}: {e}")
            error_data = {"state": "error", "progress": self.states[uuid]["progress"]}
            yield f"data: {json.dumps(error_data)}\r\n\r\n".encode("utf-8")
    
    def cleanup(self, uuid: str):
        if uuid in self.states:
            del self.states[uuid]

def get_video_stream(uuid: str):
    try:
        return VideoStream(uuid).get_video_stream(uuid)
    except Exception as e:
        logger.error(f"Error getting video stream for {uuid}: {e}")
        return None

def update_status(uuid: str, state: str, progress: int):
    try:
        VideoStream(uuid).update_status(uuid, state, progress)
    except Exception as e:
        logger.error(f"Error updating status for {uuid}: {e}")

def initialize_status(uuid: str):
    try:
        VideoStream(uuid).initialize_status(uuid)
    except Exception as e:
        logger.error(f"Error initializing status for {uuid}: {e}")

def cleanup_status(uuid: str):
    try:
        VideoStream(uuid).cleanup(uuid)
    except Exception as e:
        logger.error(f"Error cleaning up status for {uuid}: {e}")

@router.get("/stream/{uuid}")
def stream(uuid: str):
    initialize_status(uuid)
    return StreamingResponse(
        get_video_stream(uuid), 
        media_type="text/event-stream", 
        headers={
            "Cache-Control": "no-cache", 
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

########################################################
# API endpoints
########################################################

@router.post("/analyze", status_code=HTTPStatus.OK)
async def analyze_video(
    request: Request,
    background_tasks: BackgroundTasks,
    uuid: str = Form(...),
    original_video: UploadFile = File(...),
    job_description: str = Form(...),
):
    initialize_status(uuid)
    cancel_event = Event()
    temp_dir = os.path.join(TEMP_BASE, uuid)
    temp_og_video_path = os.path.join(temp_dir, "video.mp4")
    temp_frames_dir = os.path.join(temp_dir, "frames")
    temp_processed_dir = os.path.join(temp_dir, "processed")
    temp_audio_path = os.path.join(temp_dir, "audio.m4a")
    temp_rebuilt_video_path = os.path.join(temp_dir, "rebuilt_video.mp4")
    bundle_path = os.path.join(temp_dir, f"{uuid}.zip")
    analysis_path = os.path.join(temp_dir, "spoken_content_analysis.txt")
    expressions_path = os.path.join(temp_dir, "expression_stats.json")
    logger.info(f"Analyzing video {uuid}")

    try:
        # Create temp folders
        os.makedirs(temp_frames_dir, exist_ok=True)
        os.makedirs(temp_processed_dir, exist_ok=True)
        update_status(uuid, "saving video", 5)

        # Save video and extract info
        save_original_video(original_video, temp_og_video_path)
        await _abort_if_disconnected(request, cancel_event)
        
        fps_num, fps_den = get_fps_fraction(temp_og_video_path)
        await _abort_if_disconnected(request, cancel_event)
        
        update_status(uuid, "extracting audio", 10)
        audio_path = extract_audio_ffmpeg(temp_og_video_path, temp_audio_path, cancel_event)
        if not audio_path:
            logger.warning("Audio extraction failed - video will be silent")
        await _abort_if_disconnected(request, cancel_event)
        
        num_frames = split_video_into_frames(temp_og_video_path, temp_frames_dir, cancel_event)
        await _abort_if_disconnected(request, cancel_event)

        # Analyze only needed frames
        expressions = {}
        expression_stats = {}
        needed_idxs = sorted(
            set(range(0, num_frames, FACE_MESH_EVERY)) |
            set(range(0, num_frames, EMOTION_EVERY))
        )
        
        logger.info(f"Starting analysis of {len(needed_idxs)} frames out of {num_frames} total frames")
        logger.info(f"Face mesh analysis every {FACE_MESH_EVERY} frames, emotion analysis every {EMOTION_EVERY} frames")

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            logger.info("Starting frame analysis...")
            futures = {
                executor.submit(analyze_frame, os.path.join(temp_frames_dir, f"{i}.jpg"), i, cancel_event): i
                for i in needed_idxs
            }
            completed = 0
            for future in as_completed(futures):
                # bail out quickly if canceled
                if cancel_event.is_set():
                    break
                await _abort_if_disconnected(request, cancel_event)
                idx = futures[future]
                result = future.result()
                expressions[str(idx)] = result or {}
                completed += 1
                if completed % 5 == 0 or completed == len(needed_idxs):
                    progress = int(10 + round(completed/len(needed_idxs) * 30, 0))
                    update_status(uuid, "analyzing frames", progress)
                    if completed % 10 == 0:
                        logger.info(f"Analyzed {completed}/{len(needed_idxs)} frames ({completed/len(needed_idxs)*100:.1f}%)")

        await _abort_if_disconnected(request, cancel_event)

        # Rebuild frames using most recent analyzed indices
        logger.info(f"Starting frame rebuilding process for {num_frames} frames...")
        alpha = 0.4
        last_label = None
        last_conf = None

        for i in range(num_frames):
            if cancel_event.is_set():
                raise HTTPException(status_code=499, detail="Client Closed Request")
            await _abort_if_disconnected(request, cancel_event)
            mesh_idx = min((i // FACE_MESH_EVERY) * FACE_MESH_EVERY, num_frames - 1)
            emo_idx  = min((i // EMOTION_EVERY)  * EMOTION_EVERY,  num_frames - 1)

            mesh_expression = expressions.get(str(mesh_idx), {}) or {}
            emotion_expression = expressions.get(str(emo_idx), {}) or {}

            label = emotion_expression.get("emotion")
            conf  = emotion_expression.get("confidence")

            expression_stats[label] = expression_stats.get(label, 0) + 1

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
            rebuild_frame(i, temp_frames_dir, temp_processed_dir, num_frames, uuid, final_expression)

        logger.info(f"Frame rebuilding completed. Starting video reconstruction...")
        
        # Normalize expression stats
        normalized_expression_stats = normalize_expression_stats(expression_stats)

        # Rebuild video
        update_status(uuid, "rebuilding video", 90)
        rebuilt_video_path = rebuild_video_ffmpeg(
            temp_processed_dir,
            audio_path,
            temp_rebuilt_video_path,
            fps_num,
            fps_den,
            cancel_event
        )
        await _abort_if_disconnected(request, cancel_event)
        
        if not os.path.exists(rebuilt_video_path):
            logger.error("rebuilt video not found at %s", rebuilt_video_path)
            raise HTTPException(status_code=500, detail="Failed to build output video")
        
        logger.info(f"Video reconstruction completed successfully: {rebuilt_video_path}")
        logger.info(f"Output video size: {os.path.getsize(rebuilt_video_path)} bytes")

        # Spoken-content analysis (guard for missing audio)
        await _abort_if_disconnected(request, cancel_event)
        update_status(uuid, "transcribing audio", 95)
        analyze_spoken_content(audio_path, job_description, analysis_path, cancel_event)
        await _abort_if_disconnected(request, cancel_event)

        with open(expressions_path, "w") as f:
            json.dump(normalized_expression_stats, f)

        # Bundle video + analysis into a ZIP
        if cancel_event.is_set():
            raise HTTPException(status_code=499, detail="Client Closed Request")
            
        with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
            zf.write(rebuilt_video_path, arcname="processed_video.mp4")
            zf.write(analysis_path, arcname="spoken_content_analysis.txt")
            zf.write(expressions_path, arcname="expression_stats.json")

        # Validate the ZIP file was created correctly
        if not os.path.exists(bundle_path):
            logger.error(f"Bundle file was not created at {bundle_path}")
            raise HTTPException(status_code=500, detail="Failed to create output bundle")
        
        bundle_size = os.path.getsize(bundle_path)
        if bundle_size == 0:
            logger.error(f"Bundle file is empty: {bundle_path}")
            raise HTTPException(status_code=500, detail="Output bundle is empty")
        
        logger.info(f"Analysis complete for video {uuid}. Bundle created: {bundle_path}")
        logger.info(f"Bundle size: {bundle_size} bytes")
        update_status(uuid, "completed", 100)
        
        # Verify ZIP file integrity
        try:
            with ZipFile(bundle_path, 'r') as test_zip:
                file_list = test_zip.namelist()
                logger.info(f"ZIP contents: {file_list}")
                if f"processed_video.mp4" not in file_list or "spoken_content_analysis.txt" not in file_list:
                    logger.error(f"ZIP file missing required contents: {file_list}")
                    raise HTTPException(status_code=500, detail="Output bundle is corrupted")
        except Exception as e:
            logger.error(f"ZIP file validation failed: {e}")
            raise HTTPException(status_code=500, detail="Output bundle validation failed")

        # Cleanup AFTER response is sent
        background_tasks.add_task(delayed_rmtree, temp_dir, 30)
        # Also cleanup status after a delay to ensure streaming is complete
        background_tasks.add_task(delayed_cleanup_status, uuid, 30)

        return FileResponse(
            path=bundle_path,
            media_type="application/zip",
            filename=f"{uuid}.zip",
        )

    except Exception as e:
        logger.exception("unexpected error in analyze_video")
        # Best effort: if a partially created bundle exists, return a 500 instead of streaming it
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/question", status_code=HTTPStatus.OK)
async def generate_interview_question(job_description: str = Form(...)):
    try:
        question = generate_question(job_description)
        return {
            "question": question
        }
    except Exception as e:
        logger.exception("unexpected error in generate_interview_question")
        raise Exception(f"Unexpected error: {e}")

########################################################
# Video processing functions
########################################################

def save_original_video(original_video: UploadFile, path: str) -> None:
    with open(path, "wb") as f:
        f.write(original_video.file.read())


def split_video_into_frames(video_path: str, frames_dir: str, cancel_event: Event) -> int:
    video = cv2.VideoCapture(video_path)
    count = 0
    try:
        while True:
            if cancel_event.is_set():
                break
            ret, frame = video.read()
            if not ret:
                break
            frame_path = os.path.join(frames_dir, f"{count}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
    finally:
        video.release()
    return count


def analyze_frame(frame_path: str, frame_index: int, cancel_event: Event) -> dict:
    try:
        if cancel_event.is_set():
            return {"face_mesh": None, "emotion": None, "confidence": None}
        face_mesh, emotion, confidence = None, None, None
        if frame_index % FACE_MESH_EVERY == 0:
           face_mesh = face_mesh_analysis(frame_path, frame_index)
        if cancel_event.is_set():
            return {"face_mesh": None, "emotion": None, "confidence": None}
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
    # Check if models are loaded
    if emotion_processor is None or emotion_model is None:
        logger.warning(f"Emotion analysis models not loaded for frame {frame_index}")
        return None, None
    
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
    uuid: str,
    expression: Optional[dict] = None
) -> None:
    # Only log every 10th frame to reduce log noise, but always log the last frame
    if frame_index % 5 == 0 or frame_index == num_frames - 1:
        update_status(uuid, "rebuilding frames", int(round(40 + frame_index * 30 / num_frames, 0)))
        if frame_index % 10 == 0:
            logger.info(f"Rebuilding frame {frame_index + 1}/{num_frames}")
    
    frame_path = os.path.join(temp_frames_dir, f"{frame_index}.jpg")
    frame_image = cv2.imread(frame_path)
    if frame_image is None:
        logger.warning(f"Could not read frame {frame_index} from {frame_path}")
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
    
    try:
        cv2.imwrite(out_path, frame_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            logger.error(f"Failed to write frame {frame_index} to {out_path}")
    except Exception as e:
        logger.error(f"Error writing frame {frame_index}: {e}")

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

def extract_audio_ffmpeg(input_video: str, output_audio: str, cancel_event: Event) -> Optional[str]:
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
                         fps_num: int, fps_den: int, cancel_event: Event, keep_all_frames: bool = True) -> str:
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
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
        # Poll while watching for cancel
        while True:
            if cancel_event.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                raise HTTPException(status_code=499, detail="Client Closed Request")
            ret = proc.poll()
            if ret is not None:
                if ret != 0:
                    # read a small chunk of stderr for logs
                    err = (proc.stderr.read() or b"").decode(errors="ignore")[:400]
                    logger.error(f"ffmpeg video rebuild failed (code {ret}): {err}")
                    raise RuntimeError("ffmpeg failed")
                break
            # brief sleep to avoid busy-wait
            time_sleep = 0.05
            import time; time.sleep(time_sleep)
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Output video not created: {output_path}")
        return output_path
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"ffmpeg video rebuild failed: {e}")
        raise


#################################
# Transcript analysis functions
#################################

def transcribe_audio(audio_path: str, cancel_event: Event) -> str:
    if cancel_event.is_set():
        return ""
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )
    return (transcript.text or "").strip()

class SpokenContentAnalysis(BaseModel):
    analysis: str = Field(description="The analysis of the spoken content")

def analyze_spoken_content(audio_path: str, job_description: str, analysis_path: str, cancel_event: Event) -> None:
    spoken_content_analysis = "No audio stream found."
    try:
        if audio_path:
            if cancel_event.is_set():
                return
            transcript = transcribe_audio(audio_path, cancel_event).strip()
            if not transcript:
                spoken_content_analysis = "No speech detected."
            else:
                if cancel_event.is_set():
                    return

                response = client.responses.parse(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system",
                        "content": f"You are a career coach. Analyze the following interview answer and give concise, actionable feedback. The job description is: {job_description}"},
                        {"role": "user", "content": transcript},
                    ],
                    text_format=SpokenContentAnalysis,
                )
                spoken_content_analysis = response.output_parsed.analysis
    except Exception as e:
        logger.warning(f"Spoken content analysis skipped: {e}")
        spoken_content_analysis = "Analysis unavailable."

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(spoken_content_analysis)

def normalize_expression_stats(expression_stats: dict) -> dict:
    # Normalize the expression stats to be between 0 and 1
    total_expressions = sum(expression_stats.values())
    normalized_expression_stats = {}
    other_count = 0
    for label, count in expression_stats.items():
        value = round(count / total_expressions, 2) if total_expressions > 0 else 0
        if value < 0.02:
            other_count += value
        else:
            normalized_expression_stats[label] = value
    if other_count > 0:
        normalized_expression_stats["other"] = other_count
    return normalized_expression_stats

########################################################
# Delayed cleanup functions
########################################################

def delayed_rmtree(path: str, delay_seconds: int = 30):
    time.sleep(delay_seconds)
    shutil.rmtree(path, ignore_errors=True)
    logger.info(f"Cleaned up {path}")

def delayed_cleanup_status(uuid: str, delay_seconds: int = 60):
    time.sleep(delay_seconds)
    cleanup_status(uuid)
    logger.info(f"Cleaned up status for {uuid}")

########################################################
# Generate question functions
########################################################

class QuestionResponse(BaseModel):
    question: str = Field(description="The interview question")

def generate_question(job_description: str) -> str:
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": f"You are a career coach. Generate a possible interview question for the following job description: {job_description}",
            },
        ],
        text_format=QuestionResponse,
    )
    return response.output_parsed.question





