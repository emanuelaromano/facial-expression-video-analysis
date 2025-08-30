import os
import sys
import shutil
import cv2
import logging
import subprocess
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from http import HTTPStatus
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import mediapipe as mp
from threading import Lock, Event
from typing import Optional, Dict
from openai import OpenAI
from zipfile import ZipFile, ZIP_DEFLATED
from fractions import Fraction
import json
import asyncio
from subprocess import Popen, PIPE
import time
from pydantic import BaseModel, Field
import glob
from google.cloud import storage
from datetime import timedelta
import google.auth
from google.auth import impersonated_credentials
from google.api_core.exceptions import NotFound
import os
from redis_stream import set_status, get_status, clear_status, mark_cancel, is_cancelled, clear_cancel, r, _channel

load_dotenv()

########################################################
# Module initialization guard
########################################################

_INIT_DONE = False


def init_once():
    """Initialize the module once. Safe to call multiple times."""
    global _INIT_DONE, mp_face_mesh, face_mesh_model, mp_fd, face_det
    global emotion_processor, emotion_model, EMOTION_ID2LABEL, client
    global FACE_MESH_EVERY, EMOTION_EVERY, NUM_WORKERS, OPENAI_API_KEY
    if _INIT_DONE:
        return
    _INIT_DONE = True
    

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FACE_MESH_EVERY = int(os.getenv("FACE_MESH_EVERY"))
    EMOTION_EVERY = int(os.getenv("EMOTION_EVERY"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS"))
    
    # Validate model presence at startup
    MODEL_DIR = "/opt/models/vit-fer"
    model_files = [p for p in glob.glob(MODEL_DIR + "/**/*", recursive=True) if os.path.isfile(p)]
    if len(model_files) < 3:
        raise RuntimeError(f"Model dir looks empty: {MODEL_DIR} (found {len(model_files)} files)")
    
    # Create temp directory
    os.makedirs(TEMP_BASE, exist_ok=True)
    
    # Clear temp directory contents
    try:
        if os.path.exists(TEMP_BASE):
            for name in os.listdir(TEMP_BASE):
                p = os.path.join(TEMP_BASE, name)
                if os.path.isfile(p) or os.path.islink(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
            logger.info(f"Cleared contents of {TEMP_BASE}")
    except Exception as e:
        logger.warning(f"Failed to clear temp directory: {e}")
    
    # Initialize MediaPipe models
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_model = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
    mp_fd = mp.solutions.face_detection
    face_det = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    # Initialize emotion analysis models
    try:
        emotion_processor = AutoImageProcessor.from_pretrained(
            "/opt/models/vit-fer",
            local_files_only=True
        )
        emotion_model = AutoModelForImageClassification.from_pretrained("/opt/models/vit-fer").eval()
        EMOTION_ID2LABEL = emotion_model.config.id2label
    except Exception as e:
        logger.error(f"Failed to load emotion analysis model: {e}")
        # Set fallback values
        emotion_processor = None
        emotion_model = None
        EMOTION_ID2LABEL = {}
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    

########################################################
# Delayed cleanup functions
########################################################

async def delayed_rmtree(path: str, delay_seconds: int = 30):
    await asyncio.sleep(delay_seconds)
    try:
        if path == TEMP_BASE:
            for name in os.listdir(path):
                p = os.path.join(path, name)
                if os.path.isfile(p) or os.path.islink(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
            logger.info(f"Cleared contents of {path}")
        else:
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"Deleted directory {path}")
    except Exception as e:
        logger.warning(f"Failed to clean up {path}: {e}")


async def delayed_cleanup_status(uuid: str, delay_seconds: int = 30):
    await asyncio.sleep(delay_seconds)
    await clear_status(uuid)
    logger.info(f"Cleaned up Redis status for {uuid}")

########################################################
# Video processing globals
########################################################

TEMP_ROOT = os.getenv("TEMP_ROOT", "/tmp")              
TEMP_BASE = os.getenv("TEMP_BASE", os.path.join(TEMP_ROOT, "hireview"))
BUCKET = os.getenv("BUCKET", "backend-app-storage")
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "hireview-prep-470120")
TARGET_SA = os.getenv("TARGET_SERVICE_ACCOUNT", "916307297241-compute@developer.gserviceaccount.com")
SCOPES = ["https://www.googleapis.com/auth/devstorage.read_write"]

router = APIRouter()

########################################################
# Cooperative cancellation registry (per uuid)
########################################################

from typing import Dict

CANCEL_EVENTS: Dict[str, Event] = {}
CANCEL_LOCK = Lock()

def get_cancel_event(uuid: str) -> Event:
    with CANCEL_LOCK:
        if uuid not in CANCEL_EVENTS:
            CANCEL_EVENTS[uuid] = Event()
        return CANCEL_EVENTS[uuid]

def clear_cancel_event(uuid: str) -> None:
    with CANCEL_LOCK:
        CANCEL_EVENTS.pop(uuid, None)

########################################################
# Logging setup
########################################################

logger = logging.getLogger("hireview")
logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] - %(message)s "
))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

########################################################
# Video processing globals
########################################################

# Environment variables (will be parsed in init_once)
FACE_MESH_EVERY = None
EMOTION_EVERY = None
NUM_WORKERS = None
OPENAI_API_KEY = None

# Global variables for models and clients (initialized in init_once)
mp_face_mesh = None
face_mesh_model = None
mesh_lock = Lock()

mp_fd = None
face_det = None
fd_lock = Lock()

emotion_processor = None
emotion_model = None
EMOTION_ID2LABEL = {}

client = None

# Storage client setup for both local and cloud
def get_storage_client():
    try:
        source_creds, _ = google.auth.default(scopes=SCOPES)
        impersonated = impersonated_credentials.Credentials(
            source_credentials=source_creds,
            target_principal=TARGET_SA,
            target_scopes=SCOPES,
            lifetime=3600,
        )
        return storage.Client(project=PROJECT, credentials=impersonated)
    except Exception as e:
        logger.error(f"Falling back to default credentials: {e}")
        return storage.Client()

########################################################
# SSE Streaming endpoints
########################################################




@router.get("/status/{uuid}")
async def read_status(uuid: str):
    try:
        status = await get_status(uuid)
        if not isinstance(status, dict):
            # Normalize: never return None/empty
            return {"state": "not_started", "progress": 0}
        # Coerce fields and defaults
        return {
            "state": str(status.get("state", "not_started")),
            "progress": int(float(status.get("progress", 0))) if status.get("progress") is not None else 0,
            "ts": status.get("ts"),  # optional
        }
    except Exception as e:
        logger.error(f"Error reading status for {uuid}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read status: {str(e)}")

# Optional: long-poll variant (fewer requests; proxy-safe)
@router.get("/status/{uuid}/wait")
async def wait_status(uuid: str, last_progress: int = 0, timeout: int = 30):
    try:
        cur = await get_status(uuid)
        if not isinstance(cur, dict):
            cur = {"state": "not_started", "progress": 0}
        if cur.get("progress", 0) > last_progress:
            return cur

        pubsub = r.pubsub()
        await pubsub.subscribe(_channel(uuid))
        try:
            # wait up to `timeout` seconds for a progress bump
            end = time.time() + timeout
            async for msg in pubsub.listen():
                if msg["type"] != "message":
                    continue
                data = json.loads(msg["data"])
                prog = int(float(data.get("progress", 0)))
                if prog > last_progress:
                    return {"state": data["state"], "progress": prog, "ts": data.get("ts")}
                if time.time() > end:
                    break
        finally:
            await pubsub.close()
        # timeout: return current snapshot
        final_status = await get_status(uuid)
        if not isinstance(final_status, dict):
            return {"state": "not_started", "progress": 0}
        return final_status
    except Exception as e:
        logger.error(f"Error in wait_status for {uuid}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to wait for status: {str(e)}")

@router.post("/cancel/{uuid}")
async def cancel_job(uuid: str):
    try:
        await mark_cancel(uuid)
        ev = get_cancel_event(uuid)
        ev.set()
        asyncio.create_task(delayed_rmtree(os.path.join(TEMP_BASE, uuid), 0))
        return {"ok": True, "cancelled": uuid}
    except Exception as e:
        logger.error(f"Error cancelling job {uuid}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

########################################################
# Upload video to GCS
########################################################

@router.get("/upload", status_code=HTTPStatus.OK)
async def get_signed_upload_url(uuid: str, filename: str):
    logger.info(f"Generating signed upload URL for uuid: {uuid}, filename: {filename}")
    
    # Set initial status immediately when job is created
    await set_status(uuid, "loading video", 2)
    
    # URL decode the filename in case it comes encoded
    from urllib.parse import unquote
    decoded_filename = unquote(filename)    
    client = get_storage_client()
    blob_path = f"hireview/{uuid}/{decoded_filename}"
    blob = client.bucket(BUCKET).blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=10),
        method="PUT",
        content_type="application/octet-stream",
    )
    return {
        "url": url, 
        "gcs_path": blob.name,
        "headers": {"Content-Type": "application/octet-stream"}
    }


########################################################
# API endpoints
########################################################

@router.post("/analyze", status_code=HTTPStatus.OK)
async def analyze_video(
    background_tasks: BackgroundTasks,
    uuid: str = Form(...),
    gcs_path: str = Form(...), 
    scenario_description: str = Form(...),
):
    cancel_event = get_cancel_event(uuid)
    await clear_cancel(uuid)
    await set_status(uuid, "processing", 5)
    
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

        # Get video from GCS and download to temp location
        client = get_storage_client()
        bucket = client.bucket(BUCKET)
        blob = bucket.blob(gcs_path)

        try:
            blob.download_to_filename(temp_og_video_path)
            logger.info(f"Successfully downloaded video from GCS: {gcs_path}")
        except NotFound:
            logger.error(f"Video file not found in GCS: {gcs_path}")
            logger.error(f"Bucket: {BUCKET}, Full path: {gcs_path}")
            raise HTTPException(
                status_code=404, 
                detail=f"Video file not found in storage: {gcs_path}. Please ensure the file was uploaded successfully."
            )
        except Exception as e:
            logger.error(f"Failed to download video from GCS: {gcs_path}, Error: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to download video from storage: {str(e)}"
            )

        fps_num, fps_den = get_fps_fraction(temp_og_video_path)
        
        audio_path = extract_audio_ffmpeg(temp_og_video_path, temp_audio_path, cancel_event)
        if not audio_path:
            logger.warning("Audio extraction failed - video will be silent")
        
        num_frames = split_video_into_frames(temp_og_video_path, temp_frames_dir, cancel_event)
        await set_status(uuid, "analyzing frames", 15)


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
            futures_map = {
                executor.submit(analyze_frame, os.path.join(temp_frames_dir, f"{i}.jpg"), i, cancel_event): i
                for i in needed_idxs
            }
            completed = 0
            try:
                for future in as_completed(futures_map):
                    # bail out quickly if canceled
                    if cancel_event.is_set() or await is_cancelled(uuid):
                        for f in futures_map.keys():
                            f.cancel()
                        raise HTTPException(status_code=499, detail="Client Closed Request")
                    idx = futures_map[future]
                    result = future.result()
                    expressions[str(idx)] = result or {}
                    completed += 1
                    if completed % 5 == 0 or completed == len(needed_idxs):
                        progress = int(10 + round(completed/len(needed_idxs) * 30, 0))
                        if completed % 10 == 0:
                            await set_status(uuid, "analyzing frames", progress)
                            logger.info(f"Analyzed {completed}/{len(needed_idxs)} frames ({completed/len(needed_idxs)*100:.1f}%)")
            finally:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    pass
        # Rebuild frames using most recent analyzed indices
        logger.info(f"Starting frame rebuilding process for {num_frames} frames...")
        await set_status(uuid, "rebuilding frames", 50)
        alpha = 0.4
        last_label = None
        last_conf = None

        # Use bounded concurrency for frame rebuilding
        sem = asyncio.Semaphore(64)  # tune as needed

        async def _rebuild_task(i, expr):
            async with sem:
                await rebuild_frame(i, temp_frames_dir, temp_processed_dir, num_frames, uuid, expr)

        tasks = []
        for i in range(num_frames):
            if cancel_event.is_set() or await is_cancelled(uuid):
                raise HTTPException(status_code=499, detail="Client Closed Request")
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
            tasks.append(asyncio.create_task(_rebuild_task(i, final_expression)))

        # Wait for all frame rebuilding tasks to complete
        await asyncio.gather(*tasks)

        logger.info(f"Frame rebuilding completed. Starting video reconstruction...")
        await set_status(uuid, "rebuilding video", 90)
        
        # Fail fast if frames are missing before calling ffmpeg
        missing = []
        for i in range(num_frames):
            p = os.path.join(temp_processed_dir, f"{i}.jpg")
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                missing.append(i)

        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Processed frames missing or empty: {missing[:10]}{'...' if len(missing)>10 else ''}"
            )
        
        # Normalize expression stats
        normalized_expression_stats = normalize_expression_stats(expression_stats)

        # Rebuild video
        rebuilt_video_path = rebuild_video_ffmpeg(
            temp_processed_dir,
            audio_path,
            temp_rebuilt_video_path,
            fps_num,
            fps_den,
            cancel_event
        )
        
        if not os.path.exists(rebuilt_video_path):
            logger.error("rebuilt video not found at %s", rebuilt_video_path)
            raise HTTPException(status_code=500, detail="Failed to build output video")
        
        logger.info(f"Video reconstruction completed successfully: {rebuilt_video_path}")
        logger.info(f"Output video size: {os.path.getsize(rebuilt_video_path)} bytes")
        # Spoken-content analysis (guard for missing audio)
        await set_status(uuid, "transcribing audio", 95)
        analyze_spoken_content(audio_path, scenario_description, analysis_path, cancel_event)
        await set_status(uuid, "saving analysis", 98)
        with open(expressions_path, "w") as f:
            json.dump(normalized_expression_stats, f)

        # Bundle video + analysis into a ZIP
        if cancel_event.is_set() or await is_cancelled(uuid):
            raise HTTPException(status_code=499, detail="Client Closed Request")
            
        with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
            # Safety check: ensure files exist before adding to ZIP
            if os.path.exists(rebuilt_video_path):
                zf.write(rebuilt_video_path, arcname="processed_video.mp4")
            else:
                logger.warning(f"Processed video not found: {rebuilt_video_path}")
                
            if os.path.exists(analysis_path):
                zf.write(analysis_path, arcname="spoken_content_analysis.txt")
            else:
                logger.warning(f"Analysis file not found: {analysis_path}")
                
            if os.path.exists(expressions_path):
                zf.write(expressions_path, arcname="expression_stats.json")
            else:
                logger.warning(f"Expression stats not found: {expressions_path}")

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
        
        # Verify ZIP file integrity
        try:
            with ZipFile(bundle_path, 'r') as test_zip:
                file_list = test_zip.namelist()
                if f"processed_video.mp4" not in file_list or "spoken_content_analysis.txt" not in file_list:
                    logger.error(f"ZIP file missing required contents: {file_list}")
                    raise HTTPException(status_code=500, detail="Output bundle is corrupted")
        except Exception as e:
            logger.error(f"ZIP file validation failed: {e}")
            raise HTTPException(status_code=500, detail="Output bundle validation failed")

        # Finalize
        await set_status(uuid, "completed", 100)
        
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
        # report error status
        try: 
            await set_status(uuid, "error", (await get_status(uuid)).get("progress", 0))
        except: 
            pass
        # Best effort: if a partially created bundle exists, return a 500 instead of streaming it
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    finally:
        # ensure we clear the cancel event so future jobs reuse a clean state
        clear_cancel_event(uuid)


@router.post("/transcript", status_code=HTTPStatus.OK)
async def generate_video_transcript(scenario_description: str = Form(...)):
    try:
        transcript = generate_transcript(scenario_description)
        return {
            "transcript": transcript
        }
    except Exception as e:
        logger.exception("unexpected error in generate_video_transcript")
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


async def rebuild_frame(
    frame_index: int,
    temp_frames_dir: str,
    temp_processed_dir: str,
    num_frames: int,
    uuid: str,
    expression: Optional[dict] = None
) -> None:
    # Only log every 10th frame to reduce log noise, but always log the last frame
    if frame_index % 5 == 0 or frame_index == num_frames - 1:
        if frame_index % 10 == 0:
            await set_status(uuid, "rebuilding frames", int(50 + round(frame_index/num_frames * 40, 0)))
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
        "-framerate", fps_str,
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
            "-shortest",  # Handle potential audio/video length mismatches
        ]
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
                    # read more of stderr for better debugging (using tail for actionable errors)
                    err = (proc.stderr.read() or b"").decode(errors="ignore")[-4000:]
                    logger.error(f"ffmpeg video rebuild failed (code {ret}): {err}")
                    raise RuntimeError("ffmpeg failed")
                break
            # brief sleep to avoid busy-wait
            time_sleep = 0.05
            time.sleep(time_sleep)
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

def analyze_spoken_content(audio_path: str, scenario_description: str, analysis_path: str, cancel_event: Event) -> None:
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
                        "content": f"""
                        You are a speech coach. Analyze the following interview answer and give concise, actionable feedback. The scenario description is: {scenario_description}
                        """},
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
# Generate question functions
########################################################

class TranscriptResponse(BaseModel):
    transcript: str = Field(description="The transcript of the speech")

def generate_transcript(scenario_description: str) -> str:
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": f"You are a speech coach. The user is practicing a speech. The scenario description is: {scenario_description}. Write a transcript of the speech in the user's own words.",
            },
        ],
        text_format=TranscriptResponse,
    )
    return response.output_parsed.transcript
