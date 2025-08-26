from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from video_processing import router, emotion_processor, emotion_model
from fastapi.responses import StreamingResponse
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Check if required models are loaded on startup"""
    logger.info("Starting up application...")
    
    if emotion_processor is None or emotion_model is None:
        logger.error("CRITICAL: Emotion analysis models failed to load!")
        logger.error("The application may not function correctly.")
    else:
        logger.info("✓ Emotion analysis models loaded successfully")
    
    logger.info("Application startup complete")

app.include_router(router, prefix="/video")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

