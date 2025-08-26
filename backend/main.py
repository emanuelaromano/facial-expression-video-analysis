from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from video_processing import router, emotion_processor, emotion_model
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("Starting up application...")
    
    if emotion_processor is None and emotion_model is None:
        logger.error("CRITICAL: Both analysis models failed to load!")
        logger.error("The application may not function correctly.")
    elif emotion_processor is None:
        logger.error("CRITICAL: Emotion analysis processor failed to load!")
        logger.error("The application may not function correctly.")
    elif emotion_model is None:
        logger.error("CRITICAL: Emotion analysis model failed to load!")
        logger.error("The application may not function correctly.")
    else:
        logger.info("✓ Emotion analysis models loaded successfully")
    
    logger.info("Application startup complete")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/video")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

