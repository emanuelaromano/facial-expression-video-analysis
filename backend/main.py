from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from video_processing import router, init_once
from jwt_authentication import router as access_router
from contextlib import asynccontextmanager
import logging
import sys, os
sys.stderr = open(os.devnull, 'w')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the video processing module once
    logger = logging.getLogger("hireview")    
    if not getattr(app.state, "inited", False):
        init_once()
        app.state.inited = True
    else:
        logger.info("Already initialized, skipping initialization")
    yield

app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# CORS middleware with streaming-friendly headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "https://hireview-prep-2c2d5.web.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(router, prefix="/api/video")
app.include_router(access_router, prefix="/api/auth")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

