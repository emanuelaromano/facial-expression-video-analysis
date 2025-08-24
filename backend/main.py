from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from video_processing import router
from fastapi.responses import StreamingResponse
import time

app = FastAPI()

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

