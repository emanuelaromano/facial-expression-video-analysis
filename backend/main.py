from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from video_processing import router
import asyncio
from video_processing import delayed_rmtree
from video_processing import TEMP_BASE
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.create_task(delayed_rmtree(TEMP_BASE, 0))
    yield

app = FastAPI(lifespan=lifespan)

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

app.include_router(router, prefix="/video")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

