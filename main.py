"""
FastAPI Backend — Reconocimiento Facial con ESP32-CAM
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routes import esp
from app.routes import recognition
from app.routes import auth as auth_routes
from app.routes.esp import motion_capture_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(motion_capture_loop())
    yield
    task.cancel()


app = FastAPI(
    title="Face Recognition API",
    description="Reconocimiento facial con ESP32-CAM, FaceNet y FastAPI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(esp.router, prefix="/api/v1/esp", tags=["ESP32"])
app.include_router(recognition.router, prefix="/api/v1", tags=["Face Recognition"])
app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["Auth"])


@app.get("/")
async def root():
    return {"message": "Face Recognition API v2", "status": "online", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
