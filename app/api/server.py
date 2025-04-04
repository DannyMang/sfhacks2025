#!/usr/bin/env python3
"""
FastAPI server for the avatar system.
"""

import os
import time
import cv2
import numpy as np
import base64
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import threading
import queue
import io
import logging

# Import our avatar pipeline
from app.components.avatar_pipeline import AvatarPipeline

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Real-Time Avatar API", description="API for real-time avatar generation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, in production restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API requests and responses
class AudioRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    sample_rate: int = 16000

class VideoRequest(BaseModel):
    frame_data: str  # Base64 encoded image
    timestamp: float

class AvatarResponse(BaseModel):
    frame_data: str  # Base64 encoded image
    latency: float
    timestamp: float

# Global avatar pipeline instance
avatar_pipeline = None

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_frame(self, frame, latency, websocket: WebSocket):
        """Send a frame to a specific client."""
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convert to base64
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create response
        response = {
            "frame_data": frame_b64,
            "latency": latency,
            "timestamp": time.time()
        }
        
        await websocket.send_json(response)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global avatar_pipeline
    
    # Initialize avatar pipeline
    models_dir = os.path.join("app", "models")
    os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Check for CUDA availability
    device = "cpu"
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            device = "cuda"
    except:
        logger.warning("cv2.cuda not available, falling back to CPU")
        
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except:
        logger.warning("PyTorch CUDA not available, falling back to CPU")
    
    logger.info(f"Initializing avatar pipeline with device: {device}")
    avatar_pipeline = AvatarPipeline(models_dir=models_dir, device=device)
    
    # Start the pipeline
    avatar_pipeline.start()
    logger.info("Avatar pipeline started")

@app.on_event("shutdown")
async def shutdown_event():
    """Release resources on shutdown."""
    global avatar_pipeline
    
    if avatar_pipeline:
        logger.info("Stopping avatar pipeline")
        avatar_pipeline.stop()

@app.post("/api/avatar/process_frame", response_model=AvatarResponse)
async def process_frame(request: VideoRequest):
    """Process a single video frame."""
    global avatar_pipeline
    
    if not avatar_pipeline or not avatar_pipeline.is_running:
        return JSONResponse(status_code=503, content={"error": "Avatar pipeline not available"})
    
    # Decode base64 frame
    img_bytes = base64.b64decode(request.frame_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None or frame.size == 0:
        return JSONResponse(status_code=400, content={"error": "Invalid frame data"})
    
    # Process the frame
    avatar_pipeline.process_frame(frame)
    
    # Get the result (with timeout)
    result = avatar_pipeline.get_result(timeout=0.5)
    
    if not result:
        return JSONResponse(status_code=408, content={"error": "Frame processing timeout"})
    
    # Unpack result
    output_frame, latency = result
    
    # Encode output frame to base64
    _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Create response
    response = AvatarResponse(
        frame_data=frame_b64,
        latency=latency,
        timestamp=time.time()
    )
    
    return response

@app.post("/api/avatar/process_audio")
async def process_audio(request: AudioRequest):
    """Process an audio chunk."""
    global avatar_pipeline
    
    if not avatar_pipeline or not avatar_pipeline.is_running:
        return JSONResponse(status_code=503, content={"error": "Avatar pipeline not available"})
    
    # Decode base64 audio
    audio_bytes = base64.b64decode(request.audio_data)
    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
    
    # Process the audio
    avatar_pipeline.process_audio(audio_np, request.sample_rate)
    
    return {"status": "success"}

@app.websocket("/ws/avatar")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)
    
    try:
        # Result streaming task
        async def stream_results():
            while True:
                # Check for new results
                result = avatar_pipeline.get_result(timeout=0.01)
                if result:
                    output_frame, latency = result
                    await manager.send_frame(output_frame, latency, websocket)
                await asyncio.sleep(0.01)
        
        # Start streaming task
        streaming_task = asyncio.create_task(stream_results())
        
        # Process incoming messages
        while True:
            # Wait for message with a timeout
            message = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
            
            # Check message type
            if "frame_data" in message:
                # Process video frame
                img_bytes = base64.b64decode(message["frame_data"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None and frame.size > 0:
                    avatar_pipeline.process_frame(frame)
            
            elif "audio_data" in message:
                # Process audio chunk
                audio_bytes = base64.b64decode(message["audio_data"])
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                sample_rate = message.get("sample_rate", 16000)
                
                avatar_pipeline.process_audio(audio_np, sample_rate)
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except asyncio.TimeoutError:
        # This is normal, just continue the loop
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up
        manager.disconnect(websocket)
        try:
            streaming_task.cancel()
        except:
            pass

def generate_frames():
    """Generator for video streaming (for debugging)."""
    global avatar_pipeline
    
    if not avatar_pipeline:
        return
    
    while True:
        # Attempt to get a processed frame
        result = avatar_pipeline.get_result(timeout=0.1)
        
        if result:
            frame, _ = result
            
            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            # Generate a blank frame if no result
            blank_frame = np.zeros((512, 512, 3), dtype=np.uint8)
            _, jpeg = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        # Sleep to limit frame rate
        time.sleep(0.03)  # ~30 FPS

@app.get("/video_feed")
async def video_feed():
    """Video streaming endpoint (for debugging)."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Real-Time Avatar API is running", "status": "OK"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)