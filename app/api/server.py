#!/usr/bin/env python3
"""
FastAPI server for the real-time avatar system.
Handles WebSocket connections and video/audio processing.
"""

import os
import logging
import uvicorn
import argparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import asyncio
from typing import Optional
import torch
import traceback
from datetime import datetime

# Create logs directory
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(logs_dir, f'avatar_system_{timestamp}.log')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_file}")

# Initialize FastAPI app
app = FastAPI(title="Avatar System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        logger.info("Starting server initialization...")
        
        # Log environment variables
        logger.debug("Environment variables:")
        for key, value in os.environ.items():
            logger.debug(f"{key}: {value}")
        
        # Create models directory if it doesn't exist
        os.makedirs('app/models', exist_ok=True)
        
        # Force CPU mode if requested
        if os.getenv('FORCE_CPU') == '1':
            device = 'cpu'
            logger.info("Using CPU for inference (forced)")
        else:
            # Force CPU mode if CUDA is not available
            use_cuda = torch.cuda.is_available()
            device = 'cuda' if use_cuda else 'cpu'
            logger.info(f"Using {device.upper()} for inference")
            if use_cuda:
                logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        
        # Check if model files exist and log their sizes
        model_paths = {
            'stylegan': 'app/models/StyleGAN3_FFHQ_1024x1024.pkl',
            'wav2lip': 'app/models/wav2lip.pth'
        }
        
        for key, path in model_paths.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"Model {key} found at {path} (size: {size_mb:.2f} MB)")
            else:
                logger.warning(f"Model file not found: {path}")
        
        # Initialize pipeline with detailed logging
        try:
            from app.components.avatar_pipeline import AvatarPipeline
            logger.info("Initializing AvatarPipeline...")
            pipeline = AvatarPipeline(
                model_paths=model_paths,
                device=device
            )
            pipeline.start()
            logger.info("Avatar pipeline started successfully")
        except Exception as pipeline_error:
            logger.error("Pipeline initialization failed!")
            logger.error(traceback.format_exc())
            raise
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(traceback.format_exc())
        pipeline = None

@app.on_event("shutdown")
async def shutdown_event():
    global pipeline
    if pipeline:
        pipeline.stop()
        logger.info("Avatar pipeline stopped")

@app.get("/")
async def root():
    """Root endpoint to check server status."""
    status = {"status": "running", "pipeline": pipeline is not None}
    logger.info(f"Status check: {status}")
    return status

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        if pipeline is None:
            logger.error("Pipeline not initialized")
            await websocket.send_json({
                "type": "error",
                "message": "Avatar pipeline not initialized"
            })
        
        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"Received WebSocket data: {data['type']}")
                
                if "type" not in data:
                    logger.warning("Received message without type field")
                    continue
                    
                if data["type"] == "video":
                    # Process video frame
                    if "data" not in data:
                        logger.warning("Video message missing data field")
                        continue
                        
                    if pipeline:
                        frame_data = data["data"]
                        result = await pipeline.process_frame(frame_data)
                        if result:
                            await websocket.send_json({
                                "type": "video",
                                "data": result
                            })
                        else:
                            logger.warning("Frame processing returned no result")
                            # Send placeholder response for testing
                            placeholder = create_placeholder_image()
                            await websocket.send_json({
                                "type": "video",
                                "data": placeholder
                            })
                    else:
                        logger.warning("Pipeline not available, sending placeholder")
                        # Pipeline not available, send placeholder
                        placeholder = create_placeholder_image()
                        await websocket.send_json({
                            "type": "video",
                            "data": placeholder
                        })
                        
                elif data["type"] == "audio":
                    # Process audio data
                    if "data" not in data:
                        logger.warning("Audio message missing data field")
                        continue
                        
                    if pipeline:
                        audio_data = base64.b64decode(data["data"])
                        await pipeline.process_audio(audio_data)
                        
                elif data["type"] == "ping":
                    # Respond to ping with pong
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": data.get("timestamp", 0)
                    })
                    
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                logger.error(traceback.format_exc())
                continue
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())
        try:
            await websocket.close()
        except:
            pass

def create_placeholder_image():
    """Create a placeholder image for testing when pipeline is not available."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        img[:, i] = [i//2, 100, 255-i//2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Avatar Placeholder", (100, 256), font, 1, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Start the avatar system API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode even if CUDA is available")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variable for CPU mode
    if args.cpu:
        os.environ["FORCE_CPU"] = "1"
    
    # Log startup configuration
    logger.info(f"Starting server on port {args.port}")
    logger.info(f"CPU mode: {args.cpu}")
    
    # Start uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=args.port)