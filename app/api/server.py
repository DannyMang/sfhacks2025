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
import json

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

# Set OpenMP environment variable to avoid conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
            # Check CUDA availability
            use_cuda = torch.cuda.is_available()
            device = 'cuda' if use_cuda else 'cpu'
            logger.info(f"Using {device.upper()} for inference")
            if use_cuda:
                logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        
        # Import here to avoid circular imports
        from app.components.avatar_pipeline import AvatarPipeline
        
        # Define model paths
        model_paths = {
            'stylegan': os.path.abspath('app/models/stylegan3-t-ffhqu-1024x1024.pkl'),
            'wav2lip': os.path.abspath('app/models/wav2lip.pth'),
        }
        
        # Log model paths
        logger.info("Model paths:")
        for model_name, model_path in model_paths.items():
            exists = os.path.exists(model_path)
            logger.info(f"{model_name}: {model_path} (exists: {exists})")
        
        # Check if models exist
        missing_models = [name for name, path in model_paths.items() if not os.path.exists(path)]
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            logger.warning("Using placeholder mode due to missing models")
            return
        
        # Initialize pipeline
        logger.info("Initializing avatar pipeline...")
        pipeline = AvatarPipeline(model_paths, device=device)
        
        # Start pipeline
        logger.info("Starting avatar pipeline...")
        await pipeline.start()
        
        logger.info("Server initialization complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    global pipeline
    logger.info("Shutting down server...")
    if pipeline:
        await pipeline.stop()
    logger.info("Server shutdown complete")

@app.get("/")
async def root():
    """Root endpoint to check server status."""
    status = {
        "status": "running", 
        "pipeline": pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": pipeline.device if pipeline else "none"
    }
    logger.info(f"Status check: {status}")
    return status

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        logger.info("WebSocket connection attempt received")
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        if pipeline is None:
            logger.error("Pipeline not initialized")
            await websocket.send_json({
                "type": "error",
                "message": "Avatar pipeline not initialized"
            })
            await websocket.close()
            return
        
        logger.info("Starting WebSocket communication loop")
        while True:
            try:
                message = await websocket.receive()
                logger.debug(f"Received WebSocket message type: {message.get('type', 'unknown')}")
                
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        data = json.loads(message["text"])
                        logger.debug(f"Parsed message data: {list(data.keys())}")
                        
                        frame_data = data.get("frame_data", "")
                        audio_data = data.get("audio_data", "")
                        
                        if frame_data:
                            logger.debug("Processing video frame")
                            # Process the frame through the pipeline
                            result = await pipeline.process_frame(frame_data)
                            if result:
                                await websocket.send_json({
                                    "type": "video",
                                    "frame": result
                                })
                                logger.debug("Sent processed frame")
                            else:
                                # Send placeholder if processing fails
                                placeholder = create_placeholder_image()
                                await websocket.send_json({
                                    "type": "video",
                                    "frame": placeholder
                                })
                                logger.debug("Sent placeholder frame")
                        
                        if audio_data:
                            logger.debug("Processing audio data")
                            # Convert base64 audio to bytes
                            audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
                            await pipeline.process_audio(audio_bytes)
                            logger.debug("Audio processed")
                    elif "bytes" in message:
                        logger.debug("Received binary message")
                        # Handle binary data if needed
                        pass
                    else:
                        logger.warning(f"Unknown message format: {message}")
                        
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                logger.error(traceback.format_exc())
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except:
                    logger.error("Failed to send error message to client")
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
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def parse_args():
    parser = argparse.ArgumentParser(description="Start the avatar system API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode even if CUDA is available")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variable for CPU mode
    if args.cpu:
        os.environ["FORCE_CPU"] = "1"
    
    # Set debug level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log startup configuration
    logger.info(f"Starting server on port {args.port}")
    logger.info(f"CPU mode: {args.cpu}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Start uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="debug")