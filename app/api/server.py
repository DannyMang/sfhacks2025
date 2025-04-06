#!/usr/bin/env python3
"""
FastAPI server for the real-time avatar system.
Handles WebSocket connections and video/audio processing.
"""

import os
import logging
import uvicorn
import argparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import asyncio
from typing import Optional, Dict
import torch
import traceback
from datetime import datetime
import json
import time  # Add this import for the time module

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
    allow_origins=["*"],  # In production, replace with specific origins
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

# Updated main WebSocket handler
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        logger.info("WebSocket connection attempt received")
        logger.info(f"Client info: {websocket.client.host}:{websocket.client.port}")
        
        # Accept the connection
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        # Check if pipeline is initialized
        if pipeline is None:
            logger.error("Pipeline not initialized")
            await websocket.send_json({
                "type": "error",
                "message": "Avatar pipeline not initialized"
            })
            await websocket.close(code=1011, reason="Pipeline not initialized")
            return
        
        # Send initial status message
        await websocket.send_json({
            "type": "status",
            "message": "Connected to avatar system",
            "pipeline_ready": True
        })
        
        logger.info("Starting WebSocket communication loop")
        while True:
            try:
                # Wait for a message
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                logger.debug(f"Received WebSocket message type: {message.get('type', 'unknown')}")
                
                if message["type"] == "websocket.receive":
                    # Handle text messages (JSON)
                    if "text" in message:
                        try:
                            data = json.loads(message["text"])
                            logger.debug(f"Parsed message data keys: {list(data.keys())}")
                            
                            # Handle ping message
                            if data.get("type") == "ping":
                                await websocket.send_json({
                                    "type": "pong",
                                    "timestamp": data.get("timestamp", 0),
                                    "server_time": time.time()
                                })
                                continue
                            
                            # Process video frame
                            frame_data = data.get("frame_data", "")
                            if frame_data:
                                logger.debug("Processing video frame")
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
                            
                            # Process audio data
                            audio_data = data.get("audio_data", "")
                            if audio_data:
                                logger.debug("Processing audio data")
                                # Convert base64 audio to bytes
                                audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
                                await pipeline.process_audio(audio_bytes)
                                logger.debug("Audio processed")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": "Invalid JSON message format"
                            })
                            
                    # Handle binary messages
                    elif "bytes" in message:
                        logger.debug("Received binary message")
                        # Handle binary data if needed
                        await websocket.send_json({
                            "type": "error",
                            "message": "Binary messages are not supported"
                        })
                    
                    # Unknown message format
                    else:
                        logger.warning(f"Unknown message format: {message}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Unknown message format"
                        })
                
                elif message["type"] == "websocket.disconnect":
                    logger.info("Client initiated disconnect")
                    break
                
            except asyncio.TimeoutError:
                # Send a heartbeat to keep the connection alive
                logger.debug("Connection idle, sending heartbeat")
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except:
                    logger.error("Failed to send heartbeat, closing connection")
                    break
                    
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
                    break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())
        try:
            await websocket.close()
        except:
            pass

@app.websocket("/ws/calibration")
async def calibration_websocket(websocket: WebSocket):
    logger.info("New calibration connection request received")
    try:
        # Accept the connection
        await websocket.accept()
        logger.info("Calibration WebSocket accepted")

        # Send a test message immediately to confirm the connection is working
        test_message = {
            "type": "test",
            "message": "Connection successful"
        }
        logger.info(f"Sending test message: {test_message}")
        await websocket.send_json(test_message)
        
        # Check if pipeline is ready
        if pipeline is None or pipeline.avatar_generator is None:
            error_msg = "Avatar system not initialized"
            logger.error(error_msg)
            await websocket.send_json({
                "type": "error",
                "message": error_msg
            })
            await websocket.close(code=1011, reason="System not initialized")
            return
        
        # Start a background task to send training updates
        training_task = None
        
        # Process messages
        while True:
            message = await websocket.receive_json()
            logger.info(f"Received message of length: {len(str(message))}")
            
            if "type" not in message:
                logger.warning("Received message without type field")
                continue
                
            message_type = message["type"]
            logger.info(f"Received message type: {message_type}")
            
            if message_type == "ping":
                # Respond with pong to keep connection alive
                await websocket.send_json({"type": "pong"})
                
            elif message_type == "start_calibration":
                # Start the calibration process
                logger.info("Calibration start requested")
                
                # Reset calibration state
                pipeline.avatar_generator.reset_calibration()
                
                # Get the first pose
                first_pose = pipeline.avatar_generator.get_next_calibration_pose()
                
                if first_pose:
                    logger.info(f"Calibration started successfully, first pose: {first_pose}")
                    await websocket.send_json({
                        "type": "calibration_status",
                        "status": "calibrating",
                        "next_pose": first_pose,
                        "progress": 0,
                        "message": "Starting calibration. Please follow the pose instructions."
                    })
                else:
                    logger.error("Failed to start calibration")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to start calibration"
                    })
                    
            elif message_type == "calibration_frame":
                # Process a calibration frame
                if "frame_data" not in message or "pose_type" not in message:
                    logger.warning("Invalid calibration frame message")
                    continue
                    
                frame_data = message["frame_data"]
                pose_type = message["pose_type"]
                
                logger.info(f"Processing calibration frame for pose: {pose_type}")
                
                # Process the frame
                result = await pipeline.process_calibration_frame(frame_data, pose_type)
                
                if result:
                    if result.get("status") == "training":
                        # Start sending training updates
                        if training_task is None:
                            training_task = asyncio.create_task(send_training_updates(websocket))
                            
                        await websocket.send_json({
                            "type": "calibration_status",
                            "status": "training",
                            "progress": 0,
                            "message": "Starting avatar training..."
                        })
                    else:
                        # Send the next pose
                        await websocket.send_json({
                            "type": "calibration_status",
                            "status": "calibrating",
                            "next_pose": result.get("next_pose"),
                            "progress": result.get("progress", 0),
                            "message": result.get("message", "Continue with calibration")
                        })
                else:
                    logger.error("Failed to process calibration frame")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to process calibration frame"
                    })
            
    except WebSocketDisconnect:
        logger.info("Calibration WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in calibration WebSocket: {e}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Clean up any background tasks
        if training_task and not training_task.done():
            training_task.cancel()

async def send_training_updates(websocket: WebSocket):
    """Send periodic updates about training progress."""
    try:
        while True:
            if pipeline and pipeline.avatar_generator:
                status = pipeline.avatar_generator.get_training_status()
                
                # Send the status update
                await websocket.send_json({
                    "type": "calibration_status",
                    **status
                })
                
                # If training is complete or failed, stop sending updates
                if status.get("status") in ["complete", "error"]:
                    break
            
            # Wait before sending the next update
            await asyncio.sleep(1.0)
    except Exception as e:
        logger.error(f"Error sending training updates: {e}")
        logger.error(traceback.format_exc())

# Add REST endpoints for calibration status
@app.get("/calibration/status")
async def get_calibration_status():
    """Get current calibration/training status."""
    if pipeline is None or pipeline.avatar_generator is None:
        return {"status": "error", "message": "Avatar system not initialized"}
    
    return pipeline.avatar_generator.get_training_status()

@app.post("/calibration/start")
async def start_calibration():
    """Start or restart calibration process."""
    logger.info("Calibration start requested")
    
    if pipeline is None:
        logger.error("Pipeline not initialized")
        raise HTTPException(status_code=500, message="Avatar system not initialized - pipeline is None")
        
    if pipeline.avatar_generator is None:
        logger.error("Avatar generator not initialized")
        raise HTTPException(status_code=500, message="Avatar system not initialized - avatar generator is None")
    
    try:
        next_pose = pipeline.avatar_generator.start_calibration()
        logger.info(f"Calibration started successfully, first pose: {next_pose}")
        return {
            "status": "started",
            "next_pose": next_pose
        }
    except Exception as e:
        logger.error(f"Error starting calibration: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify server is running."""
    return {
        "status": "ok",
        "message": "Server is running",
        "pipeline_initialized": pipeline is not None,
        "avatar_generator_initialized": pipeline is not None and pipeline.avatar_generator is not None
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify server connectivity."""
    return {
        "status": "ok", 
        "time": time.time(),
        "pipeline_initialized": pipeline is not None
    }

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