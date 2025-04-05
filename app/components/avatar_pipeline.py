#!/usr/bin/env python3
"""
Main avatar pipeline that integrates all components.
"""

import os
import time
import cv2
import numpy as np
import threading
import queue
import torch
import logging
import asyncio
import base64
from typing import Optional, Tuple, Dict
from app.utils.face_detector import FaceDetector
from app.components.avatar_generator import AvatarGenerator
from app.components.voice2face import Voice2Face
import traceback

class AvatarPipeline:
    def __init__(self, model_paths: Dict[str, str], device: str = 'cuda'):
        """
        Initialize the avatar pipeline.
        
        Args:
            model_paths: Dictionary containing paths to model files
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model_paths = model_paths
        
        # Initialize components
        self.face_detector = None
        self.avatar_generator = None
        self.voice2face = None
        self.is_running = asyncio.Event()
        
        # State variables
        self.prev_frame = None
        self.last_generated_frame_time = 0
        self.frame_interval = 1.0 / 15.0  # Target 15 FPS for GAN generation
        self.interpolation_factor = 0.5   # Blend factor for frame interpolation
        
        # For face detection
        self.current_head_pose = None
        self.current_facial_features = None
        
        # Initialize pipeline
        self.initialize_pipeline()
        
        # Worker threads
        self.worker_threads = []
    
    def initialize_pipeline(self):
        """Initialize all pipeline components."""
        try:
            self.logger.info("Starting pipeline initialization...")
            
            # Verify model files exist
            if not os.path.exists(self.model_paths['stylegan']):
                raise FileNotFoundError(f"StyleGAN3 model not found at {self.model_paths['stylegan']}")
            
            # Initialize face detector with error handling
            try:
                self.face_detector = FaceDetector()
                self.logger.info("Face detector initialized")
            except Exception as e:
                self.logger.error(f"Face detector failed: {e}")
                self.face_detector = None
            
            # Initialize avatar generator with error handling
            try:
                self.avatar_generator = AvatarGenerator(
                    model_path=self.model_paths['stylegan'],
                    device=self.device
                )
                self.logger.info("Avatar generator initialized")
            except Exception as e:
                self.logger.error(f"Avatar generator failed: {e}")
                self.logger.error(traceback.format_exc())
                self.avatar_generator = None
            
            # Initialize Voice2Face if we're using CUDA
            if self.device == 'cuda' and torch.cuda.is_available() and 'wav2lip' in self.model_paths:
                try:
                    self.voice2face = Voice2Face(
                        model_path=self.model_paths['wav2lip'],
                        device=self.device
                    )
                    self.logger.info("Voice2Face initialized")
                except Exception as e:
                    self.logger.error(f"Voice2Face failed: {e}")
                    self.logger.error(traceback.format_exc())
                    self.voice2face = None
            else:
                self.voice2face = None
                self.logger.info("Voice2Face skipped (CPU mode or model not available)")
            
            # Check if we have minimum required components
            if not self.face_detector or not self.avatar_generator:
                raise RuntimeError("Failed to initialize required components")
            
            self.logger.info("Pipeline initialization complete")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def start(self):
        """Start the pipeline."""
        self.logger.info("Avatar pipeline started")
        self.is_running.set()
        return True  # Return a value to avoid NoneType error
    
    def stop(self):
        """Stop the avatar pipeline."""
        self.is_running.clear()
        self.logger.info("Avatar pipeline stopped")
    
    async def process_frame(self, frame_data: str) -> Optional[str]:
        """Process a single video frame."""
        if not self.is_running.is_set():
            self.logger.warning("Pipeline not running")
            return None
        
        try:
            # Log frame data length for debugging
            self.logger.debug(f"Received frame data length: {len(frame_data) if frame_data else 0}")
            
            # Decode base64 frame
            if not frame_data:
                self.logger.warning("Empty frame data received")
                return None
            
            try:
                # Split off the data URL prefix if present
                if ',' in frame_data:
                    self.logger.debug("Frame data contains data URL prefix")
                    frame_data = frame_data.split(',')[1]
                
                image_data = base64.b64decode(frame_data)
                self.logger.debug(f"Decoded image data size: {len(image_data)} bytes")
                
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    self.logger.error("Failed to decode frame")
                    return None
                
                self.logger.debug(f"Decoded frame shape: {frame.shape}")
                
            except Exception as e:
                self.logger.error(f"Error decoding frame: {e}")
                self.logger.error(traceback.format_exc())
                return None
            
            # Detect face landmarks
            self.logger.debug("Detecting face landmarks")
            landmarks = self.face_detector.detect(frame)
            if landmarks is None:
                self.logger.warning("No face detected in frame - using default pose")
                head_pose = None  # Use default pose instead of returning None
            else:
                head_pose = self.face_detector.get_head_pose(landmarks, frame)
            
            # Generate avatar frame
            self.logger.info("Generating avatar frame")
            avatar_frame = self.avatar_generator.generate_frame(
                head_pose=head_pose,
                expressions=None
            )
            
            if avatar_frame is None:
                self.logger.warning("Failed to generate avatar frame")
                return None
            
            self.logger.debug(f"Generated avatar frame shape: {avatar_frame.shape}")
            
            # Encode frame for sending
            try:
                _, buffer = cv2.imencode('.jpg', avatar_frame)
                encoded_frame = base64.b64encode(buffer).decode('utf-8')
                self.logger.debug(f"Encoded frame size: {len(encoded_frame)} bytes")
                return f"data:image/jpeg;base64,{encoded_frame}"
            except Exception as e:
                self.logger.error(f"Error encoding frame: {e}")
                self.logger.error(traceback.format_exc())
                return None
            
        except Exception as e:
            self.logger.error(f"Error in process_frame: {e}")
            self.logger.error(traceback.format_exc())
            return None

    async def process_audio(self, audio_data: bytes):
        """Process an audio chunk."""
        if not self.is_running.is_set():
            return
            
        try:
            if self.voice2face:
                await self.voice2face.process_audio(audio_data)
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            self.logger.error(traceback.format_exc())

    def get_result(self, timeout=0.1):
        """Get the latest processed frame."""
        return self.prev_frame 