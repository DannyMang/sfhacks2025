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
from app.utils.face_detector import FaceDetector
from app.components.avatar_generator import AvatarGenerator
from app.components.voice2face import Voice2Face

class AvatarPipeline:
    def __init__(self, models_dir, device='cuda'):
        """
        Initialize the avatar pipeline.
        
        Args:
            models_dir: Directory containing model files
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.models_dir = models_dir
        self.device = device
        
        # Load model paths
        self.stylegan_model_path = os.path.join(models_dir, 'stylegan3_t.pt')
        self.wav2lip_model_path = os.path.join(models_dir, 'wav2lip.pth')
        
        # Initialize components
        self.face_detector = None
        self.avatar_generator = None
        self.voice2face = None
        
        # State variables
        self.prev_frame = None
        self.last_generated_frame_time = 0
        self.frame_interval = 1.0 / 15.0  # Target 15 FPS for GAN generation
        self.interpolation_factor = 0.5   # Blend factor for frame interpolation
        
        # For face detection
        self.current_head_pose = None
        self.current_facial_features = None
        
        # Async processing queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Performance metrics
        self.pipeline_latency = 0
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Initialize pipeline
        self.initialize_pipeline()
        
        # Worker threads
        self.is_running = False
        self.worker_threads = []
    
    def initialize_pipeline(self):
        """Initialize all components of the pipeline."""
        print("Initializing avatar pipeline...")
        
        # Initialize face detector
        self.face_detector = FaceDetector(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize avatar generator
        print(f"Loading StyleGAN model from {self.stylegan_model_path}")
        self.avatar_generator = AvatarGenerator(
            model_path=self.stylegan_model_path,
            device=self.device
        )
        # Optimize for inference
        self.avatar_generator.optimize_for_inference()
        
        # Initialize Voice2Face module
        print(f"Loading Wav2Lip model from {self.wav2lip_model_path}")
        self.voice2face = Voice2Face(
            model_path=self.wav2lip_model_path,
            device=self.device
        )
        # Optimize for inference
        self.voice2face.optimize_for_inference()
        
        print("Avatar pipeline initialized")
    
    def start(self):
        """Start the avatar pipeline processing threads."""
        if self.is_running:
            print("Pipeline already running")
            return
        
        self.is_running = True
        
        # Start worker threads
        frame_thread = threading.Thread(target=self._frame_processing_worker)
        frame_thread.daemon = True
        frame_thread.start()
        self.worker_threads.append(frame_thread)
        
        audio_thread = threading.Thread(target=self._audio_processing_worker)
        audio_thread.daemon = True
        audio_thread.start()
        self.worker_threads.append(audio_thread)
        
        print("Avatar pipeline started")
    
    def stop(self):
        """Stop the avatar pipeline."""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=1.0)
        
        # Release resources
        if self.face_detector:
            self.face_detector.release()
        
        if self.avatar_generator:
            self.avatar_generator.release()
        
        if self.voice2face:
            self.voice2face.release()
        
        print("Avatar pipeline stopped")
    
    def _frame_processing_worker(self):
        """Worker thread for processing input frames."""
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                start_time = time.time()
                
                input_frame, timestamp = frame_data
                
                # Detect face landmarks
                landmarks_list, face_rect, _ = self.face_detector.detect_face_landmarks(input_frame)
                
                # If face detected, extract pose and features
                if landmarks_list:
                    # Get head pose
                    _, _, euler_angles = self.face_detector.get_head_pose(landmarks_list, input_frame.shape)
                    self.current_head_pose = euler_angles if euler_angles else self.current_head_pose
                    
                    # Get facial features
                    facial_features = self.face_detector.get_facial_features(landmarks_list)
                    self.current_facial_features = facial_features if facial_features else self.current_facial_features
                
                # Check if it's time to generate a new frame with StyleGAN
                current_time = time.time()
                if current_time - self.last_generated_frame_time >= self.frame_interval:
                    # Generate avatar frame
                    expressions = None
                    if self.current_facial_features:
                        # Simple expression mapping - in a real system this would be more sophisticated
                        expressions = {
                            'smile': 0.0,  # Placeholder
                            'eye_open': 1.0,  # Placeholder
                            'brow_up': 0.0,  # Placeholder
                        }
                    
                    # Generate the frame
                    avatar_frame = self.avatar_generator.generate_frame(self.current_head_pose, expressions)
                    
                    # Apply lip sync if we have voice data
                    lip_keypoints = self.voice2face.predict_lip_shapes()
                    if lip_keypoints is not None:
                        avatar_frame = self.voice2face.apply_lip_shapes_to_face(avatar_frame, lip_keypoints)
                    
                    # Apply frame interpolation for smooth animation
                    if self.prev_frame is not None:
                        avatar_frame = self.avatar_generator.apply_frame_interpolation(
                            self.prev_frame, avatar_frame, self.interpolation_factor)
                    
                    # Save for next frame
                    self.prev_frame = avatar_frame.copy()
                    self.last_generated_frame_time = current_time
                else:
                    # Reuse previous frame if we have one
                    avatar_frame = self.prev_frame if self.prev_frame is not None else np.zeros((512, 512, 3), dtype=np.uint8)
                
                # Calculate pipeline latency
                self.pipeline_latency = time.time() - start_time
                
                # Calculate FPS
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                
                # Add latency and FPS overlay
                cv2.putText(avatar_frame, f"Latency: {self.pipeline_latency*1000:.1f}ms", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(avatar_frame, f"FPS: {self.fps}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add to result queue
                self.result_queue.put((avatar_frame, self.pipeline_latency))
                
                # Signal task completion
                self.frame_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in frame processing: {e}")
    
    def _audio_processing_worker(self):
        """Worker thread for processing audio chunks."""
        while self.is_running:
            try:
                # Get audio from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is None:
                    continue
                
                audio_chunk, sr = audio_data
                
                # Process audio for lip sync
                if self.voice2face:
                    self.voice2face.update_audio_buffer(audio_chunk, sr)
                
                # Signal task completion
                self.audio_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in audio processing: {e}")
    
    def process_frame(self, frame):
        """
        Process a video frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            In async mode, returns immediately. Results can be retrieved with get_result().
        """
        if not self.is_running:
            print("Pipeline not running. Call start() first.")
            return None
        
        # Add to frame queue
        try:
            self.frame_queue.put((frame, time.time()), block=False)
        except queue.Full:
            print("Warning: Frame queue full, dropping frame")
    
    def process_audio(self, audio_chunk, sample_rate):
        """
        Process an audio chunk.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            sample_rate: Sample rate of the audio
        """
        if not self.is_running:
            print("Pipeline not running. Call start() first.")
            return
        
        # Add to audio queue
        try:
            self.audio_queue.put((audio_chunk, sample_rate), block=False)
        except queue.Full:
            # Less critical to drop audio frames
            pass
    
    def get_result(self, timeout=0.1):
        """
        Get the processed result.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (frame, latency) or None if no result available
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None 