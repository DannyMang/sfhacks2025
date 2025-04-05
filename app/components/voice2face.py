#!/usr/bin/env python3
"""
Voice2Face module for mapping speech audio to lip movements.
Integrates with Wav2Lip to provide real-time lip synchronization.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
import librosa
import logging
import asyncio
import traceback

class Voice2Face:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the Voice2Face module for lip sync.
        
        Args:
            model_path: Path to the Wav2Lip model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device
        self.model = None
        self.use_tensorrt = False
        self.tensorrt_model_path = None
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.mel_window_length = 25  # in milliseconds
        self.mel_window_step = 10    # in milliseconds
        self.mel_n_channels = 80
        
        # Buffer to store audio history
        self.audio_buffer = np.array([])
        self.buffer_duration = 0.2  # seconds
        
        # For performance tracking
        self.inference_time = 0
        
        # Check for TensorRT model
        if device == 'cuda' and torch.cuda.is_available():
            model_dir = os.path.dirname(os.path.abspath(model_path))
            tensorrt_path = os.path.join(model_dir, 'wav2lip_tensorrt.engine')
            if os.path.exists(tensorrt_path):
                self.tensorrt_model_path = tensorrt_path
                self.logger.info(f"Found TensorRT model at {tensorrt_path}")
                self.use_tensorrt = True
        
        # Load model
        self.load_model()
        
        # Optimize for inference if using CUDA
        if device == 'cuda' and torch.cuda.is_available() and self.model is not None:
            self.optimize_for_inference()
    
    def load_model(self):
        """Load the Wav2Lip model."""
        self.logger.info(f"Loading Wav2Lip model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            if self.use_tensorrt:
                self._load_tensorrt_model()
            else:
                self._load_wav2lip_model()
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _load_tensorrt_model(self):
        """Load TensorRT optimized Wav2Lip model."""
        try:
            # Try to import TensorRT
            import tensorrt as trt
            from torch2trt import TRTModule
            
            # Load TensorRT model
            self.model = TRTModule()
            self.model.load_state_dict(torch.load(self.tensorrt_model_path))
            self.logger.info("TensorRT Wav2Lip model loaded successfully")
            
        except ImportError as e:
            self.logger.error(f"TensorRT import failed: {e}")
            self.use_tensorrt = False
            self._load_wav2lip_model()
            
        except Exception as e:
            self.logger.error(f"Failed to load TensorRT model: {e}")
            self.logger.error(traceback.format_exc())
            self.use_tensorrt = False
            self._load_wav2lip_model()
    
    def _load_wav2lip_model(self):
        """Load original Wav2Lip model or dummy model for development."""
        # In a real implementation, you'd load the actual Wav2Lip model
        # For this hackathon version, we'll use a dummy model
        class DummyWav2Lip(nn.Module):
            def __init__(self):
                super().__init__()
                # Simple mel spectrogram to lip keypoints mapping
                self.audio_encoder = nn.Sequential(
                    nn.Conv1d(80, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)
                )
                
                self.lip_predictor = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 20)  # 10 keypoints (x,y)
                )
            
            def forward(self, mel_spectrogram):
                # Reshape and process
                x = self.audio_encoder(mel_spectrogram)
                x = x.view(x.size(0), -1)
                lip_keypoints = self.lip_predictor(x)
                return lip_keypoints
        
        self.model = DummyWav2Lip().to(self.device)
        if self.model_path.endswith('.pth'):
            # Pretend to load weights
            self.logger.info("Initialized dummy Wav2Lip model for development")
        else:
            raise ValueError("Unsupported model format")
    
    def optimize_for_inference(self):
        """Optimize the model for inference."""
        if self.use_tensorrt:
            self.logger.info("Model already using TensorRT, skipping additional optimization")
            return
            
        self.logger.info("Optimizing Voice2Face model for inference...")
        
        if self.device == 'cuda' and torch.cuda.is_available():
            # Freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Use torch.jit.script to optimize
            try:
                self.model = torch.jit.script(self.model)
                self.logger.info("Model optimized with torch.jit.script")
            except Exception as e:
                self.logger.warning(f"Warning: Failed to optimize model with JIT: {e}")
        else:
            self.logger.info("CUDA not available, skipping optimization")
    
    def _extract_mel_features(self, audio, sr=16000):
        """
        Extract Mel-spectrogram features from audio.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            mel_features: Mel-spectrogram features
        """
        # Convert to correct sample rate if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Extract mel spectrogram
        n_fft = int(self.mel_window_length * self.sample_rate / 1000)
        hop_length = int(self.mel_window_step * self.sample_rate / 1000)
        
        mel = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=self.mel_n_channels
        )
        
        # Convert to log scale
        mel = np.log(np.maximum(1e-5, mel))
        
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        
        return mel
    
    def update_audio_buffer(self, audio_chunk, sr):
        """
        Update the audio buffer with new audio chunk.
        
        Args:
            audio_chunk: New audio chunk as numpy array
            sr: Sample rate
        """
        # Resample if needed
        if sr != self.sample_rate:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=self.sample_rate)
        
        # Append to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
        
        # Trim buffer to keep only the most recent audio
        max_buffer_size = int(self.buffer_duration * self.sample_rate)
        if len(self.audio_buffer) > max_buffer_size:
            self.audio_buffer = self.audio_buffer[-max_buffer_size:]
    
    async def process_audio(self, audio_data):
        """
        Process audio data to extract lip movements.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            lip_keypoints: Predicted lip keypoints
        """
        try:
            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
            
            # Update audio buffer
            self.update_audio_buffer(audio_chunk, self.sample_rate)
            
            # Extract mel features
            mel_features = self._extract_mel_features(self.audio_buffer)
            
            # Predict lip keypoints
            lip_keypoints = await self.predict_lip_keypoints(mel_features)
            
            return lip_keypoints
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    async def predict_lip_keypoints(self, mel_features):
        """
        Predict lip keypoints from mel features.
        
        Args:
            mel_features: Mel spectrogram features
            
        Returns:
            lip_keypoints: Predicted lip keypoints
        """
        start_time = time.time()
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_features).unsqueeze(0).to(self.device)
        
        # Predict lip keypoints
        with torch.no_grad():
            lip_keypoints = self.model(mel_tensor)
        
        # Convert to numpy
        lip_keypoints = lip_keypoints.cpu().numpy().reshape(-1, 2)
        
        self.inference_time = time.time() - start_time
        self.logger.debug(f"Lip keypoint prediction time: {self.inference_time:.3f}s")
        
        return lip_keypoints
    
    def apply_lip_shapes_to_face(self, face_image, lip_keypoints, face_landmarks=None):
        """
        Apply predicted lip shapes to a face image.
        
        Args:
            face_image: Face image as numpy array
            lip_keypoints: Predicted lip keypoints
            face_landmarks: Full face landmarks (optional)
            
        Returns:
            Modified face image with updated lip shapes
        """
        # This is a simplified placeholder implementation
        # In a real system, you would use these keypoints to:
        # 1. Either directly modify the StyleGAN latent space
        # 2. Or apply warping to the generated image
        
        # For this hackathon demo, we'll just visualize the keypoints
        result = face_image.copy()
        
        if lip_keypoints is not None:
            # Scale keypoints to image size
            h, w = face_image.shape[:2]
            scaled_keypoints = []
            for kp in lip_keypoints:
                x = int((kp[0] + 1) * w / 2)  # Assuming keypoints are in [-1, 1] range
                y = int((kp[1] + 1) * h / 2)
                scaled_keypoints.append((x, y))
            
            # Draw keypoints
            for point in scaled_keypoints:
                cv2.circle(result, point, 2, (0, 0, 255), -1)
            
            # Connect keypoints to form a mouth shape
            if len(scaled_keypoints) >= 8:  # Assuming we have enough keypoints
                # Upper lip
                for i in range(5):
                    cv2.line(result, scaled_keypoints[i], scaled_keypoints[i+1], (0, 0, 255), 1)
                # Lower lip
                for i in range(5, 9):
                    cv2.line(result, scaled_keypoints[i], scaled_keypoints[i+1], (0, 0, 255), 1)
                # Close the mouth
                cv2.line(result, scaled_keypoints[0], scaled_keypoints[9], (0, 0, 255), 1)
                cv2.line(result, scaled_keypoints[5], scaled_keypoints[4], (0, 0, 255), 1)
        
        return result
    
    def extract_viseme_features(self, audio_chunk, sr=16000):
        """
        Extract viseme features from audio for more accurate lip sync.
        Visemes are visual counterparts to phonemes.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            sr: Sample rate
            
        Returns:
            viseme_features: Array of viseme features
        """
        # This is a simplified placeholder implementation
        # In a production system, you'd use a more sophisticated approach:
        # 1. Use speech recognition to get phonemes
        # 2. Map phonemes to visemes
        # 3. Time-align visemes with audio
        
        # For hackathon purposes, we'll just extract simplified features
        # based on audio energy in different frequency bands
        
        # Resample if needed
        if sr != self.sample_rate:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=self.sample_rate)
        
        # Calculate short-time Fourier transform
        D = librosa.stft(audio_chunk)
        
        # Convert to power spectrogram
        S = np.abs(D)**2
        
        # Calculate energy in different frequency bands
        bands = [
            (0, 500),      # Low frequencies (vowels)
            (500, 2000),   # Mid frequencies (most consonants)
            (2000, 8000)   # High frequencies (sibilants)
        ]
        
        # Extract band energies
        band_energies = []
        for low, high in bands:
            low_bin = librosa.core.hz_to_fft_bin(low, sr=self.sample_rate, n_fft=D.shape[0]*2-2)
            high_bin = librosa.core.hz_to_fft_bin(high, sr=self.sample_rate, n_fft=D.shape[0]*2-2)
            band_energy = np.mean(S[low_bin:high_bin], axis=0)
            band_energies.append(band_energy)
        
        # Normalize
        band_energies = np.array(band_energies)
        if band_energies.max() > 0:
            band_energies = band_energies / band_energies.max()
        
        # Take the mean over time
        viseme_features = np.mean(band_energies, axis=1)
        
        return viseme_features
    
    def release(self):
        """Release resources."""
        # Clean up
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 