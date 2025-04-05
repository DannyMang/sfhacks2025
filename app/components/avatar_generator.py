#!/usr/bin/env python3
"""
Avatar generator using StyleGAN3 for real-time avatar generation.
Includes optimizations for low-latency inference.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
from PIL import Image
import logging
import traceback
import pickle
import sys

class AvatarGenerator:
    def __init__(self, model_path, device='cpu'):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AvatarGenerator with StyleGAN3")
        
        # Get absolute paths
        model_dir = os.path.dirname(os.path.abspath(model_path))
        stylegan3_dir = os.path.join(model_dir, 'stylegan3')
        
        # Add to Python path
        if stylegan3_dir not in sys.path:
            sys.path.insert(0, stylegan3_dir)
            self.logger.info(f"Added StyleGAN3 path: {stylegan3_dir}")
        
        # Load StyleGAN3 model
        try:
            import dnnlib
            import legacy
            
            self.logger.info("Loading StyleGAN3 model...")
            with dnnlib.util.open_url(model_path) as f:
                self.model = legacy.load_network_pkl(f)['G_ema'].to(device)
            self.model.eval()
            self.device = device
            
            # Generate random latent vector for initialization
            self.latent = torch.randn(1, self.model.z_dim).to(device)
            # Add class conditioning vector (usually zeros for unconditional models)
            self.class_vector = torch.zeros([1, self.model.c_dim], device=device)
            self.logger.info("StyleGAN3 model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load StyleGAN3: {e}")
            self.logger.error(traceback.format_exc())
            self.model = None

        self.frame_count = 0

    def generate_frame(self, head_pose=None, expressions=None):
        if self.model is None:
            return None
            
        try:
            self.logger.info("Generating StyleGAN3 frame")
            
            with torch.no_grad():
                # Update latent vector based on head pose if provided
                if head_pose is not None:
                    # Simple mapping of head pose to latent space
                    pose_scale = 0.1
                    
                    # Create a pose tensor of the same size as latent
                    pose_tensor = torch.zeros_like(self.latent)
                    
                    if isinstance(head_pose, list):
                        # Map the first 3 dimensions of latent space to pose
                        pose_tensor[0, :3] = torch.tensor([
                            head_pose[0] * pose_scale,
                            head_pose[1] * pose_scale,
                            head_pose[2] * pose_scale
                        ]).to(self.device)
                    else:
                        pose_tensor[0, :3] = torch.tensor([
                            head_pose['pitch'] * pose_scale,
                            head_pose['yaw'] * pose_scale,
                            head_pose['roll'] * pose_scale
                        ]).to(self.device)
                    
                    # Add the pose influence to latent
                    self.latent = self.latent + pose_tensor
                
                # Generate image with both z and c inputs
                img = self.model(self.latent, self.class_vector)
                
                # Convert to numpy array
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = img[0].cpu().numpy()
                
                self.frame_count += 1
                self.logger.info(f"Generated StyleGAN3 frame {self.frame_count}")
                return img
                
        except Exception as e:
            self.logger.error(f"Error generating StyleGAN3 frame: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def apply_frame_interpolation(self, prev_frame, current_frame, factor=0.5):
        """
        Apply frame interpolation to smooth animation.
        
        Args:
            prev_frame: Previous frame
            current_frame: Current frame
            factor: Interpolation factor (0-1)
            
        Returns:
            Interpolated frame
        """
        if prev_frame is None or current_frame is None:
            return current_frame
        
        # Simple linear interpolation for hackathon
        # In production, you'd use optical flow based methods like RIFE
        return cv2.addWeighted(prev_frame, 1-factor, current_frame, factor, 0)
    
    def export_onnx(self, onnx_path):
        """
        Export the model to ONNX format for deployment.
        
        Args:
            onnx_path: Path to save the ONNX model
        """
        try:
            # Check if we can import onnx
            import onnx
            
            # Create a dummy input
            dummy_input = torch.randn(1, self.z_dim, device=self.device)
            
            # Export the model
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
            
            print(f"Model exported to {onnx_path}")
            return True
        except ImportError:
            print("ONNX not available, skipping export")
            return False
    
    def release(self):
        """Release resources."""
        # Clean up any resources
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 