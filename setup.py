#!/usr/bin/env python3
"""
Setup script for the Real-Time Avatar System.
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages

# Clone StyleGAN3 if not already present
stylegan3_dir = os.path.join('app', 'models', 'stylegan3')
if not os.path.exists(stylegan3_dir):
    print("Cloning StyleGAN3 repository...")
    subprocess.check_call([
        'git', 'clone', 'https://github.com/NVlabs/stylegan3.git', stylegan3_dir
    ])

# Add StyleGAN3 to Python path
sys.path.append(stylegan3_dir)

setup(
    name="avatar-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.4.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "websockets>=11.0.0",
        "python-multipart>=0.0.5",
        "aiofiles>=0.8.0",
        "onnx>=1.13.0",
        "onnxruntime-gpu>=1.14.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.0",
        "ninja",
    ],
) 