#!/usr/bin/env python3
"""
Utility script for training StyleGAN3 models on Vast.ai.

This script handles:
1. Setting up the environment on a Vast.ai instance
2. Preparing data for training
3. Fine-tuning StyleGAN3 on custom data
4. Exporting the model for use in the avatar system

Usage:
    python vast_training.py --data_dir path/to/images --output_dir path/to/output
"""

import os
import sys
import argparse
import subprocess
import time
import logging
from pathlib import Path
import shutil
import urllib.request
import zipfile
import tarfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
STYLEGAN3_REPO = "https://github.com/NVlabs/stylegan3.git"
STYLEGAN3_BRANCH = "main"
VAST_SETUP_COMMANDS = [
    "apt-get update",
    "apt-get install -y git wget unzip python3-pip",
    "pip install torch==2.0.1 torchvision==0.15.2 numpy==1.24.3 opencv-python==4.7.0.72",
    "pip install ninja matplotlib tqdm pillow==9.5.0 imageio==2.31.1 pyspng==0.1.1"
]

def run_command(cmd, shell=False):
    """Run a shell command and log output."""
    logger.info(f"Running command: {cmd}")
    
    if shell:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
    else:
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
    
    stdout, stderr = process.communicate()
    
    if stdout:
        logger.info(stdout)
    
    if stderr:
        if process.returncode != 0:
            logger.error(stderr)
        else:
            logger.info(stderr)
    
    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}")
        return False
    
    return True

def setup_vast_environment():
    """Set up the environment on Vast.ai instance."""
    logger.info("Setting up environment on Vast.ai...")
    
    for cmd in VAST_SETUP_COMMANDS:
        if not run_command(cmd, shell=True):
            logger.error(f"Failed to run command: {cmd}")
            return False
    
    return True

def clone_stylegan3_repo():
    """Clone the StyleGAN3 repository."""
    logger.info(f"Cloning StyleGAN3 repository from {STYLEGAN3_REPO}...")
    
    # Clone the repository
    cmd = f"git clone -b {STYLEGAN3_BRANCH} {STYLEGAN3_REPO}"
    if not run_command(cmd, shell=True):
        logger.error("Failed to clone StyleGAN3 repository")
        return False
    
    return True

def prepare_training_data(data_dir, output_dir):
    """Prepare training data for StyleGAN3."""
    logger.info(f"Preparing training data from {data_dir} to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the dataset_tool.py script from StyleGAN3
    cmd = f"python stylegan3/dataset_tool.py --source={data_dir} --dest={output_dir}/dataset.zip"
    if not run_command(cmd, shell=True):
        logger.error("Failed to prepare training data")
        return False
    
    return True

def train_stylegan3(dataset_path, output_dir, resume_pkl=None, gpus=1, batch_size=32, kimg=10000):
    """Train StyleGAN3 on the prepared dataset."""
    logger.info(f"Training StyleGAN3 on {dataset_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the training command
    cmd = [
        "python", "stylegan3/train.py",
        f"--outdir={output_dir}",
        f"--data={dataset_path}",
        f"--gpus={gpus}",
        f"--batch={batch_size}",
        f"--kimg={kimg}",
        "--cfg=stylegan3-t",
        "--cbase=16384",
        "--glr=0.0025",
        "--dlr=0.0025",
        "--gamma=8.0"
    ]
    
    if resume_pkl:
        cmd.append(f"--resume={resume_pkl}")
    
    # Convert list to string for shell execution
    cmd_str = " ".join(cmd)
    
    if not run_command(cmd_str, shell=True):
        logger.error("Failed to train StyleGAN3")
        return False
    
    return True

def export_model(checkpoint_path, output_path):
    """Export the trained model for use in the avatar system."""
    logger.info(f"Exporting model from {checkpoint_path} to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Copy the checkpoint file to the output path
    shutil.copy2(checkpoint_path, output_path)
    
    logger.info(f"Model exported to {output_path}")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="StyleGAN3 Training on Vast.ai")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for training results")
    parser.add_argument("--resume_pkl", type=str, help="Path to existing checkpoint to resume training")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--kimg", type=int, default=10000, help="Training duration in thousands of images")
    parser.add_argument("--skip_setup", action="store_true", help="Skip environment setup (use if already set up)")
    
    args = parser.parse_args()
    
    # Setup environment if needed
    if not args.skip_setup:
        if not setup_vast_environment():
            logger.error("Failed to set up environment")
            sys.exit(1)
        
        if not clone_stylegan3_repo():
            logger.error("Failed to clone StyleGAN3 repository")
            sys.exit(1)
    
    # Prepare training data
    dataset_path = os.path.join(args.output_dir, "dataset.zip")
    if not os.path.exists(dataset_path):
        if not prepare_training_data(args.data_dir, args.output_dir):
            logger.error("Failed to prepare training data")
            sys.exit(1)
    
    # Train StyleGAN3
    if not train_stylegan3(
        dataset_path, 
        args.output_dir, 
        resume_pkl=args.resume_pkl,
        gpus=args.gpus,
        batch_size=args.batch_size,
        kimg=args.kimg
    ):
        logger.error("Failed to train StyleGAN3")
        sys.exit(1)
    
    # Find the latest checkpoint
    network_dir = os.path.join(args.output_dir, "network-snapshot-final.pkl")
    if not os.path.exists(network_dir):
        # Try to find the latest snapshot
        network_dir = None
        for file in os.listdir(args.output_dir):
            if file.startswith("network-snapshot-") and file.endswith(".pkl"):
                if network_dir is None or file > network_dir:
                    network_dir = file
        
        if network_dir:
            network_dir = os.path.join(args.output_dir, network_dir)
    
    if not network_dir:
        logger.error("No checkpoint found in output directory")
        sys.exit(1)
    
    # Export model to app/models directory
    export_path = os.path.join("app", "models", "stylegan3_t.pt")
    if not export_model(network_dir, export_path):
        logger.error("Failed to export model")
        sys.exit(1)
    
    logger.info("Training completed successfully!")
    logger.info(f"Exported model to {export_path}")

if __name__ == "__main__":
    main() 