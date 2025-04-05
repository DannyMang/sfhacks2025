#!/usr/bin/env python3
"""
Launcher script for the Real-Time Avatar System.
Starts both the backend API server and the frontend web server.
"""

import os
import sys
import time
import signal
import logging
import subprocess
import argparse
from typing import Optional, List
import socket
from contextlib import closing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_free_port(start_port: int, max_attempts: int = 10) -> Optional[int]:
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    return None

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    # First check if packages are already installed
    try:
        import fastapi
        import uvicorn
        import torch
        import cv2
        import numpy
        import base64
        import asyncio
        logger.info("All required packages are already installed")
        return
    except ImportError as e:
        logger.info(f"Some packages missing: {e}")
    
    try:
        # Try to install from requirements-macos.txt first
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements-macos.txt"],
            check=True
        )
        logger.info("Dependencies installed from requirements-macos.txt")
    except subprocess.CalledProcessError:
        try:
            # Fall back to requirements.txt if macos version fails
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True
            )
            logger.info("Dependencies installed from requirements.txt")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            sys.exit(1)

def download_models(use_dummy: bool = False):
    """Download pre-trained models or use dummy models."""
    logger.info("Downloading pre-trained models...")
    try:
        if use_dummy:
            logger.info("Using dummy model files for development")
            # Create dummy model files if they don't exist
            model_dir = "app/models"
            os.makedirs(model_dir, exist_ok=True)
            dummy_files = ["stylegan3_t.pt", "wav2lip.pth", "first_order_model.pth"]
            for file in dummy_files:
                file_path = os.path.join(model_dir, file)
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write("Dummy model file for development")
        else:
            subprocess.run([sys.executable, "download_models.py"], check=True)
        logger.info("Models downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download models: {e}")
        sys.exit(1)

def start_api_server(port: int = 8000, use_cpu: bool = False) -> subprocess.Popen:
    """Start the FastAPI backend server."""
    logger.info(f"Starting API server on port {port}...")
    if use_cpu:
        logger.info("Running in CPU-only mode")
        cmd = [sys.executable, "app/api/server.py", "--port", str(port), "--cpu"]
    else:
        cmd = [sys.executable, "app/api/server.py", "--port", str(port)]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        logger.info(f"API server started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

def start_frontend_server(port: int = 8080) -> subprocess.Popen:
    """Start the frontend web server."""
    logger.info(f"Starting frontend server on port {port}...")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port)],
            cwd="app",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        logger.info(f"Frontend server started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start frontend server: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Launch the Real-Time Avatar System")
    parser.add_argument("--cpu", action="store_true", help="Run in CPU-only mode")
    parser.add_argument("--dummy-models", action="store_true", help="Use dummy models for development")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--frontend-port", type=int, default=8080, help="Port for the frontend server")
    args = parser.parse_args()

    # Find free ports if the specified ones are in use
    api_port = find_free_port(args.api_port)
    frontend_port = find_free_port(args.frontend_port)
    
    if api_port is None:
        logger.error(f"Could not find a free port for API server starting from {args.api_port}")
        sys.exit(1)
    if frontend_port is None:
        logger.error(f"Could not find a free port for frontend server starting from {args.frontend_port}")
        sys.exit(1)

    # Check and install dependencies
    check_dependencies()
    
    # Download or create dummy models
    download_models(args.dummy_models)
    
    # Start servers
    api_process = start_api_server(api_port, args.cpu)
    frontend_process = start_frontend_server(frontend_port)
    
    # Log server URLs
    logger.info("Real-Time Avatar System started")
    logger.info(f"API server: http://localhost:{api_port}")
    logger.info(f"Frontend: http://localhost:{frontend_port}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Monitor processes
        while True:
            # Check API server
            api_returncode = api_process.poll()
            if api_returncode is not None:
                logger.error(f"API server has stopped unexpectedly with return code {api_returncode}")
                break
            
            # Check frontend server
            frontend_returncode = frontend_process.poll()
            if frontend_returncode is not None:
                logger.error(f"Frontend server has stopped unexpectedly with return code {frontend_returncode}")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
    finally:
        # Terminate processes
        for process in [api_process, frontend_process]:
            if process.poll() is None:  # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        logger.info("Servers stopped")

if __name__ == "__main__":
    main() 