#!/usr/bin/env python3
"""
Launcher script for the real-time avatar system.
This script starts both the backend API server and the frontend web server.
"""

import os
import sys
import argparse
import subprocess
import time
import signal
import logging
import threading
import atexit
import webbrowser
import platform
import gdown

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to track processes
processes = []

# Define model URLs and output paths
MODEL_URLS = {
    'stylegan3_t.pt': 'https://drive.google.com/uc?id=1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK',
    'wav2lip.pth': 'https://drive.google.com/uc?id=1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK'
}

MODEL_DIR = 'app/models'

def signal_handler(sig, frame):
    """Handle keyboard interrupt (Ctrl+C)."""
    logger.info("Shutting down avatar system...")
    stop_all_processes()
    sys.exit(0)

def stop_all_processes():
    """Stop all running processes."""
    for process in processes:
        if process and process.poll() is None:  # Check if process exists and is running
            logger.info(f"Terminating process PID {process.pid}")
            try:
                process.terminate()
                # Wait for a moment to give process time to terminate
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {process.pid} did not terminate gracefully, killing...")
                process.kill()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")

def start_api_server(port, debug=False, use_cpu=False):
    """Start the FastAPI backend server."""
    logger.info(f"Starting API server on port {port}...")
    
    env = os.environ.copy()
    if use_cpu:
        # Force CPU mode
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["USE_CPU"] = "1"
        logger.info("Running in CPU-only mode")
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.api.server:app", 
        "--host", "0.0.0.0", 
        "--port", str(port)
    ]
    
    if debug:
        cmd.append("--reload")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not debug else None,
            stderr=subprocess.PIPE if not debug else None,
            universal_newlines=True,
            env=env
        )
        processes.append(process)
        logger.info(f"API server started with PID {process.pid}")
        
        # Wait a moment to ensure the server starts
        time.sleep(2)
        
        return process
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return None

def start_frontend_server(port, no_browser=False):
    """Start the frontend web server."""
    logger.info(f"Starting frontend server on port {port}...")
    
    cmd = [
        sys.executable, 
        os.path.join("app", "frontend.py"),
        "--port", str(port),
    ]
    
    if no_browser:
        cmd.append("--no-browser")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        processes.append(process)
        logger.info(f"Frontend server started with PID {process.pid}")
        
        # Wait a moment to ensure the server starts
        time.sleep(1)
        
        return process
    except Exception as e:
        logger.error(f"Failed to start frontend server: {e}")
        return None

def download_models(dummy=False, force=False):
    """Run the script to download pre-trained models."""
    logger.info("Downloading pre-trained models...")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    for model_name, url in MODEL_URLS.items():
        output_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(output_path):
            logger.info(f"Downloading {model_name}...")
            try:
                gdown.download(url, output_path, quiet=False)
                logger.info(f"Successfully downloaded {model_name}")
            except Exception as e:
                logger.error(f"Error downloading {model_name}: {e}")
        else:
            logger.info(f"{model_name} already exists, skipping download.")

    if dummy:
        logger.info("Using dummy model files for development")
    
    if force:
        logger.info("Force re-download of models")
    
    try:
        process = subprocess.run(
            [sys.executable, "download_models.py"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.info("Models downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download models: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False
    
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    logger.info("Checking dependencies...")
    
    # Check if we're on macOS
    is_macos = platform.system() == "Darwin"
    req_file = "requirements-macos.txt" if is_macos else "requirements.txt"
    
    if not os.path.exists(req_file):
        if is_macos and os.path.exists("requirements.txt"):
            logger.warning(f"No macOS-specific requirements file found. Using general requirements file.")
            req_file = "requirements.txt"
        else:
            logger.error(f"Requirements file '{req_file}' not found.")
            return False
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.info(f"Dependencies installed from {req_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def log_monitor(process, prefix):
    """Monitor and log process output."""
    if process.stdout is None or process.stderr is None:
        return
        
    while process.poll() is None:
        # Read stdout
        line = process.stdout.readline()
        if line:
            logger.info(f"{prefix}: {line.strip()}")
        
        # Read stderr
        err_line = process.stderr.readline()
        if err_line:
            logger.error(f"{prefix} ERR: {err_line.strip()}")

def open_browser(port):
    """Open the browser after a short delay."""
    time.sleep(3)  # Give the servers time to start
    url = f"http://localhost:{port}"
    logger.info(f"Opening browser at {url}")
    webbrowser.open(url)

def check_cuda_availability():
    """Check if CUDA is available for PyTorch."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. Running in CPU mode (this will be slow).")
        return cuda_available
    except ImportError:
        logger.warning("PyTorch not installed yet. Cannot check CUDA availability.")
        return False
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-Time Avatar System Launcher")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--frontend-port", type=int, default=8080, help="Port for the frontend server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading models")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--dummy-models", action="store_true", help="Use dummy model files for development")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of models")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode even if CUDA is available")
    
    args = parser.parse_args()
    
    # Register cleanup handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(stop_all_processes)
    
    # Check system info
    system_info = platform.system()
    logger.info(f"Running on {system_info} {platform.release()}")
    
    # Check if we need to use CPU
    use_cpu = args.cpu
    if not use_cpu and not check_cuda_availability():
        use_cpu = True
        logger.info("Defaulting to CPU mode")
    
    # Install dependencies if needed
    if not args.skip_deps:
        if not check_dependencies():
            logger.error("Failed to install required dependencies. Exiting.")
            sys.exit(1)
    
    # Create necessary directories
    os.makedirs(os.path.join("app", "models"), exist_ok=True)
    
    # Download models if needed
    if not args.skip_download:
        if not download_models(args.dummy_models, args.force_download):
            logger.warning("Model download incomplete. Some features may not work.")
            # Continue anyway with warning
    
    # Start the API server
    api_process = start_api_server(args.api_port, args.debug, use_cpu)
    if not api_process:
        logger.error("Failed to start API server. Exiting.")
        sys.exit(1)
    
    # Start the frontend server
    frontend_process = start_frontend_server(args.frontend_port, args.no_browser)
    if not frontend_process:
        logger.error("Failed to start frontend server. Exiting.")
        sys.exit(1)
    
    # Open browser if requested
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.frontend_port,), daemon=True).start()
    
    # Monitor server outputs
    if not args.debug:  # Only monitor logs if not in debug mode
        api_monitor = threading.Thread(target=log_monitor, args=(api_process, "API"), daemon=True)
        frontend_monitor = threading.Thread(target=log_monitor, args=(frontend_process, "Frontend"), daemon=True)
        api_monitor.start()
        frontend_monitor.start()
    
    logger.info(f"Real-Time Avatar System started")
    logger.info(f"API server: http://localhost:{args.api_port}")
    logger.info(f"Frontend: http://localhost:{args.frontend_port}")
    logger.info("Press Ctrl+C to stop")
    
    # Keep the main thread alive
    try:
        while True:
            # Check if any process has died
            if api_process.poll() is not None:
                logger.error("API server has stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                logger.error("Frontend server has stopped unexpectedly")
                break
                
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down avatar system...")
    finally:
        stop_all_processes()

if __name__ == "__main__":
    main() 