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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to track processes
processes = []

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

def start_api_server(port, debug=False):
    """Start the FastAPI backend server."""
    logger.info(f"Starting API server on port {port}...")
    
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
            universal_newlines=True
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

def download_models():
    """Run the script to download pre-trained models."""
    logger.info("Downloading pre-trained models...")
    
    cmd = [sys.executable, "download_models.py"]
    
    try:
        process = subprocess.run(
            cmd,
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

def log_monitor(process, prefix):
    """Monitor and log process output."""
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

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-Time Avatar System Launcher")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--frontend-port", type=int, default=8080, help="Port for the frontend server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading models")
    
    args = parser.parse_args()
    
    # Register cleanup handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(stop_all_processes)
    
    # Download models if needed
    if not args.skip_download:
        if not download_models():
            logger.error("Failed to download required models. Exiting.")
            sys.exit(1)
    
    # Start the API server
    api_process = start_api_server(args.api_port, args.debug)
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