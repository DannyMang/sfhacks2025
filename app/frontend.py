#!/usr/bin/env python3
"""
Simple web server for the avatar system frontend.
"""

import os
import sys
import argparse
import http.server
import socketserver
import webbrowser
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create HTML content for the frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Avatar System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        header {
            background-color: #1a1a1a;
            width: 100%;
            padding: 1rem 0;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        h1 {
            margin: 0;
            color: #bb86fc;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            padding: 2rem;
            max-width: 1200px;
            width: 100%;
        }
        .video-container {
            flex: 1;
            min-width: 300px;
            max-width: 600px;
            background-color: #1e1e1e;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .video-container h2 {
            background-color: #2d2d2d;
            margin: 0;
            padding: 0.8rem;
            text-align: center;
            color: #bb86fc;
            font-size: 1.2rem;
        }
        video, #avatar-output {
            width: 100%;
            height: 400px;
            background-color: #000;
            object-fit: cover;
        }
        .controls {
            background-color: #1e1e1e;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            flex: 1;
            min-width: 300px;
            max-width: 600px;
        }
        .control-group {
            margin-bottom: 1.5rem;
        }
        .control-group h3 {
            color: #bb86fc;
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        button {
            background-color: #bb86fc;
            color: #000;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.2s;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        button:hover {
            background-color: #a370f7;
        }
        button:disabled {
            background-color: #6b6b6b;
            cursor: not-allowed;
        }
        .status {
            margin-top: 1rem;
            padding: 0.8rem;
            border-radius: 4px;
            background-color: #2d2d2d;
            overflow-y: auto;
            max-height: 120px;
        }
        .log {
            font-family: monospace;
            margin: 0;
            padding: 0.3rem 0;
            border-bottom: 1px solid #3a3a3a;
        }
        .log:last-child {
            border-bottom: none;
        }
        .metrics {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }
        .metric {
            background-color: #2d2d2d;
            border-radius: 4px;
            padding: 0.8rem;
            flex: 1;
            min-width: 120px;
            text-align: center;
        }
        .metric .value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #cf6679;
            margin-top: 0.5rem;
        }
        footer {
            margin-top: auto;
            width: 100%;
            background-color: #1a1a1a;
            text-align: center;
            padding: 1rem 0;
            font-size: 0.9rem;
            color: #999;
        }
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
                align-items: center;
            }
            .video-container, .controls {
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Real-Time Avatar System</h1>
    </header>
    
    <div class="container">
        <div class="video-container">
            <h2>Input Webcam</h2>
            <video id="webcam" autoplay playsinline></video>
        </div>
        
        <div class="video-container">
            <h2>Avatar Output</h2>
            <canvas id="avatar-output"></canvas>
        </div>
    </div>
    
    <div class="container">
        <div class="controls">
            <div class="control-group">
                <h3>Connection</h3>
                <button id="connect-btn">Connect WebSocket</button>
                <button id="disconnect-btn" disabled>Disconnect</button>
            </div>
            
            <div class="control-group">
                <h3>WebCam Controls</h3>
                <button id="start-cam-btn">Start Camera</button>
                <button id="stop-cam-btn" disabled>Stop Camera</button>
            </div>
            
            <div class="control-group">
                <h3>Audio Controls</h3>
                <button id="start-audio-btn">Start Microphone</button>
                <button id="stop-audio-btn" disabled>Stop Microphone</button>
            </div>
            
            <div class="status">
                <p class="log">System ready. Connect to WebSocket server to begin.</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div>Latency</div>
                    <div id="latency" class="value">0 ms</div>
                </div>
                <div class="metric">
                    <div>FPS</div>
                    <div id="fps" class="value">0</div>
                </div>
                <div class="metric">
                    <div>Status</div>
                    <div id="status" class="value">Idle</div>
                </div>
            </div>
        </div>
    </div>
    
    <footer>
        <p>Real-Time Avatar System for Live Interviews | Hackathon Project</p>
    </footer>

    <script>
        // DOM Elements
        const webcamVideo = document.getElementById('webcam');
        const avatarCanvas = document.getElementById('avatar-output');
        const avatarCtx = avatarCanvas.getContext('2d');
        
        const connectBtn = document.getElementById('connect-btn');
        const disconnectBtn = document.getElementById('disconnect-btn');
        const startCamBtn = document.getElementById('start-cam-btn');
        const stopCamBtn = document.getElementById('stop-cam-btn');
        const startAudioBtn = document.getElementById('start-audio-btn');
        const stopAudioBtn = document.getElementById('stop-audio-btn');
        
        const latencyEl = document.getElementById('latency');
        const fpsEl = document.getElementById('fps');
        const statusEl = document.getElementById('status');
        const statusLog = document.querySelector('.status');
        
        // Global variables
        let websocket = null;
        let mediaStream = null;
        let audioContext = null;
        let audioSource = null;
        let audioProcessor = null;
        let videoInterval = null;
        let lastFrameTime = 0;
        let framesReceived = 0;
        let fpsCounter = 0;
        let fpsInterval = null;
        
        // Initialize canvas
        avatarCanvas.width = 512;
        avatarCanvas.height = 512;
        avatarCtx.fillStyle = 'black';
        avatarCtx.fillRect(0, 0, avatarCanvas.width, avatarCanvas.height);
        
        // Helper function to log messages
        function logMessage(message) {
            const log = document.createElement('p');
            log.className = 'log';
            log.textContent = message;
            statusLog.appendChild(log);
            statusLog.scrollTop = statusLog.scrollHeight;
            
            // Limit number of messages
            if (statusLog.children.length > 20) {
                statusLog.removeChild(statusLog.children[0]);
            }
        }
        
        // Connect to WebSocket server
        function connectWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsHost = window.location.hostname;
            const wsPort = 8000; // FastAPI server port
            const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws/avatar`;
            
            logMessage(`Connecting to ${wsUrl}...`);
            
            try {
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function(event) {
                    logMessage('WebSocket connection established');
                    connectBtn.disabled = true;
                    disconnectBtn.disabled = false;
                    statusEl.textContent = 'Connected';
                    startFpsCounter();
                };
                
                websocket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.frame_data) {
                        renderAvatarFrame(data.frame_data);
                        updateMetrics(data);
                        framesReceived++;
                    }
                };
                
                websocket.onerror = function(error) {
                    logMessage(`WebSocket error: ${error}`);
                    statusEl.textContent = 'Error';
                };
                
                websocket.onclose = function(event) {
                    logMessage('WebSocket connection closed');
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                    statusEl.textContent = 'Disconnected';
                    stopFpsCounter();
                };
            } catch (error) {
                logMessage(`Error connecting to WebSocket: ${error.message}`);
            }
        }
        
        // Start webcam stream
        async function startWebcam() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 }
                    } 
                });
                
                webcamVideo.srcObject = mediaStream;
                startCamBtn.disabled = true;
                stopCamBtn.disabled = false;
                
                // Start sending frames
                startSendingFrames();
                
                logMessage('Webcam started');
            } catch (error) {
                logMessage(`Error accessing webcam: ${error.message}`);
            }
        }
        
        // Stop webcam stream
        function stopWebcam() {
            if (mediaStream) {
                // Stop sending frames
                stopSendingFrames();
                
                // Stop all tracks
                mediaStream.getTracks().forEach(track => track.stop());
                webcamVideo.srcObject = null;
                mediaStream = null;
                
                startCamBtn.disabled = false;
                stopCamBtn.disabled = true;
                
                logMessage('Webcam stopped');
            }
        }
        
        // Start audio processing
        async function startAudio() {
            try {
                // Get audio stream
                const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // Create audio context
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                
                // Create source node
                audioSource = audioContext.createMediaStreamSource(audioStream);
                
                // Create script processor for sampling audio
                const bufferSize = 4096;
                audioProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
                
                // Connect nodes
                audioSource.connect(audioProcessor);
                audioProcessor.connect(audioContext.destination);
                
                // Process audio data
                audioProcessor.onaudioprocess = function(event) {
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        // Get audio data
                        const inputData = event.inputBuffer.getChannelData(0);
                        
                        // Send to server
                        const audio_data = btoa(String.fromCharCode.apply(null, new Uint8Array(inputData.buffer)));
                        
                        websocket.send(JSON.stringify({
                            audio_data: audio_data,
                            sample_rate: audioContext.sampleRate
                        }));
                    }
                };
                
                startAudioBtn.disabled = true;
                stopAudioBtn.disabled = false;
                
                logMessage('Audio started');
            } catch (error) {
                logMessage(`Error accessing microphone: ${error.message}`);
            }
        }
        
        // Stop audio processing
        function stopAudio() {
            if (audioContext) {
                if (audioProcessor) {
                    audioProcessor.disconnect();
                    audioProcessor = null;
                }
                
                if (audioSource) {
                    audioSource.disconnect();
                    audioSource = null;
                }
                
                audioContext.close();
                audioContext = null;
                
                startAudioBtn.disabled = false;
                stopAudioBtn.disabled = true;
                
                logMessage('Audio stopped');
            }
        }
        
        // Start sending frames to server
        function startSendingFrames() {
            if (videoInterval) return;
            
            videoInterval = setInterval(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN && mediaStream) {
                    // Create offscreen canvas to get webcam frame
                    const canvas = document.createElement('canvas');
                    canvas.width = webcamVideo.videoWidth;
                    canvas.height = webcamVideo.videoHeight;
                    const ctx = canvas.getContext('2d');
                    
                    // Draw video frame to canvas
                    ctx.drawImage(webcamVideo, 0, 0);
                    
                    // Convert to base64
                    const imageData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                    
                    // Send to server
                    websocket.send(JSON.stringify({
                        frame_data: imageData,
                        timestamp: Date.now()
                    }));
                }
            }, 33); // ~30 FPS (1000/30)
        }
        
        // Stop sending frames
        function stopSendingFrames() {
            if (videoInterval) {
                clearInterval(videoInterval);
                videoInterval = null;
            }
        }
        
        // Render avatar frame
        function renderAvatarFrame(base64Data) {
            const img = new Image();
            img.onload = function() {
                avatarCtx.drawImage(img, 0, 0, avatarCanvas.width, avatarCanvas.height);
            };
            img.src = 'data:image/jpeg;base64,' + base64Data;
        }
        
        // Update metrics
        function updateMetrics(data) {
            if (data.latency) {
                latencyEl.textContent = `${(data.latency * 1000).toFixed(1)} ms`;
            }
        }
        
        // FPS counter
        function startFpsCounter() {
            if (fpsInterval) return;
            
            fpsInterval = setInterval(() => {
                fpsEl.textContent = `${framesReceived}`;
                framesReceived = 0;
            }, 1000);
        }
        
        // Stop FPS counter
        function stopFpsCounter() {
            if (fpsInterval) {
                clearInterval(fpsInterval);
                fpsInterval = null;
                fpsEl.textContent = '0';
            }
        }
        
        // Disconnect WebSocket
        function disconnectWebSocket() {
            if (websocket) {
                websocket.close();
                websocket = null;
                
                // Reset UI
                connectBtn.disabled = false;
                disconnectBtn.disabled = true;
                
                // Stop sending frames
                stopSendingFrames();
                
                logMessage('WebSocket disconnected');
            }
        }
        
        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            stopWebcam();
            stopAudio();
            disconnectWebSocket();
        });
        
        // Button event listeners
        connectBtn.addEventListener('click', connectWebSocket);
        disconnectBtn.addEventListener('click', disconnectWebSocket);
        startCamBtn.addEventListener('click', startWebcam);
        stopCamBtn.addEventListener('click', stopWebcam);
        startAudioBtn.addEventListener('click', startAudio);
        stopAudioBtn.addEventListener('click', stopAudio);
    </script>
</body>
</html>
"""

class HttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode())
        else:
            super().do_GET()

def main():
    parser = argparse.ArgumentParser(description='Web frontend for the real-time avatar system')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the web server on')
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser automatically')
    args = parser.parse_args()
    
    # Create web server
    handler = HttpRequestHandler
    httpd = socketserver.TCPServer(("", args.port), handler)
    
    logger.info(f"Web server running at http://localhost:{args.port}")
    
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        httpd.server_close()

if __name__ == '__main__':
    main() 