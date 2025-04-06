import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import WebcamView from './components/WebcamView';
import AvatarView from './components/AvatarView';
import ControlPanel from './components/ControlPanel';
import CalibrationView from './components/CalibrationView';

function App() {
  const [connected, setConnected] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [audioActive, setAudioActive] = useState(false);
  const [latency, setLatency] = useState(0);
  const [fps, setFps] = useState(0);
  const [status, setStatus] = useState('Idle');
  const [logs, setLogs] = useState(['System ready. Connect to WebSocket server to begin.']);
  const [calibrated, setCalibrated] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  
  const websocketRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const audioSourceRef = useRef(null);
  const audioProcessorRef = useRef(null);
  const videoIntervalRef = useRef(null);
  const framesReceivedRef = useRef(0);
  const fpsIntervalRef = useRef(null);
  const webcamRef = useRef(null);
  const avatarCanvasRef = useRef(null);
  
  // Add a log message
  const addLog = (message) => {
    setLogs(prevLogs => {
      const newLogs = [...prevLogs, message];
      // Keep only the last 20 logs
      return newLogs.slice(-20);
    });
  };
  
  // Connect to WebSocket
  // Fixed WebSocket connection function
  const connectWebSocket = () => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.hostname;
    const wsPort = 8000; // FastAPI server port
    const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws`;
    
    addLog(`Connecting to ${wsUrl}...`);
    
    try {
      const ws = new WebSocket(wsUrl);
      
      // Add connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          addLog('WebSocket connection timeout - server not responding');
          setStatus('Error');
          // Close the connection attempt if it's still pending
          if (ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }
        }
      }, 5000);
      
      ws.onopen = function(event) {
        clearTimeout(connectionTimeout);
        addLog('WebSocket connection established');
        setConnected(true);
        setStatus('Connected');
        startFpsCounter();
        
        // Send a ping message to test the connection
        ws.send(JSON.stringify({
          type: "ping",
          timestamp: Date.now()
        }));
      };
      
      ws.onmessage = function(event) {
        try {
          const data = JSON.parse(event.data);
          console.log('Received WebSocket message:', data);
          
          if (data.type === 'video' && data.frame) {
            console.log('Rendering frame');
            renderAvatarFrame(data.frame);
            framesReceivedRef.current++;
          } else if (data.type === 'pong') {
            console.log('Ping-pong successful');
            addLog('Server connection confirmed');
          } else if (data.type === 'error') {
            console.error('Server error:', data.message);
            addLog(`Server error: ${data.message}`);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err, 'Raw message:', event.data);
          addLog('Error parsing server message');
        }
      };
      
      ws.onerror = function(error) {
        clearTimeout(connectionTimeout);
        console.error('WebSocket error:', error);
        addLog(`WebSocket error: ${error.message || 'Connection failed'}`);
        setStatus('Error');
      };
      
      ws.onclose = function(event) {
        clearTimeout(connectionTimeout);
        console.log('WebSocket closed:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean
        });
        addLog(`WebSocket closed: ${event.wasClean ? 'Clean disconnect' : 'Connection lost'} (Code: ${event.code})`);
        setConnected(false);
        setStatus('Disconnected');
        stopFpsCounter();
      };
      
      websocketRef.current = ws;
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      addLog(`Error connecting to WebSocket: ${error.message}`);
      setStatus('Error');
    }
  };
  
  // Disconnect WebSocket
  const disconnectWebSocket = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
      setConnected(false);
      stopSendingFrames();
      addLog('WebSocket disconnected');
    }
  };
  
  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 }
        } 
      });
      
      mediaStreamRef.current = stream;
      if (webcamRef.current) {
        webcamRef.current.srcObject = stream;
      }
      
      setCameraActive(true);
      startSendingFrames();
      addLog('Webcam started');
    } catch (error) {
      addLog(`Error accessing webcam: ${error.message}`);
    }
  };
  
  // Stop webcam
  const stopWebcam = () => {
    if (mediaStreamRef.current) {
      stopSendingFrames();
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      if (webcamRef.current) {
        webcamRef.current.srcObject = null;
      }
      mediaStreamRef.current = null;
      setCameraActive(false);
      addLog('Webcam stopped');
    }
  };
  
  // Start audio
  const startAudio = async () => {
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      audioSourceRef.current = audioContextRef.current.createMediaStreamSource(audioStream);
      
      const bufferSize = 4096;
      audioProcessorRef.current = audioContextRef.current.createScriptProcessor(bufferSize, 1, 1);
      
      audioSourceRef.current.connect(audioProcessorRef.current);
      audioProcessorRef.current.connect(audioContextRef.current.destination);
      
      audioProcessorRef.current.onaudioprocess = function(event) {
        if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
          const inputData = event.inputBuffer.getChannelData(0);
          const audio_data = btoa(String.fromCharCode.apply(null, new Uint8Array(inputData.buffer)));
          
          websocketRef.current.send(JSON.stringify({
            audio_data: audio_data,
            sample_rate: audioContextRef.current.sampleRate
          }));
        }
      };
      
      setAudioActive(true);
      addLog('Audio started');
    } catch (error) {
      addLog(`Error accessing microphone: ${error.message}`);
    }
  };
  
  // Stop audio
  const stopAudio = () => {
    if (audioContextRef.current) {
      if (audioProcessorRef.current) {
        audioProcessorRef.current.disconnect();
        audioProcessorRef.current = null;
      }
      
      if (audioSourceRef.current) {
        audioSourceRef.current.disconnect();
        audioSourceRef.current = null;
      }
      
      audioContextRef.current.close();
      audioContextRef.current = null;
      setAudioActive(false);
      addLog('Audio stopped');
    }
  };
  
  // Start sending frames
  const startSendingFrames = () => {
    if (videoIntervalRef.current) return;
    
    videoIntervalRef.current = setInterval(() => {
      if (websocketRef.current && 
          websocketRef.current.readyState === WebSocket.OPEN && 
          mediaStreamRef.current &&
          webcamRef.current) {
        
        const video = webcamRef.current;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        websocketRef.current.send(JSON.stringify({
          frame_data: imageData,
          timestamp: Date.now()
        }));
      }
    }, 33); // ~30 FPS
  };
  
  // Stop sending frames
  const stopSendingFrames = () => {
    if (videoIntervalRef.current) {
      clearInterval(videoIntervalRef.current);
      videoIntervalRef.current = null;
    }
  };
  
  // Render avatar frame
  const renderAvatarFrame = (frameData) => {
    if (!avatarCanvasRef.current) return;
    
    const canvas = avatarCanvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.onload = function() {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = frameData;  // Use the frame data directly since it already includes the prefix
  };
  
  // FPS counter
  const startFpsCounter = () => {
    if (fpsIntervalRef.current) return;
    
    fpsIntervalRef.current = setInterval(() => {
      setFps(framesReceivedRef.current);
      framesReceivedRef.current = 0;
    }, 1000);
  };
  
  // Stop FPS counter
  const stopFpsCounter = () => {
    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
      fpsIntervalRef.current = null;
      setFps(0);
    }
  };
  
    
  // Add calibration handlers
  const startCalibration = async () => {
    try {
      // First check if we're connected
      if (!connected || !websocketRef.current) {
        addLog('Please connect to WebSocket first');
        return;
      }

      // Check if camera is active
      if (!cameraActive) {
        addLog('Please start camera first');
        return;
      }

      setCalibrating(true);
      addLog('Starting avatar calibration...');
      setStatus('Calibrating');
      
    } catch (error) {
      console.error('Calibration error:', error);
      addLog(`Calibration error: ${error.message}`);
      setCalibrating(false);
    }
  };
    
  const handleCalibrationComplete = () => {
    setCalibrating(false);
    setCalibrated(true);
    addLog('Avatar calibration complete!');
  };
    
  const handleCalibrationError = (error) => {
    setCalibrating(false);
    addLog(`Calibration error: ${error.message}`);
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
      stopAudio();
      disconnectWebSocket();
      stopFpsCounter();
    };
  }, []);
  
  return (
    <div className="app">
      <header className="header">
        <h1>Real-Time Avatar System</h1>
      </header>
      
      <div className="container">
        <WebcamView webcamRef={webcamRef} />
        {calibrating ? (
          <CalibrationView 
            webcamRef={webcamRef}
            onCalibrationComplete={handleCalibrationComplete}
            onCalibrationError={handleCalibrationError}
          />
        ) : (
          <AvatarView canvasRef={avatarCanvasRef} />
        )}
      </div>
      
      <div className="container">
        <ControlPanel 
          connected={connected}
          cameraActive={cameraActive}
          audioActive={audioActive}
          latency={latency}
          fps={fps}
          status={status}
          logs={logs}
          onConnect={connectWebSocket}
          onDisconnect={disconnectWebSocket}
          onStartCamera={startWebcam}
          onStopCamera={stopWebcam}
          onStartAudio={startAudio}
          onStopAudio={stopAudio}
          calibrated={calibrated}
          calibrating={calibrating}
          onStartCalibration={startCalibration}
          disabled={!connected || calibrating}
        />
      </div>
      
      <footer className="footer">
        <p>Real-Time Avatar System for Live Interviews | Hackathon Project</p>
      </footer>
    </div>
  );
}

export default App; 