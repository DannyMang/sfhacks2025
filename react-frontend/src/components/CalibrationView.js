import React, { useState, useEffect, useRef } from 'react';

const CalibrationView = ({ 
  webcamRef, 
  onCalibrationComplete,
  onCalibrationError 
}) => {
  const [currentPose, setCurrentPose] = useState(null);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [status, setStatus] = useState('waiting');
  const [error, setError] = useState(null);
  const [connecting, setConnecting] = useState(false);
  
  // Use a ref to store the WebSocket to prevent re-renders from creating multiple connections
  const wsRef = useRef(null);
  const pingIntervalRef = useRef(null);
  const isConnectingRef = useRef(false);
  
  // Connect to the WebSocket when the component mounts
  useEffect(() => {
    console.log("Calibration WebSocket effect triggered");
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    
    // Connect to the calibration WebSocket with exponential backoff
    const connectWebSocket = () => {
      // Prevent multiple connection attempts
      if (isConnectingRef.current || 
          (wsRef.current && 
           (wsRef.current.readyState === WebSocket.CONNECTING || 
            wsRef.current.readyState === WebSocket.OPEN))) {
        console.log('WebSocket already connected or connecting');
        return;
      }
      
      if (reconnectAttempts >= maxReconnectAttempts) {
        console.error(`Failed to connect after ${maxReconnectAttempts} attempts`);
        setError(`Failed to connect to calibration server after ${maxReconnectAttempts} attempts`);
        if (onCalibrationError) {
          onCalibrationError(new Error('Maximum reconnection attempts reached'));
        }
        return;
      }
      
      setConnecting(true);
      setMessage(`Connecting to calibration server (attempt ${reconnectAttempts + 1})...`);
      isConnectingRef.current = true;
      console.log(`Attempting to connect to calibration WebSocket (attempt ${reconnectAttempts + 1})`);
      
      // Create a new WebSocket connection
      // Use wss:// for secure connections if your server supports it
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${protocol}//${window.location.hostname}:8000/ws/calibration`);
      wsRef.current = ws;
      
      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          console.log('Connection timeout, closing socket');
          ws.close();
        }
      }, 5000);
      
      ws.onopen = () => {
        console.log('Calibration WebSocket connected');
        clearTimeout(connectionTimeout);
        setConnecting(false);
        isConnectingRef.current = false;
        reconnectAttempts = 0; // Reset reconnect attempts on successful connection
        setStatus('connected');
        setMessage('Connected to calibration server. Waiting for instructions...');
        
        // Start sending ping messages to keep the connection alive
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 5000);
        
        // Request the first pose
        ws.send(JSON.stringify({ 
          type: 'start_calibration'
        }));
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Received calibration message:', data);
          
          if (data.type === 'test') {
            console.log('Test message received:', data.message);
          } else if (data.type === 'pong') {
            console.log('Pong received, connection confirmed');
          } else if (data.type === 'calibration_status') {
            // Handle calibration status updates
            if (data.status === 'calibrating') {
              // Update to next pose
              if (data.next_pose) {
                setCurrentPose(data.next_pose);
                setMessage(`Please ${data.next_pose.instruction}`);
              }
              setProgress(data.progress || 0);
            } else if (data.status === 'training') {
              setStatus('training');
              setMessage(data.message || 'Training avatar model...');
              setProgress(data.progress || 0);
            } else if (data.status === 'complete') {
              setStatus('complete');
              setMessage('Calibration complete!');
              setProgress(100);
              if (onCalibrationComplete) {
                onCalibrationComplete();
              }
            } else if (data.status === 'error') {
              setError(data.message || 'Unknown calibration error');
              if (onCalibrationError) {
                onCalibrationError(new Error(data.message));
              }
            }
          } else if (data.type === 'error') {
            setError(data.message || 'Unknown error');
            if (onCalibrationError) {
              onCalibrationError(new Error(data.message));
            }
          }
        } catch (error) {
          console.error('Error parsing calibration message:', error);
        }
      };
      
      ws.onclose = (event) => {
        console.log(`WebSocket closed with code: ${event.code}, reason: ${event.reason || 'No reason provided'}`);
        clearTimeout(connectionTimeout);
        isConnectingRef.current = false;
        
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        
        // Only attempt reconnection if not explicitly closed by the client
        if (event.code === 1006) {
          console.log('Abnormal closure, attempting to reconnect...');
          reconnectAttempts++;
          
          // Exponential backoff for reconnection
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
          console.log(`Scheduling reconnection in ${delay}ms`);
          
          setTimeout(() => {
            if (wsRef.current) {
              wsRef.current = null;
            }
            connectWebSocket();
          }, delay);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Don't set error state here, let onclose handle it
        // This prevents duplicate error messages
      };
    };
    
    // Connect to WebSocket
    connectWebSocket();
    
    // Cleanup function
    return () => {
      console.log('Cleaning up calibration WebSocket');
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
      
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []); // Empty dependency array means this effect runs once on mount
  
  // Function to capture a frame for the current pose
  const captureFrame = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('WebSocket connection not available');
      return;
    }
    
    if (!currentPose) {
      setError('No pose selected for calibration');
      return;
    }
    
    try {
      const video = webcamRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      
      ctx.drawImage(video, 0, 0);
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      console.log(`Sending frame for pose: ${currentPose.pose}`);
      wsRef.current.send(JSON.stringify({
        type: 'calibration_frame',
        frame_data: imageData,
        pose_type: currentPose.pose
      }));
    } catch (error) {
      console.error('Error capturing frame:', error);
      setError('Failed to capture frame');
    }
  };
  
  const retry = () => {
    setError(null);
    setStatus('waiting');
    
    // Reconnect the WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };
  
  return (
    <div className="calibration-view">
      {error ? (
        <div className="error-container">
          <h3>Calibration Error</h3>
          <p className="error-message">{error}</p>
          <button onClick={retry} className="retry-button">Retry</button>
        </div>
      ) : (
        <>
          <div className="calibration-status">
            <div className="progress-bar">
              <div 
                className="progress" 
                style={{width: `${progress}%`}}
              />
            </div>
            <p className="message">{message || 'Initializing...'}</p>
            {connecting && <p>Connecting to calibration server...</p>}
          </div>
          
          {currentPose && (
            <div className="pose-instructions">
              <h3>Current Pose: {currentPose.pose}</h3>
              <p>{currentPose.instruction}</p>
              <button 
                onClick={captureFrame}
                className="capture-button"
                disabled={!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN}
              >
                Capture Frame
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default CalibrationView;