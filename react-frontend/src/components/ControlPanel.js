import React from 'react';

const ControlPanel = ({ 
  connected, 
  cameraActive, 
  audioActive, 
  latency, 
  fps, 
  status, 
  logs,
  onConnect,
  onDisconnect,
  onStartCamera,
  onStopCamera,
  onStartAudio,
  onStopAudio
}) => {
  return (
    <div className="controls">
      <div className="control-group">
        <h3>Connection</h3>
        <button onClick={onConnect} disabled={connected}>Connect WebSocket</button>
        <button onClick={onDisconnect} disabled={!connected}>Disconnect</button>
      </div>
      
      <div className="control-group">
        <h3>WebCam Controls</h3>
        <button onClick={onStartCamera} disabled={cameraActive}>Start Camera</button>
        <button onClick={onStopCamera} disabled={!cameraActive}>Stop Camera</button>
      </div>
      
      <div className="control-group">
        <h3>Audio Controls</h3>
        <button onClick={onStartAudio} disabled={audioActive}>Start Microphone</button>
        <button onClick={onStopAudio} disabled={!audioActive}>Stop Microphone</button>
      </div>
      
      <div className="status">
        {logs.map((log, index) => (
          <p key={index} className="log">{log}</p>
        ))}
      </div>
      
      <div className="metrics">
        <div className="metric">
          <div>Latency</div>
          <div className="value">{latency} ms</div>
        </div>
        <div className="metric">
          <div>FPS</div>
          <div className="value">{fps}</div>
        </div>
        <div className="metric">
          <div>Status</div>
          <div className="value">{status}</div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel; 