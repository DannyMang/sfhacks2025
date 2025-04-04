import React from 'react';

const WebcamView = ({ webcamRef }) => {
  return (
    <div className="video-container">
      <h2>Input Webcam</h2>
      <video 
        ref={webcamRef} 
        autoPlay 
        playsInline 
      />
    </div>
  );
};

export default WebcamView; 