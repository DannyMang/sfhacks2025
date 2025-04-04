import React, { useEffect } from 'react';

const AvatarView = ({ canvasRef }) => {
  useEffect(() => {
    if (canvasRef.current) {
      // Initialize canvas with black background
      const canvas = canvasRef.current;
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, [canvasRef]);
  
  return (
    <div className="video-container">
      <h2>Avatar Output</h2>
      <canvas ref={canvasRef} />
    </div>
  );
};

export default AvatarView; 