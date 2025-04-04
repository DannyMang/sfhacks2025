# React Frontend for Real-Time Avatar System

This is a React-based frontend for the Real-Time Avatar System. It provides a more modern and maintainable interface compared to the vanilla HTML/JS approach.

## Features

- Clean component-based architecture
- Real-time WebSocket communication with the backend
- Webcam and audio capture
- Canvas rendering for avatar output
- Responsive design

## Setup Instructions

1. Install dependencies:
```
cd react-frontend
npm install
```

2. Start the development server:
```
npm start
```

This will run the app in development mode. Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

## Production Build

To create an optimized production build:
```
npm run build
```

This will create a `build` folder with production-ready files.

## Integration with Backend

The React frontend is configured to connect to the backend API server running on port 8000. Make sure the backend is running before using the frontend.

To use a different backend URL, modify the following in `src/App.js`:

```javascript
const wsPort = 8000; // Change this to your backend port
```

## Project Structure

- `src/App.js` - Main application component with WebSocket logic
- `src/components/WebcamView.js` - Component for displaying webcam input
- `src/components/AvatarView.js` - Component for rendering avatar output
- `src/components/ControlPanel.js` - Component for UI controls and status display

## How to Use

1. Start the backend server first (see main project README)
2. Start the React frontend
3. Click "Connect WebSocket" to establish a connection to the backend
4. Click "Start Camera" to activate your webcam
5. Click "Start Microphone" to activate audio capture
6. The avatar will be generated based on your facial expressions and audio

## Notes for Development

- Modify `package.json` if you need to add additional dependencies
- The `proxy` field in `package.json` is set to redirect API requests to the backend server 