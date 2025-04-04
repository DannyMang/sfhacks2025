# Real-Time Avatar System for Live Interviews

A hackathon project that implements a low-latency avatar generation system for live video interviews, based on StyleGAN3 with optimizations for real-time performance.

## Features

- Real-time facial animation using pre-trained StyleGAN3 models
- Audio-to-lip sync with Wav2Lip integration
- Facial landmark detection via MediaPipe
- Frame interpolation for smooth video at reduced computational cost
- TensorRT optimization for inference speedup

## Setup Instructions

### Prerequisites

- NVIDIA GPU with at least 8GB VRAM (recommended 16GB+)
- CUDA 11.8+ and cuDNN installed
- Python 3.8+

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/realtime-avatar.git
cd realtime-avatar
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download pre-trained models:
```
python download_models.py
```

### Running the Application

1. Start the backend server:
```
python app/api/server.py
```

2. In a new terminal, start the frontend:
```
python app/frontend.py
```

3. Open your browser and navigate to http://localhost:8000

## Project Structure

- `app/api/` - Backend FastAPI server
- `app/components/` - Core avatar generation components
- `app/utils/` - Helper functions and utilities
- `app/models/` - Pre-trained model definitions

## Architecture

The system uses a pipeline approach:
1. Capture video input from webcam
2. Extract facial landmarks with MediaPipe
3. Generate avatar frames with StyleGAN3
4. Synchronize audio with Wav2Lip
5. Apply frame interpolation for smoothness
6. Stream output to browser

## Optimization Techniques

- TensorRT model quantization (FP16/INT8)
- Early layer freezing for faster inference
- Frame interpolation to maintain perceived FPS
- Parallel processing pipeline

## Credits

This project builds upon several open-source tools and research:
- StyleGAN3: https://github.com/NVlabs/stylegan3
- Wav2Lip: https://github.com/Rudrabha/Wav2Lip
- MediaPipe: https://google.github.io/mediapipe/ 