# Real-Time Avatar System for Live Interviews

A hackathon project that implements a low-latency avatar generation system for live video interviews, based on StyleGAN3 with optimizations for real-time performance.

## Features

- Real-time facial animation using pre-trained StyleGAN3 models
- Audio-to-lip sync with Wav2Lip integration
- Facial landmark detection via MediaPipe
- Frame interpolation for smooth video at reduced computational cost
- TensorRT optimization for inference speedup (when available)
- Cross-platform support (macOS, Linux, Windows)
- Graceful fallback to CPU mode when GPU is not available

## Setup Instructions

### Prerequisites

- NVIDIA GPU with at least 8GB VRAM (recommended 16GB+) or CPU for development
- CUDA 11.8+ and cuDNN installed (for GPU acceleration)
- Python 3.8+

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/realtime-avatar.git
cd realtime-avatar
```

2. Run the application with automatic setup:
```
python run.py
```

This will:
- Install the appropriate dependencies based on your OS
- Download the required pre-trained models
- Start both the backend and frontend servers
- Open a browser window with the application

### Advanced Usage

The `run.py` script provides several options for customization:

```
python run.py --help
```

Common options:
- `--cpu` - Force CPU mode even if CUDA is available
- `--debug` - Enable debug mode with more verbose output
- `--dummy-models` - Use placeholder model files for development
- `--api-port PORT` - Specify custom API server port (default: 8000)
- `--frontend-port PORT` - Specify custom frontend server port (default: 8080)

For macOS users:
```
python run.py --cpu --dummy-models
```

### Manual Setup

If you prefer manual setup:

1. Install dependencies:
```
# For macOS
pip install -r requirements-macos.txt

# For Linux/Windows with GPU
pip install -r requirements.txt
```

2. Download pre-trained models:
```
python download_models.py [--dummy]
```

3. Start the backend server:
```
python app/api/server.py
```

4. In a new terminal, start the frontend:
```
python app/frontend.py
```

5. Open your browser and navigate to http://localhost:8080

## Project Structure

- `app/api/` - Backend FastAPI server
- `app/components/` - Core avatar generation components
- `app/utils/` - Helper functions and utilities
- `app/models/` - Pre-trained models directory
- `requirements.txt` - Project dependencies for GPU systems
- `requirements-macos.txt` - Project dependencies for macOS development
- `run.py` - All-in-one launcher script
- `download_models.py` - Script to download pre-trained models
- `vast_training.py` - Utility for training on Vast.ai

## Architecture

The system uses a pipeline approach:
1. Capture video input from webcam
2. Extract facial landmarks with MediaPipe
3. Generate avatar frames with StyleGAN3
4. Synchronize audio with Wav2Lip
5. Apply frame interpolation for smoothness
6. Stream output to browser

## Optimization Techniques

- TensorRT model quantization (FP16/INT8) where available
- Early layer freezing for faster inference
- Frame interpolation to maintain perceived FPS
- Parallel processing pipeline
- Robust error handling for component failures

## Credits

This project builds upon several open-source tools and research:
- StyleGAN3: https://github.com/NVlabs/stylegan3
- Wav2Lip: https://github.com/Rudrabha/Wav2Lip
- MediaPipe: https://google.github.io/mediapipe/ 