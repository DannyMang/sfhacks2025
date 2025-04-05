#!/usr/bin/env python3
"""
Script to download pre-trained models needed for the real-time avatar system.
"""

import os
import sys
import urllib.request
import urllib.error
import time
from tqdm import tqdm
import argparse
import platform
import subprocess

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join('app', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Define models to download
MODELS = [
    {
        'name': 'StyleGAN3_FFHQ_1024x1024.pkl',
        'url': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
        'file_path': os.path.join(MODELS_DIR, 'StyleGAN3_FFHQ_1024x1024.pkl'),
        'description': 'StyleGAN3 model for face generation',
    },
    {
        'name': 'wav2lip.pth',
        'url': 'https://github.com/Rudrabha/Wav2Lip/releases/download/weights/wav2lip.pth',
        'file_path': os.path.join(MODELS_DIR, 'wav2lip.pth'),
        'description': 'Wav2Lip model for lip sync',
        'fallback_gdrive_id': '1Pz0_MJi_j-oHG3O5TSLRJSIsAXC0V3lT'
    },
    {
        'name': 'first_order_model.pth',
        'gdrive_id': '1PyQJmkdCsAkOYwUyaj_l-l0fr8vEdRmi',
        'file_path': os.path.join(MODELS_DIR, 'first_order_model.pth'),
        'description': 'First Order Motion Model for face animation',
    }
]

# TensorRT models (only for CUDA systems)
TENSORRT_MODELS = [
    {
        'name': 'stylegan3_tensorrt.engine',
        'description': 'TensorRT optimized StyleGAN3 model',
        'file_path': os.path.join(MODELS_DIR, 'stylegan3_tensorrt.engine'),
        'requires_conversion': True,
        'source_model': os.path.join(MODELS_DIR, 'StyleGAN3_FFHQ_1024x1024.pkl')
    },
    {
        'name': 'wav2lip_tensorrt.engine',
        'description': 'TensorRT optimized Wav2Lip model',
        'file_path': os.path.join(MODELS_DIR, 'wav2lip_tensorrt.engine'),
        'requires_conversion': True,
        'source_model': os.path.join(MODELS_DIR, 'wav2lip.pth')
    }
]

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path, description, timeout=30, max_retries=3):
    """Download a file from a URL with retry logic and timeout."""
    for attempt in range(max_retries):
        try:
            with DownloadProgressBar(unit='B', unit_scale=True,
                                    miniters=1, desc=description) as t:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, filename=output_path, 
                                        reporthook=t.update_to)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Download attempt {attempt+1} failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download after {max_retries} attempts: {e}")
                return False
    return False

def create_dummy_model(file_path, size_kb=10):
    """Create a dummy model file for development when download fails."""
    print(f"Creating dummy model file at {file_path} for development purposes.")
    with open(file_path, 'wb') as f:
        f.write(b'\0' * size_kb * 1024)
    return True

def check_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def convert_to_tensorrt(model_info):
    """Convert PyTorch model to TensorRT format."""
    try:
        print(f"Converting {model_info['source_model']} to TensorRT format...")
        
        # Import necessary modules for conversion
        import torch
        from torch2trt import torch2trt
        
        # For StyleGAN3 model
        if 'stylegan3' in model_info['name']:
            from app.components.avatar_generator import AvatarGenerator
            
            # Initialize the model
            model = AvatarGenerator(model_info['source_model'], device='cuda')
            
            # Create dummy input
            dummy_input = torch.randn(1, model.model.z_dim).cuda()
            dummy_class = torch.zeros([1, model.model.c_dim], device='cuda')
            
            # Convert to TensorRT
            model_trt = torch2trt(
                model.model, 
                [dummy_input, dummy_class],
                fp16_mode=True,
                max_workspace_size=1 << 30,
                max_batch_size=1
            )
            
            # Save the TensorRT model
            torch.save(model_trt.state_dict(), model_info['file_path'])
            print(f"Successfully converted {model_info['name']} to TensorRT format")
            return True
            
        # For Wav2Lip model
        elif 'wav2lip' in model_info['name']:
            from app.components.voice2face import Voice2Face
            
            # Initialize the model
            model = Voice2Face(model_info['source_model'], device='cuda')
            
            # Create dummy input (mel spectrogram)
            dummy_input = torch.randn(1, 80, 16).cuda()  # Batch, mel channels, time frames
            
            # Convert to TensorRT
            model_trt = torch2trt(
                model.model, 
                [dummy_input],
                fp16_mode=True,
                max_workspace_size=1 << 30,
                max_batch_size=1
            )
            
            # Save the TensorRT model
            torch.save(model_trt.state_dict(), model_info['file_path'])
            print(f"Successfully converted {model_info['name']} to TensorRT format")
            return True
            
    except Exception as e:
        print(f"Failed to convert model to TensorRT: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download pre-trained models for the avatar system')
    parser.add_argument('--dummy', action='store_true', help='Create dummy model files for development')
    parser.add_argument('--force', action='store_true', help='Force re-download of models')
    parser.add_argument('--skip-tensorrt', action='store_true', help='Skip TensorRT model conversion')
    args = parser.parse_args()
    
    print("Downloading pre-trained models...")
    success_count = 0
    total_models = len(MODELS)

    # Download base models
    for model in MODELS:
        if os.path.exists(model['file_path']) and not args.force:
            print(f"Model {model['name']} already exists, skipping download.")
            success_count += 1
            continue
        
        print(f"Downloading {model['name']}...")
        download_success = False
        
        # Try URL if available
        if 'url' in model:
            download_success = download_url(model['url'], model['file_path'], model['description'])
        
        # Create dummy if download failed and --dummy flag is set
        if not download_success and args.dummy:
            download_success = create_dummy_model(model['file_path'])
        
        if download_success:
            success_count += 1
            print(f"Successfully downloaded {model['name']}.")
        else:
            print(f"Failed to download {model['name']}. The avatar system may not work properly.")
    
    # Check if CUDA is available for TensorRT conversion
    cuda_available = check_cuda_available()
    
    # Process TensorRT models if CUDA is available
    if cuda_available and not args.skip_tensorrt:
        print("CUDA is available. Processing TensorRT models...")
        
        # Check if torch2trt is installed
        try:
            import torch2trt
            tensorrt_available = True
        except ImportError:
            print("torch2trt not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch2trt"])
                tensorrt_available = True
            except Exception as e:
                print(f"Failed to install torch2trt: {e}")
                tensorrt_available = False
        
        if tensorrt_available:
            for model in TENSORRT_MODELS:
                if os.path.exists(model['file_path']) and not args.force:
                    print(f"TensorRT model {model['name']} already exists, skipping conversion.")
                    continue
                
                if model['requires_conversion']:
                    # Check if source model exists
                    if not os.path.exists(model['source_model']):
                        print(f"Source model {model['source_model']} not found. Cannot convert to TensorRT.")
                        continue
                    
                    # Convert to TensorRT
                    convert_success = convert_to_tensorrt(model)
                    
                    if convert_success:
                        print(f"Successfully created TensorRT model {model['name']}.")
                    else:
                        print(f"Failed to create TensorRT model {model['name']}.")
        else:
            print("TensorRT conversion skipped due to missing dependencies.")
    elif not cuda_available:
        print("CUDA is not available. Skipping TensorRT model conversion.")
    else:
        print("TensorRT conversion skipped as requested.")
    
    if success_count == total_models:
        print("All base models downloaded successfully!")
    else:
        print(f"Downloaded {success_count}/{total_models} base models. Some features may be limited.")
        if not args.dummy:
            print("You can run with --dummy flag to create placeholder files for development.")

if __name__ == "__main__":
    main() 