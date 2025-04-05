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

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join('app', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Define models to download (removed StyleGAN3 since you have it locally)
MODELS = [
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

def main():
    parser = argparse.ArgumentParser(description='Download pre-trained models for the avatar system')
    parser.add_argument('--dummy', action='store_true', help='Create dummy model files for development')
    parser.add_argument('--force', action='store_true', help='Force re-download of models')
    args = parser.parse_args()
    
    print("Downloading pre-trained models...")
    success_count = 0
    total_models = len(MODELS)

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
    
    if success_count == total_models:
        print("All models downloaded successfully!")
    else:
        print(f"Downloaded {success_count}/{total_models} models. Some features may be limited.")
        if not args.dummy:
            print("You can run with --dummy flag to create placeholder files for development.")

if __name__ == "__main__":
    main() 