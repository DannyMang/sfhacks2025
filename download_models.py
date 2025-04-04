#!/usr/bin/env python3
"""
Script to download pre-trained models needed for the real-time avatar system.
"""

import os
import urllib.request
import zipfile
import tarfile
import hashlib
from tqdm import tqdm
import gdown

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join('app', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Define models to download
MODELS = [
    {
        'name': 'stylegan3_t.pt',
        'url': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
        'file_path': os.path.join(MODELS_DIR, 'stylegan3_t.pt'),
        'description': 'StyleGAN3-T trained on FFHQ dataset (1024x1024)',
    },
    {
        'name': 'wav2lip.pth',
        'url': 'https://github.com/Rudrabha/Wav2Lip/releases/download/weights/wav2lip.pth',
        'file_path': os.path.join(MODELS_DIR, 'wav2lip.pth'),
        'description': 'Wav2Lip model for lip sync',
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

def download_url(url, output_path, description):
    with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_gdrive(gdrive_id, output_path, description):
    print(f"Downloading {description}...")
    gdown.download(id=gdrive_id, output=output_path, quiet=False)

def main():
    print("Downloading pre-trained models...")

    for model in MODELS:
        if os.path.exists(model['file_path']):
            print(f"Model {model['name']} already exists, skipping download.")
            continue
        
        print(f"Downloading {model['name']}...")
        if 'url' in model:
            download_url(model['url'], model['file_path'], model['description'])
        elif 'gdrive_id' in model:
            download_gdrive(model['gdrive_id'], model['file_path'], model['description'])
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    main() 