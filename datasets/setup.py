import os
import json
import subprocess
from pathlib import Path

def setup_kaggle():
    """Set up Kaggle API credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    with open('kaggle.json') as f:
        creds = json.load(f)
    
    (kaggle_dir / 'kaggle.json').write_text(json.dumps(creds))
    os.chmod(kaggle_dir / 'kaggle.json', 0o600)

def download_dataset():
    """Download and organize dataset"""
    raw_path = Path('datasets/raw')
    dataset = "princekhunt19/road-detection-imgs-and-labels"
    
    # Download dataset
    subprocess.run(f"kaggle datasets download -d {dataset} -p {raw_path}", shell=True, check=True)
    
    # Unzip and organize
    subprocess.run(f"unzip {raw_path/'road-detection-imgs-and-labels.zip'} -d {raw_path}", shell=True)
    subprocess.run(f"rm {raw_path/'road-detection-imgs-and-labels.zip'}", shell=True)
    
    # Move files to appropriate directories
    (raw_path / 'Annotations').mkdir(exist_ok=True)
    (raw_path / 'RGB').mkdir(exist_ok=True)
    (raw_path / 'Thermal').mkdir(exist_ok=True)
    
    # Organize files
    for f in raw_path.glob('*'):
        if f.is_file():
            if 'thermal' in f.name.lower():
                f.rename(raw_path / 'Thermal' / f.name)
            elif f.suffix == '.xml':  # Annotation files
                f.rename(raw_path / 'Annotations' / f.name)
            else:
                f.rename(raw_path / 'RGB' / f.name)

if __name__ == "__main__":
    setup_kaggle()
    download_dataset()
    print("✅ Dataset downloaded and organized successfully")