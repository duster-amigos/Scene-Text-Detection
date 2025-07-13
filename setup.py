#!/usr/bin/env python3
"""
Setup script for DBNet project
"""

import os
import sys
import subprocess
import json

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def install_requirements():
    """Install required packages"""
    try:
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    dirs = [
        "checkpoints",
        "results", 
        "data/icdar2015/train/images",
        "data/icdar2015/train/labels",
        "data/icdar2015/val/images",
        "data/icdar2015/val/labels",
        "data/icdar2015/test/images",
        "data/icdar2015/test/labels"
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("CUDA is not available. Training will use CPU.")
    except ImportError:
        print("PyTorch not installed yet. Run install_requirements() first.")

def test_imports():
    """Test if all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import torch
        import torchvision
        import numpy as np
        import cv2
        import PIL
        
        # Test project imports
        from model import Model
        from dataset import ICDAR2015Dataset
        from losses import DBLoss
        
        print("All imports successful!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def create_sample_config():
    """Create a sample configuration if it doesn't exist"""
    if not os.path.exists("config.json"):
        config = {
            "model": {
                "backbone": {
                    "type": "MobileNetV3",
                    "pretrained": True,
                    "in_channels": 3
                },
                "neck": {
                    "type": "FPEM_FFM",
                    "inner_channels": 128,
                    "fpem_repeat": 2
                },
                "head": {
                    "type": "DBHead",
                    "out_channels": 2,
                    "k": 50
                }
            },
            "loss": {
                "alpha": 1.0,
                "beta": 10.0,
                "ohem_ratio": 3.0,
                "reduction": "mean",
                "eps": 1e-6
            },
            "training": {
                "epochs": 100,
                "batch_size": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "lr_step_size": 30,
                "lr_gamma": 0.1,
                "image_size": 640,
                "num_workers": 4,
                "resume": False,
                "checkpoint_path": "checkpoints/dbnet_checkpoint.pth"
            },
            "data": {
                "train_images": "data/icdar2015/train/images",
                "train_labels": "data/icdar2015/train/labels",
                "val_images": "data/icdar2015/val/images",
                "val_labels": "data/icdar2015/val/labels",
                "test_images": "data/icdar2015/test/images",
                "test_labels": "data/icdar2015/test/labels"
            },
            "inference": {
                "min_area": 100,
                "thresh": 0.3,
                "box_thresh": 0.5,
                "max_candidates": 1000
            },
            "evaluation": {
                "iou_threshold": 0.5,
                "save_predictions": True,
                "output_dir": "results"
            }
        }
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Created sample config.json")

def main():
    """Main setup function"""
    print("=== DBNet Setup ===")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Check CUDA
    check_cuda()
    
    # Test imports
    if test_imports():
        print("Setup completed successfully!")
        
        # Create sample config
        create_sample_config()
        
        print("\nNext steps:")
        print("1. Prepare your ICDAR 2015 dataset in the data/ directory")
        print("2. Run demo: python demo.py")
        print("3. Start training: python train.py --config config.json")
        
    else:
        print("Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 