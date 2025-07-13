#!/usr/bin/env python3
"""
Demo script for DBNet text detection
This script demonstrates how to use the trained model for text detection
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Model
from inference import TextDetector
from utils import create_directories, print_device_info

def create_sample_image():
    """
    Create a sample image with text for demonstration
    """
    try:
        # Create a white background
        img = Image.new('RGB', (640, 480), color='white')
        
        # Add some text using PIL
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
            except:
                font = ImageFont.load_default()
        
        # Add text
        texts = [
            ("Hello World!", (50, 50)),
            ("DBNet Demo", (50, 150)),
            ("Text Detection", (50, 250)),
            ("Sample Image", (50, 350))
        ]
        
        for text, position in texts:
            draw.text(position, text, fill='black', font=font)
        
        # Save the image
        sample_path = "sample_image.jpg"
        img.save(sample_path)
        print(f"Created sample image: {sample_path}")
        
        return sample_path
        
    except Exception as e:
        print(f"Error creating sample image: {e}")
        return None

def demo_without_model():
    """
    Demo without trained model - just show the architecture
    """
    try:
        print("=== DBNet Demo (Without Trained Model) ===")
        
        # Load configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Print device info
        print_device_info()
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(config['model']).to(device)
        
        print(f"Model created: {model.name}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create sample input
        sample_input = torch.randn(1, 3, 640, 640).to(device)
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            output = model(sample_input)
        
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output channels: {output.size(1)}")
        
        # Create sample image
        sample_path = create_sample_image()
        if sample_path:
            print(f"\nSample image created: {sample_path}")
            print("You can use this image for testing once you have a trained model.")
        
        print("\nDemo completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your ICDAR 2015 dataset")
        print("2. Train the model: python train.py --config config.json")
        print("3. Run inference: python inference.py --model checkpoints/dbnet_checkpoint_best.pth --config config.json --image sample_image.jpg")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

def demo_with_model(model_path):
    """
    Demo with trained model
    """
    try:
        print("=== DBNet Demo (With Trained Model) ===")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Please train the model first or provide correct path.")
            return
        
        # Initialize detector
        detector = TextDetector(model_path, 'config.json')
        
        # Create sample image if it doesn't exist
        sample_path = "sample_image.jpg"
        if not os.path.exists(sample_path):
            sample_path = create_sample_image()
            if not sample_path:
                print("Could not create sample image")
                return
        
        # Run detection
        print(f"Running detection on: {sample_path}")
        boxes = detector.detect_text(sample_path)
        
        print(f"Found {len(boxes)} text regions:")
        for i, box in enumerate(boxes):
            print(f"  Box {i+1}: {box['bbox']}, Confidence: {box['confidence']:.3f}")
        
        # Visualize results
        output_path = "demo_result.jpg"
        detector.visualize_results(sample_path, boxes, output_path)
        
        print(f"\nResults saved to: {output_path}")
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Error in demo with model: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='DBNet Demo')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to trained model checkpoint (optional)')
    args = parser.parse_args()
    
    try:
        # Create necessary directories
        create_directories(['checkpoints', 'results', 'data'])
        
        if args.model:
            demo_with_model(args.model)
        else:
            demo_without_model()
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 