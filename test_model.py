#!/usr/bin/env python3
"""
Simple test script to verify the DBNet model can be built and run.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Model
from config import get_default_config

def test_model():
    """Test if the model can be built and run forward pass."""
    print("Testing DBNet model...")
    
    try:
        # Get configuration
        config = get_default_config()
        
        # Build model
        model = Model(config.model)
        print(f"Model built successfully: {model.name}")
        
        # Create dummy input
        batch_size = 2
        channels = 3
        height = 640
        width = 640
        
        dummy_input = torch.randn(batch_size, channels, height, width)
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!")
        
        # Test on GPU if available
        if torch.cuda.is_available():
            print("\nTesting on GPU...")
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"GPU output shape: {output.shape}")
            print("GPU forward pass successful!")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1) 