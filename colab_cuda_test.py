#!/usr/bin/env python3
"""
Google Colab CUDA Test
======================

Minimal test script to isolate CUDA device-side assert errors.
Run this in Google Colab to diagnose the issue.

Usage:
    Copy and paste this code into a Google Colab cell and run it.
"""

import os
import gc
import warnings
warnings.filterwarnings("ignore")

def test_cuda_basic():
    """Test basic CUDA operations."""
    print("=" * 50)
    print("Basic CUDA Test")
    print("=" * 50)
    
    # Set environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        device = torch.device('cuda:0')
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
        # Test 1: Create tensor on CPU first
        print("\nTest 1: CPU tensor creation")
        x_cpu = torch.randn(2, 3, 64, 64)
        print(f"✅ CPU tensor: {x_cpu.shape}")
        
        # Test 2: Move to GPU
        print("\nTest 2: Move to GPU")
        x_gpu = x_cpu.to(device, non_blocking=False)
        print(f"✅ GPU tensor: {x_gpu.shape}")
        
        # Test 3: Basic operation
        print("\nTest 3: Basic operation")
        y = x_gpu + 1.0
        print(f"✅ Operation successful: {y.shape}")
        
        # Clean up
        del x_cpu, x_gpu, y
        torch.cuda.empty_cache()
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_components():
    """Test model components."""
    print("\n" + "=" * 50)
    print("Model Component Test")
    print("=" * 50)
    
    try:
        import torch
        
        # Test CPU first
        print("\nTesting on CPU...")
        device_cpu = torch.device('cpu')
        
        # Import and test backbone
        from backbone_mobilenetv3 import MobileNetV3
        backbone = MobileNetV3()
        backbone.eval()
        
        x = torch.randn(1, 3, 640, 640, device=device_cpu)
        with torch.no_grad():
            features = backbone(x)
        print(f"✅ Backbone CPU: {len(features)} features")
        
        # Test GPU if basic test passed
        if test_cuda_basic():
            print("\nTesting on GPU...")
            device_gpu = torch.device('cuda:0')
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Move to GPU
            backbone = backbone.to(device_gpu)
            x = x.to(device_gpu)
            
            with torch.no_grad():
                features = backbone(x)
            print(f"✅ Backbone GPU: {len(features)} features")
            
            # Clean up
            del backbone, x, features
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False

def main():
    """Main test function."""
    print("Google Colab CUDA Diagnostic Test")
    print("This will test basic CUDA operations and model components.")
    
    # Clear memory
    gc.collect()
    
    # Test basic CUDA
    basic_success = test_cuda_basic()
    
    # Test model components
    model_success = test_model_components()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Basic CUDA: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Model Components: {'✅ PASS' if model_success else '❌ FAIL'}")
    
    if basic_success and model_success:
        print("\n🎉 All tests passed! CUDA is working correctly.")
    elif basic_success and not model_success:
        print("\n⚠️  Basic CUDA works but model has issues.")
    else:
        print("\n❌ CUDA has fundamental issues. Consider using CPU fallback.")

if __name__ == "__main__":
    main() 