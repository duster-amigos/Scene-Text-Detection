#!/usr/bin/env python3
"""
Minimal CUDA test to isolate device-side assert issues.
"""

import torch
import torch.nn as nn

def test_minimal_cuda():
    """Test minimal CUDA operations."""
    print("=" * 60)
    print("Minimal CUDA Test")
    print("=" * 60)
    
    try:
        # Check CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("CUDA not available")
            return False
            
        # Get device info
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test 1: Basic tensor creation and movement
        print("\nTest 1: Basic tensor operations")
        try:
            x = torch.zeros(2, 3, 64, 64, dtype=torch.float32)
            print(f"CPU tensor created: {x.shape}")
            
            x_cuda = x.to(device)
            print(f"Tensor moved to CUDA: {x_cuda.device}")
            print("✅ Basic tensor operations passed")
        except Exception as e:
            print(f"❌ Basic tensor operations failed: {e}")
            return False
        
        # Test 2: Simple model
        print("\nTest 2: Simple model")
        try:
            model = nn.Conv2d(3, 64, 3, padding=1)
            model = model.to(device)
            output = model(x_cuda)
            print(f"Simple model output: {output.shape}")
            print("✅ Simple model passed")
        except Exception as e:
            print(f"❌ Simple model failed: {e}")
            return False
        
        # Test 3: Random tensor
        print("\nTest 3: Random tensor")
        try:
            x_rand = torch.randn(2, 3, 64, 64, dtype=torch.float32)
            x_rand_cuda = x_rand.to(device)
            output_rand = model(x_rand_cuda)
            print(f"Random tensor output: {output_rand.shape}")
            print("✅ Random tensor passed")
        except Exception as e:
            print(f"❌ Random tensor failed: {e}")
            return False
        
        print("\n🎉 All minimal CUDA tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Minimal CUDA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_cuda_launch_blocking():
    """Test with CUDA_LAUNCH_BLOCKING=1 to get better error messages."""
    print("\n" + "=" * 60)
    print("Testing with CUDA_LAUNCH_BLOCKING=1")
    print("=" * 60)
    
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    try:
        # Re-import torch to pick up the environment variable
        import torch
        import torch.nn as nn
        
        device = torch.device('cuda:0')
        print(f"Testing with device: {device}")
        
        # Test basic operations
        x = torch.zeros(2, 3, 64, 64, dtype=torch.float32)
        x_cuda = x.to(device)
        print(f"Tensor moved to CUDA: {x_cuda.device}")
        
        model = nn.Conv2d(3, 64, 3, padding=1)
        model = model.to(device)
        output = model(x_cuda)
        print(f"Model output: {output.shape}")
        
        print("✅ CUDA_LAUNCH_BLOCKING test passed!")
        return True
        
    except Exception as e:
        print(f"❌ CUDA_LAUNCH_BLOCKING test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Minimal CUDA Tests")
    
    # Test 1: Normal CUDA operations
    result1 = test_minimal_cuda()
    
    # Test 2: With CUDA_LAUNCH_BLOCKING
    result2 = test_with_cuda_launch_blocking()
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Normal CUDA test: {'PASSED' if result1 else 'FAILED'}")
    print(f"CUDA_LAUNCH_BLOCKING test: {'PASSED' if result2 else 'FAILED'}")
    
    if result1 and result2:
        print("\n🎉 All tests passed! CUDA is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 