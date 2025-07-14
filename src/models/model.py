# model.py
from addict import Dict
from torch import nn
import torch.nn.functional as F

from src.models.build import build_backbone, build_neck, build_head

class Model(nn.Module):
    """
    A PyTorch model class that combines a backbone, neck, and head based on the provided configuration.
    """
    def __init__(self, model_config: dict):
        """
        Initialize the model with the given configuration.
        :param model_config: Dictionary containing the configuration for backbone, neck, and head.
        """
        super().__init__()
        try:
            print("Initializing DBNet model...")
            model_config = Dict(model_config)
            # Extract the type for backbone, neck, and head from the configuration
            backbone_type = model_config.backbone.pop('type')
            neck_type = model_config.neck.pop('type')
            head_type = model_config.head.pop('type')
            
            # Build the backbone, neck, and head using the specified types and configurations
            self.backbone = build_backbone(backbone_type, **model_config.backbone)
            self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
            self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
            
            # Set the model name based on the types of backbone, neck, and head
            self.name = f'{backbone_type}_{neck_type}_{head_type}'
            
            # Print model summary
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Model: {self.name} | Parameters: {total_params:,} | Trainable: {trainable_params:,}")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor.
        :return: Output tensor after passing through the model and upsampling.
        """
        try:
            _, _, H, W = x.size()
            
            # Pass the input through the backbone to extract features
            backbone_out = self.backbone(x)
            
            # Pass the backbone output through the neck to refine features
            neck_out = self.neck(backbone_out)
            
            # Pass the neck output through the head to produce the final output
            y = self.head(neck_out)
            
            # Upsample the output to match the input size using bilinear interpolation
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
            return y
            
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            raise

if __name__ == '__main__':
    import torch
    import time

    try:
        print("Testing DBNet model...")
        # Set the device to CPU for testing
        device = torch.device('cpu')
        print(f"Using device: {device}")
        
        # Create a dummy input tensor for testing
        x = torch.zeros(2, 3, 640, 640).to(device)
        print(f"Created test input with shape: {x.shape}")

        # Define a sample model configuration for testing
        model_config = {
            'backbone': {'type': 'MobileNetV3', 'pretrained': True, "in_channels": 3},
            'neck': {'type': 'FPEM_FFM', 'inner_channels': 256},
            'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
        }
        print("Model configuration:", model_config)
        
        # Initialize the model with the sample configuration
        model = Model(model_config=model_config).to(device)
        print(f"Model created: {model.name}")

        # Perform a forward pass with the dummy input and measure the time
        print("Running forward pass...")
        tic = time.time()
        y = model(x)
        inference_time = time.time() - tic
        
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Output shape: {y.shape}")
        print(f"Model name: {model.name}")
        print("Model test completed successfully!")
        
    except Exception as e:
        print(f"Error in model test: {e}")
        import traceback
        traceback.print_exc()