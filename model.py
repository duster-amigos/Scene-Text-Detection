# model.py
import torch
from addict import Dict
from torch import nn
import torch.nn.functional as F

from build import build_backbone, build_neck, build_head
from utils.logger import logger, log_exception

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
            logger.info("Initializing DBNet model...")
            model_config = Dict(model_config)
            
            # Extract the type for backbone, neck, and head from the configuration
            backbone_type = model_config.backbone.pop('type')
            neck_type = model_config.neck.pop('type')
            head_type = model_config.head.pop('type')
            
            logger.debug(f"Building backbone: {backbone_type}")
            self.backbone = build_backbone(backbone_type, **model_config.backbone)
            
            logger.debug(f"Building neck: {neck_type}")
            self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
            
            logger.debug(f"Building head: {head_type}")
            self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
            
            # Initialize weights
            logger.debug("Initializing model weights...")
            self._initialize_weights()
            
            # Set the model name based on the types of backbone, neck, and head
            self.name = f'{backbone_type}_{neck_type}_{head_type}'
            logger.info(f"Model initialized successfully: {self.name}")
            
            # Log model structure
            logger.debug(f"Model structure:\n{str(self)}")
            
        except Exception as e:
            log_exception(e, "Failed to initialize DBNet model")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _initialize_weights(self):
        """Initialize model weights."""
        try:
            logger.debug("Starting weight initialization...")
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.debug(f"Initializing Conv2d weights for {name}")
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    logger.debug(f"Initializing normalization weights for {name}")
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.debug("Weight initialization completed successfully")
        except Exception as e:
            log_exception(e, "Failed to initialize weights")
            raise RuntimeError(f"Weight initialization failed: {str(e)}")

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor.
        :return: Output tensor after passing through the model and upsampling.
        """
        try:
            logger.debug(f"Forward pass - Input shape: {x.shape}")
            
            _, _, H, W = x.size()
            # Pass the input through the backbone to extract features
            backbone_out = self.backbone(x)
            logger.debug(f"Backbone output shapes: {[f.shape for f in backbone_out]}")
            
            # Pass the backbone output through the neck to refine features
            neck_out = self.neck(backbone_out)
            logger.debug(f"Neck output shape: {neck_out.shape}")
            
            # Pass the neck output through the head to produce the final output
            y = self.head(neck_out)
            logger.debug(f"Head output shape: {y.shape}")
            
            # Upsample the output to match the input size using bilinear interpolation
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
            logger.debug(f"Final output shape after interpolation: {y.shape}")
            
            return y
            
        except Exception as e:
            log_exception(e, "Forward pass failed")
            raise RuntimeError(f"Forward pass failed: {str(e)}")

if __name__ == '__main__':
    import time

    try:
        logger.info("Running model test...")
        
        # Set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create a dummy input tensor for testing
        x = torch.zeros(2, 3, 640, 640)
        logger.debug(f"Created test input tensor with shape: {x.shape}")

        # Define a sample model configuration for testing
        model_config = {
            'backbone': {'type': 'MobileNetV3', 'pretrained': True, "in_channels": 3},
            'neck': {'type': 'FPEM_FFM', 'inner_channels': 256},
            'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
        }
        
        # Initialize the model
        logger.info("Creating model...")
        model = Model(model_config)
        logger.info(f"Model created successfully: {model.name}")
        
        # Move model to device after initialization
        model = model.to(device)
        x = x.to(device)
        logger.debug("Model and input moved to device")

        # Perform a forward pass with the dummy input and measure the time
        logger.info("Running forward pass...")
        tic = time.time()
        y = model(x)
        inference_time = time.time() - tic
        
        logger.info(f"Forward pass completed in {inference_time:.4f} seconds")
        logger.info(f"Output shape: {y.shape}")
        logger.info(f"Test completed successfully")
        
    except Exception as e:
        log_exception(e, "Model test failed")
        logger.error("Test failed. See error details above.")