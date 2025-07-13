# backbone_mobilenetv3_small.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from utils.logger import logger, log_exception

class MobileNetV3(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        logger.info(f"Initializing MobileNetV3 backbone (pretrained={pretrained}, in_channels={in_channels})")

        try:
            # Initialize model with proper weights
            if pretrained:
                logger.debug("Loading pretrained weights from ImageNet")
                weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
                m = mobilenet_v3_small(weights=weights)
            else:
                logger.debug("Initializing model without pretrained weights")
                m = mobilenet_v3_small(weights=None)

            # Handle different input channels if needed
            if in_channels != 3:
                logger.debug(f"Modifying input layer for {in_channels} channels")
                old_conv = m.features[0][0]
                new_conv = nn.Conv2d(
                    in_channels, 
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                if pretrained:
                    # Initialize new conv weights using existing weights
                    logger.debug("Transferring pretrained weights to new input layer")
                    with torch.no_grad():
                        new_conv.weight[:, :3, :, :] = old_conv.weight
                        if in_channels > 3:
                            new_conv.weight[:, 3:, :, :] = 0
                m.features[0][0] = new_conv

            self.features = m.features
            self._idxs = [2, 4, 9, len(self.features) - 1]
            self.out_channels = [24, 40, 96, 576]
            
            logger.debug(f"Feature extraction indices: {self._idxs}")
            logger.debug(f"Output channels: {self.out_channels}")
            logger.info("MobileNetV3 backbone initialized successfully")

        except Exception as e:
            log_exception(e, "Failed to initialize MobileNetV3 backbone")
            raise RuntimeError(f"Failed to initialize MobileNetV3: {str(e)}")

    def forward(self, x):
        """Forward pass of the backbone.
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
        Returns:
            list: List of feature maps at different scales
        """
        try:
            logger.debug(f"Backbone input shape: {x.shape}")
            outs = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self._idxs:
                    outs.append(x)
                    logger.debug(f"Feature map {len(outs)}: shape={x.shape}")
            return outs
            
        except Exception as e:
            log_exception(e, "Failed during backbone forward pass")
            raise RuntimeError(f"Backbone forward pass failed: {str(e)}")

if __name__ == '__main__':
    # Test the backbone
    try:
        logger.info("Testing MobileNetV3 backbone...")
        
        # Test parameters
        batch_size = 2
        in_channels = 3
        height = 640
        width = 640
        
        # Create dummy input
        x = torch.randn(batch_size, in_channels, height, width)
        logger.debug(f"Created test input with shape: {x.shape}")
        
        # Test both pretrained and non-pretrained
        for pretrained in [True, False]:
            logger.info(f"\nTesting with pretrained={pretrained}")
            
            # Initialize backbone
            backbone = MobileNetV3(pretrained=pretrained, in_channels=in_channels)
            logger.info("Backbone created successfully")
            
            # Test forward pass
            features = backbone(x)
            logger.info("Forward pass completed successfully")
            
            # Print feature shapes
            for i, feat in enumerate(features):
                logger.info(f"Feature {i+1} shape: {feat.shape}")
            
            # Test with different input channels
            if pretrained:
                logger.info("\nTesting with different input channels")
                backbone_4ch = MobileNetV3(pretrained=True, in_channels=4)
                x_4ch = torch.randn(batch_size, 4, height, width)
                features_4ch = backbone_4ch(x_4ch)
                logger.info("Different input channels test passed")
        
        logger.info("All backbone tests passed successfully!")
        
    except Exception as e:
        log_exception(e, "Backbone test failed")
        logger.error("Test failed. See error details above.")