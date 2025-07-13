# neck_fpem_ffm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger, log_exception

class ConvBnRelu(nn.Module):
    """
    A module that combines a 2D convolution, batch normalization, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inplace=True):
        """
        Initialize the ConvBnRelu module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Padding added to the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If True, adds a learnable bias to the output. Default: True
            padding_mode (str, optional): Padding mode. Default: 'zeros'
            inplace (bool, optional): If True, performs the ReLU operation in-place. Default: True
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        """
        Forward pass of the ConvBnRelu module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after convolution, batch normalization, and ReLU activation.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FPEM_FFM(nn.Module):
    """
    Feature Pyramid Enhancement Module (FPEM) with Feature Fusion Module (FFM)
    """
    def __init__(self, in_channels, inner_channels=128):
        """
        Initialize FPEM_FFM module.
        
        Args:
            in_channels (list): List of input channel numbers for each level
            inner_channels (int): Number of inner channels
        """
        super().__init__()
        logger.info(f"Initializing FPEM_FFM (in_channels={in_channels}, inner_channels={inner_channels})")
        
        try:
            self.in_channels = in_channels
            self.inner_channels = inner_channels
            
            # Reduce channels for each input level
            logger.debug("Building input reduction layers...")
            self.in_conv = nn.ModuleList()
            for c in in_channels:
                self.in_conv.append(
                    nn.Sequential(
                        nn.Conv2d(c, inner_channels, 1),
                        nn.BatchNorm2d(inner_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            
            # FPEM module
            logger.debug("Building FPEM modules...")
            self.fpem1 = FPEM(inner_channels)
            self.fpem2 = FPEM(inner_channels)
            
            # FFM module
            logger.debug("Building FFM module...")
            self.out_conv = nn.Sequential(
                nn.Conv2d(inner_channels * 4, inner_channels * 4, 3, padding=1),
                nn.BatchNorm2d(inner_channels * 4),
                nn.ReLU(inplace=True)
            )
            
            self.out_channels = inner_channels * 4
            logger.info("FPEM_FFM initialized successfully")
            
        except Exception as e:
            log_exception(e, "Failed to initialize FPEM_FFM")
            raise RuntimeError(f"FPEM_FFM initialization failed: {str(e)}")

    def forward(self, x):
        """
        Forward pass of FPEM_FFM.
        
        Args:
            x (list): List of input feature maps from different levels
            
        Returns:
            Tensor: Fused feature map
        """
        try:
            logger.debug(f"FPEM_FFM forward pass - Input shapes: {[f.shape for f in x]}")
            
            # Reduce input channels
            features = []
            for i, feature in enumerate(x):
                conv_out = self.in_conv[i](feature)
                features.append(conv_out)
                logger.debug(f"Reduced feature {i+1} shape: {conv_out.shape}")
            
            # Apply FPEM twice
            fpem1_out = self.fpem1(features)
            logger.debug(f"FPEM1 output shapes: {[f.shape for f in fpem1_out]}")
            
            fpem2_out = self.fpem2(fpem1_out)
            logger.debug(f"FPEM2 output shapes: {[f.shape for f in fpem2_out]}")
            
            # Resize and concatenate features
            out_size = fpem2_out[0].size()[2:]
            outputs = []
            for i, feature in enumerate(fpem2_out):
                resized = F.interpolate(feature, size=out_size, mode='bilinear', align_corners=True)
                outputs.append(resized)
                logger.debug(f"Resized feature {i+1} shape: {resized.shape}")
            
            # Concatenate and apply final convolution
            out = torch.cat(outputs, dim=1)
            logger.debug(f"Concatenated feature shape: {out.shape}")
            
            out = self.out_conv(out)
            logger.debug(f"Final output shape: {out.shape}")
            
            return out
            
        except Exception as e:
            log_exception(e, "Failed during FPEM_FFM forward pass")
            raise RuntimeError(f"FPEM_FFM forward pass failed: {str(e)}")

class FPEM(nn.Module):
    """
    Feature Pyramid Enhancement Module
    """
    def __init__(self, channels):
        """
        Initialize FPEM module.
        
        Args:
            channels (int): Number of channels
        """
        super().__init__()
        logger.debug(f"Initializing FPEM (channels={channels})")
        
        try:
            # Up convolution
            self.up_conv = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                ) for _ in range(3)
            ])
            
            # Down convolution
            self.down_conv = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                ) for _ in range(3)
            ])
            
            logger.debug("FPEM initialized successfully")
            
        except Exception as e:
            log_exception(e, "Failed to initialize FPEM")
            raise RuntimeError(f"FPEM initialization failed: {str(e)}")

    def forward(self, x):
        """
        Forward pass of FPEM.
        
        Args:
            x (list): List of input feature maps
            
        Returns:
            list: Enhanced feature maps
        """
        try:
            logger.debug(f"FPEM forward pass - Input shapes: {[f.shape for f in x]}")
            
            # Up path
            up_features = []
            up_x = x[3]  # Start from the deepest level
            up_features.append(up_x)
            
            for i in range(2, -1, -1):
                logger.debug(f"Up path - Processing level {i+1}")
                up_size = x[i].size()[2:]
                up_x = F.interpolate(up_x, size=up_size, mode='bilinear', align_corners=True)
                up_x = x[i] + up_x
                up_x = self.up_conv[i](up_x)
                up_features.insert(0, up_x)
                logger.debug(f"Up feature {i+1} shape: {up_x.shape}")
            
            # Down path
            down_features = []
            down_x = up_features[0]
            down_features.append(down_x)
            
            for i in range(3):
                logger.debug(f"Down path - Processing level {i+1}")
                down_x = F.adaptive_avg_pool2d(down_x, up_features[i+1].size()[2:])
                down_x = up_features[i+1] + down_x
                down_x = self.down_conv[i](down_x)
                down_features.append(down_x)
                logger.debug(f"Down feature {i+1} shape: {down_x.shape}")
            
            return down_features
            
        except Exception as e:
            log_exception(e, "Failed during FPEM forward pass")
            raise RuntimeError(f"FPEM forward pass failed: {str(e)}")

if __name__ == '__main__':
    # Test the FPEM_FFM module
    try:
        logger.info("Testing FPEM_FFM module...")
        
        # Test parameters
        batch_size = 2
        in_channels = [24, 40, 96, 576]  # MobileNetV3 output channels
        inner_channels = 256
        feature_sizes = [(80, 80), (40, 40), (20, 20), (20, 20)]
        
        # Create dummy input features
        features = []
        for i, (c, size) in enumerate(zip(in_channels, feature_sizes)):
            feature = torch.randn(batch_size, c, size[0], size[1])
            features.append(feature)
            logger.debug(f"Created feature {i+1} with shape: {feature.shape}")
        
        # Initialize and test FPEM_FFM
        fpem_ffm = FPEM_FFM(in_channels, inner_channels)
        logger.info("FPEM_FFM created successfully")
        
        # Test forward pass
        output = fpem_ffm(features)
        logger.info(f"Output shape: {output.shape}")
        logger.info("Forward pass completed successfully")
        
        # Verify output channels
        assert output.shape[1] == inner_channels * 4, "Output channels mismatch"
        logger.info("Output channel verification passed")
        
        logger.info("All FPEM_FFM tests passed successfully!")
        
    except Exception as e:
        log_exception(e, "FPEM_FFM test failed")
        logger.error("Test failed. See error details above.")