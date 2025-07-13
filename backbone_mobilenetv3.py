# backbone_mobilenetv3_small.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetV3(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        super().__init__()

        # Initialize model with proper weights
        try:
            if pretrained:
                weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
                m = mobilenet_v3_small(weights=weights)
            else:
                m = mobilenet_v3_small(weights=None)

            # Handle different input channels if needed
            if in_channels != 3:
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
                    with torch.no_grad():
                        new_conv.weight[:, :3, :, :] = old_conv.weight
                        if in_channels > 3:
                            new_conv.weight[:, 3:, :, :] = 0
                m.features[0][0] = new_conv

            self.features = m.features
            self._idxs = [2, 4, 9, len(self.features) - 1]
            self.out_channels = [24, 40, 96, 576]

        except Exception as e:
            raise RuntimeError(f"Failed to initialize MobileNetV3: {str(e)}")

    def forward(self, x):
        """Forward pass of the backbone.
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
        Returns:
            list: List of feature maps at different scales
        """
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._idxs:
                outs.append(x)
        return outs

# # backbone_mobilenetv3_large.py
# import torch.nn as nn
# from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# class MobileNetV3(nn.Module):
#     """
#     Wraps torchvision's MobileNetV3-Large so that it outputs exactly
#     four feature maps with channel sizes [40, 80, 160, 960].
#     """
#     def __init__(self, pretrained: bool = True, in_channels: int = 3):
#         super().__init__()
#         if pretrained:
#             m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
#         else:
#             m = mobilenet_v3_large(weights=None)
#         self.features = m.features

#         # The layer indices in m.features where we tap out:
#         #   idx  4 → 40 ch
#         #   idx  7 → 80 ch
#         #   idx 13 →160 ch
#         #   idx 16 →960 ch
#         self._idxs = [4, 7, 13, 16]
#         self.out_channels = [40, 80, 160, 960]

#     def forward(self, x):
#         outs = []
#         for i, layer in enumerate(self.features):
#             x = layer(x)
#             if i in self._idxs:
#                 outs.append(x)
#         return outs