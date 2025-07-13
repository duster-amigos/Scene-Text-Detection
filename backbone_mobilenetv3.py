# backbone_mobilenetv3_small.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from utils import logger

class MobileNetV3(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        logger.model_info(f"Initializing MobileNetV3 backbone")

        def set_module_inplace_false(module):
            for m in module.modules():
                if isinstance(m, (nn.ReLU, nn.ReLU6, nn.Hardswish)):
                    m.inplace = False

        try:
            if pretrained:
                m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                m = mobilenet_v3_small(weights=None)
            set_module_inplace_false(m)
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
            m = mobilenet_v3_small(weights=None)
            set_module_inplace_false(m)

        self.features = m.features
        self._idxs = [2, 4, 9, len(self.features) - 1]
        self.out_channels = [24, 40, 96, 576]

    def forward(self, x):
        try:
            logger.model_info(f"MobileNetV3 forward pass - input shape: {x.shape}")
            outs = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self._idxs:
                    outs.append(x)
            logger.model_info(f"MobileNetV3 forward pass completed - {len(outs)} feature maps extracted")
            return outs
        except Exception as e:
            logger.error(f"Error in MobileNetV3 forward pass: {e}")
            raise

# # backbone_mobilenetv3_large.py
# import torch.nn as nn
# from torchvision.models import mobilenet_v3_large

# class MobileNetV3(nn.Module):
#     """
#     Wraps torchvision’s MobileNetV3-Large so that it outputs exactly
#     four feature maps with channel sizes [40, 80, 160, 960].
#     """
#     def __init__(self, pretrained: bool = True, in_channels: int = 3):
#         super().__init__()
#         m = mobilenet_v3_large(pretrained=pretrained)
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