# backbone_mobilenetv3_small.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetV3(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        print(f"Initializing MobileNetV3 backbone with pretrained={pretrained}, in_channels={in_channels}")

        try:
            if pretrained:
                print("Loading pretrained MobileNetV3-Small weights from ImageNet")
                m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                print("Loading MobileNetV3-Small without pretrained weights")
                m = mobilenet_v3_small(weights=None)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Falling back to random initialization")
            m = mobilenet_v3_small(weights=None)

        self.features = m.features
        self._idxs = [2, 4, 9, len(self.features) - 1]
        self.out_channels = [24, 40, 96, 576]
        print(f"MobileNetV3 backbone initialized with output channels: {self.out_channels}")
        print(f"Feature extraction indices: {self._idxs}")

    def forward(self, x):
        try:
            print(f"MobileNetV3 forward pass - input shape: {x.shape}")
            outs = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self._idxs:
                    outs.append(x)
                    print(f"Extracted feature at layer {i}, shape: {x.shape}")
            print(f"MobileNetV3 forward pass completed - {len(outs)} feature maps extracted")
            return outs
        except Exception as e:
            print(f"Error in MobileNetV3 forward pass: {e}")
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