# neck_fpn.py
import torch
import torch.nn.functional as F
from torch import nn

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

class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) module that combines multi-level features from a backbone network.
    It reduces the channels of input feature maps, combines them through top-down and lateral connections,
    and concatenates them to produce a single output feature map.
    """
    def __init__(self, in_channels, inner_channels=256, **kwargs):
        """
        Initialize the FPN module.

        Args:
            in_channels (list or tuple): List of input channel sizes from the backbone network.
            inner_channels (int): Number of channels in the output feature map. Defaults to 256.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # Reduce layers to adjust channel dimensions
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers to refine feature maps
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        # Final convolution to process concatenated features
        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_channels = self.conv_out

    def forward(self, x):
        """
        Forward pass of the FPN module.

        Args:
            x (tuple): Tuple of four tensors (c2, c3, c4, c5) from the backbone network.

        Returns:
            Tensor: The output feature map after combining and processing the input feature maps.
        """
        c2, c3, c4, c5 = x
        # Top-down pathway to combine features
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)

        # Upsample and concatenate all levels
        x = self._upsample_cat(p2, p3, p4, p5)
        # Process concatenated features
        x = self.conv(x)
        return x

    def _upsample_add(self, x, y):
        """
        Upsample x to match y's spatial size and add them element-wise.

        Args:
            x (Tensor): Feature map to upsample.
            y (Tensor): Feature map to add to.

        Returns:
            Tensor: The sum of upsampled x and y.
        """
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        """
        Upsample p3, p4, p5 to p2's spatial size and concatenate all four feature maps.

        Args:
            p2 (Tensor): Feature map with the target spatial size.
            p3 (Tensor): Feature map to upsample.
            p4 (Tensor): Feature map to upsample.
            p5 (Tensor): Feature map to upsample.

        Returns:
            Tensor: The concatenated feature map.
        """
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)