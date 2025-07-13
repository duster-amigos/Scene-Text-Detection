# neck_fpem_ffm.py
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

class FPEM_FFM(nn.Module):
    """
    Feature Pyramid Enhancement Module (FPEM) and Feature Fusion Module (FFM) for combining multi-level features.
    """
    def __init__(self, in_channels, inner_channels=128, fpem_repeat=2, **kwargs):
        """
        Initialize the FPEM_FFM module.

        Args:
            in_channels (list or tuple): List of input channel sizes from the backbone network.
            inner_channels (int, optional): Number of channels in the intermediate feature maps. Defaults to 128.
            fpem_repeat (int, optional): Number of times to repeat the FPEM module. Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.conv_out = inner_channels
        inplace = True
        # Reduce layers to adjust channel dimensions
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(self.conv_out))
        self.out_channels = self.conv_out * 4

    def forward(self, x):
        """
        Forward pass of the FPEM_FFM module.

        Args:
            x (tuple): Tuple of four tensors (c2, c3, c4, c5) from the backbone network.

        Returns:
            Tensor: The fused feature map.
        """
        c2, c3, c4, c5 = x
        # Reduce channel dimensions
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # Apply FPEM modules and accumulate features
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5

        # Feature Fusion: upsample and concatenate
        c5 = F.interpolate(c5_ffm, size=c2_ffm.size()[-2:], mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4_ffm, size=c2_ffm.size()[-2:], mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3_ffm, size=c2_ffm.size()[-2:], mode='bilinear', align_corners=False)
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        return Fy

class FPEM(nn.Module):
    """
    Feature Pyramid Enhancement Module (FPEM) for enhancing multi-level features.
    """
    def __init__(self, in_channels=128):
        """
        Initialize the FPEM module.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 128.
        """
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        """
        Forward pass of the FPEM module.

        Args:
            c2 (Tensor): Feature map from level 2.
            c3 (Tensor): Feature map from level 3.
            c4 (Tensor): Feature map from level 4.
            c5 (Tensor): Feature map from level 5.

        Returns:
            tuple: Enhanced feature maps (c2, c3, c4, c5).
        """
        # Up stage: top-down path
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))

        # Down stage: bottom-up path
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        """
        Upsample x to match y's spatial size and add them element-wise.

        Args:
            x (Tensor): Feature map to upsample.
            y (Tensor): Feature map to add to.

        Returns:
            Tensor: The sum of upsampled x and y.
        """
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

class SeparableConv2d(nn.Module):
    """
    Separable Convolution module consisting of depthwise and pointwise convolutions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the SeparableConv2d module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride of the depthwise convolution. Defaults to 1.
        """
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                                        stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the SeparableConv2d module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after separable convolution, batch normalization, and ReLU activation.
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x