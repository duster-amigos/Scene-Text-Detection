# head_DBHead.py
import torch
from torch import nn
from utils import logger

class DBHead(nn.Module):
    """DBHead module for differentiable binarization in text detection tasks.
    Generates shrink maps and threshold maps; includes binary maps during training.
    """
    def __init__(self, in_channels, out_channels, k=50):
        """Initialize the DBHead module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (unused in this implementation).
            k (int): Steepness parameter for the step function.
        """
        super().__init__()
        try:
            logger.model_info(f"Initializing DBHead")
            self.k = k
            # Binarization module: produces shrink maps
            inc4 = in_channels // 4
            self.binarize = nn.Sequential(
                nn.Conv2d(in_channels, inc4, 3, padding=1),
                nn.BatchNorm2d(inc4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(inc4, inc4, 2, 2),
                nn.BatchNorm2d(inc4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(inc4, 1, 2, 2),
                nn.Sigmoid())
            self.binarize.apply(self.weights_init)

            # Threshold module: produces threshold maps
            self.thresh = self._init_thresh(in_channels)
            self.thresh.apply(self.weights_init)
        except Exception as e:
            logger.error(f"Error initializing DBHead: {e}")
            raise

    def forward(self, x):
        """Forward pass of the DBHead module.

        Args:
            x (Tensor): Input feature maps.

        Returns:
            Tensor: Concatenated maps (shrink and threshold maps; binary maps added during training).
        """
        try:
            logger.model_info(f"DBHead forward pass - input shape: {x.shape}")
            shrink_maps = self.binarize(x)
            threshold_maps = self.thresh(x)
            
            if self.training:
                binary_maps = self.step_function(shrink_maps, threshold_maps)
                y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
            else:
                y = torch.cat((shrink_maps, threshold_maps), dim=1)
            return y
        except Exception as e:
            logger.error(f"Error in DBHead forward pass: {e}")
            raise

    def weights_init(self, m):
        """Custom weight initialization for convolutional and batch norm layers.

        Args:
            m (nn.Module): Module to initialize.
        """
        try:
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight.data)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
        except Exception as e:
            logger.error(f"Error in weight initialization: {e}")

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        """Initialize the threshold module.

        Args:
            inner_channels (int): Number of inner channels.
            serial (bool): If True, adds an extra input channel.
            smooth (bool): If True, uses smooth upsampling.
            bias (bool): If True, adds bias to convolutional layers.

        Returns:
            nn.Sequential: The threshold module.
        """
        try:
            in_channels = inner_channels
            if serial:
                in_channels += 1

            ic4 = inner_channels // 4

            return nn.Sequential(
                nn.Conv2d(in_channels, ic4, 3, padding=1, bias=bias),
                nn.BatchNorm2d(ic4),
                nn.ReLU(inplace=True),
                self._init_upsample(ic4, ic4, smooth=smooth, bias=bias),
                nn.BatchNorm2d(ic4),
                nn.ReLU(inplace=True),
                self._init_upsample(ic4, 1, smooth=smooth, bias=bias),
                nn.Sigmoid())
        except Exception as e:
            logger.error(f"Error initializing threshold module: {e}")
            raise

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        """Initialize upsampling layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            smooth (bool): If True, uses upsampling followed by convolution.
            bias (bool): If True, adds bias to convolutional layers.

        Returns:
            nn.Module: The upsampling module.
        """
        try:
            if smooth:
                inter_out_channels = out_channels
                if out_channels == 1:
                    inter_out_channels = in_channels
                module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
                if out_channels == 1:
                    module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
                return nn.Sequential(module_list)
            else:
                return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        except Exception as e:
            logger.error(f"Error initializing upsampling: {e}")
            raise

    def step_function(self, x, y):
        """Differentiable approximation of the step function.

        Args:
            x (Tensor): Input tensor.
            y (Tensor): Threshold tensor.

        Returns:
            Tensor: Element-wise step function approximation.
        """
        try:
            # return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
            result = torch.sigmoid(self.k * (x - y))
            return result
        except Exception as e:
            logger.error(f"Error in step function: {e}")
            raise