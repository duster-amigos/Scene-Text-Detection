# head_DBHead.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger, log_exception

class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) head for text detection.
    """
    def __init__(self, in_channels, out_channels=2, k=50, adaptive=False):
        """
        Initialize DB Head.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels (2 for inference, 3 for training)
            k (int): Threshold for binarization
            adaptive (bool): Whether to use adaptive thresholding
        """
        super().__init__()
        logger.info(f"Initializing DBHead (in_channels={in_channels}, out_channels={out_channels}, k={k}, adaptive={adaptive})")
        
        try:
            self.k = k
            self.adaptive = adaptive
            self.out_channels = out_channels
            
            # Validate input parameters
            if in_channels <= 0:
                raise ValueError(f"in_channels must be positive, got {in_channels}")
            if out_channels not in [2, 3]:
                raise ValueError(f"out_channels must be 2 or 3, got {out_channels}")
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            
            # Build convolutional layers
            logger.debug("Building convolutional layers...")
            self.binarize = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
                nn.Sigmoid()
            )
            
            if self.adaptive:
                logger.debug("Building adaptive threshold branch...")
                self.thresh = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
                    nn.BatchNorm2d(in_channels // 4),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
                    nn.BatchNorm2d(in_channels // 4),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
                    nn.Sigmoid()
                )
            
            # Initialize weights
            self._initialize_weights()
            logger.info("DBHead initialized successfully")
            
        except Exception as e:
            log_exception(e, "Failed to initialize DBHead")
            raise RuntimeError(f"DBHead initialization failed: {str(e)}")

    def _initialize_weights(self):
        """Initialize weights for all layers."""
        try:
            logger.debug("Initializing DBHead weights...")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.debug("DBHead weights initialized successfully")
        except Exception as e:
            log_exception(e, "Failed to initialize DBHead weights")
            raise RuntimeError(f"Weight initialization failed: {str(e)}")

    def step_function(self, x, y):
        """
        Differentiable step function using piece-wise linear approximation.
        """
        try:
            logger.debug(f"Computing step function with k={self.k}")
            return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
        except Exception as e:
            log_exception(e, "Failed in step function computation")
            raise RuntimeError(f"Step function failed: {str(e)}")

    def forward(self, x):
        """
        Forward pass of DB Head.
        
        Args:
            x (Tensor): Input features
            
        Returns:
            Tensor: Probability map and threshold map
        """
        try:
            logger.debug(f"DBHead forward pass - Input shape: {x.shape}")
            
            # Validate input
            if x.dim() != 4:
                raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
            
            # Get probability map
            binary = self.binarize(x)
            logger.debug(f"Probability map shape: {binary.shape}")
            
            if self.training:
                if self.adaptive:
                    logger.debug("Computing adaptive threshold")
                    thresh = self.thresh(x)
                    logger.debug(f"Threshold map shape: {thresh.shape}")
                    thresh_binary = self.step_function(binary, thresh)
                    logger.debug(f"Thresholded binary map shape: {thresh_binary.shape}")
                    return torch.cat((binary, thresh, thresh_binary), dim=1)
                else:
                    logger.debug("Using fixed threshold")
                    return torch.cat((binary, torch.zeros_like(binary), binary), dim=1)
            else:
                logger.debug("Inference mode - returning probability and threshold maps")
                return torch.cat((binary, torch.zeros_like(binary)), dim=1)
                
        except Exception as e:
            log_exception(e, "Failed during DBHead forward pass")
            raise RuntimeError(f"DBHead forward pass failed: {str(e)}")

if __name__ == '__main__':
    # Test the DBHead
    try:
        logger.info("Testing DBHead...")
        
        # Test parameters
        batch_size = 2
        in_channels = 256
        height = 160
        width = 160
        
        # Create dummy input
        x = torch.randn(batch_size, in_channels, height, width)
        logger.debug(f"Created test input with shape: {x.shape}")
        
        # Test both adaptive and non-adaptive modes
        for adaptive in [True, False]:
            logger.info(f"\nTesting with adaptive={adaptive}")
            
            # Initialize head
            head = DBHead(in_channels=in_channels, out_channels=2, adaptive=adaptive)
            logger.info("DBHead created successfully")
            
            # Test training mode
            head.train()
            y_train = head(x)
            logger.info(f"Training output shape: {y_train.shape}")
            
            # Test inference mode
            head.eval()
            y_eval = head(x)
            logger.info(f"Inference output shape: {y_eval.shape}")
        
        logger.info("All DBHead tests passed successfully!")
        
    except Exception as e:
        log_exception(e, "DBHead test failed")
        logger.error("Test failed. See error details above.")