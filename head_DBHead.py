# head_DBHead.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger, log_exception

class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) head for text detection.
    """
    def __init__(self, in_channels, k=50, adaptive=False, out_channels=2):
        """
        Initialize DB Head.
        
        Args:
            in_channels (int): Number of input channels
            k (int): Threshold for binarization
            adaptive (bool): Whether to use adaptive thresholding
            out_channels (int): Number of output channels (2 for inference, 3 for training)
        """
        super().__init__()
        logger.info(f"Initializing DBHead (in_channels={in_channels}, k={k}, adaptive={adaptive}, out_channels={out_channels})")
        
        try:
            self.k = k
            self.adaptive = adaptive
            
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
            
            logger.info("DBHead initialized successfully")
            
        except Exception as e:
            log_exception(e, "Failed to initialize DBHead")
            raise RuntimeError(f"DBHead initialization failed: {str(e)}")

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
            head = DBHead(in_channels=in_channels, adaptive=adaptive)
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