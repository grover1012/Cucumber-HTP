import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

class EdgeLoss(nn.Module):
    """
    EdgeLoss that penalizes differences in edge gradients between predicted and ground truth masks.
    Inspired by TomatoScanner's approach for better segmentation accuracy on edges.
    
    Args:
        lambda_edge (float): Weight for edge loss (default: 0.3)
        kernel_size (int): Size of Sobel kernels (default: 3)
        sigma (float): Gaussian blur sigma for edge smoothing (default: 1.0)
    """
    
    def __init__(self, lambda_edge: float = 0.3, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Pre-compute Sobel kernels
        self.register_buffer('sobel_x', self._create_sobel_x())
        self.register_buffer('sobel_y', self._create_sobel_y())
    
    def _create_sobel_x(self) -> torch.Tensor:
        """Create Sobel X kernel."""
        if self.kernel_size == 3:
            kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        elif self.kernel_size == 5:
            kernel = torch.tensor([
                [-1, -2, 0, 2, 1],
                [-2, -3, 0, 3, 2],
                [-3, -4, 0, 4, 3],
                [-2, -3, 0, 3, 2],
                [-1, -2, 0, 2, 1]
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported kernel size: {self.kernel_size}")
        
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)
    
    def _create_sobel_y(self) -> torch.Tensor:
        """Create Sobel Y kernel."""
        if self.kernel_size == 3:
            kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        elif self.kernel_size == 5:
            kernel = torch.tensor([
                [-1, -2, -3, -2, -1],
                [-2, -3, -4, -3, -2],
                [0, 0, 0, 0, 0],
                [2, 3, 4, 3, 2],
                [1, 2, 3, 2, 1]
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported kernel size: {self.kernel_size}")
        
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)
    
    def sobel_gradients(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Sobel gradients for edge detection.
        
        Args:
            mask (torch.Tensor): Input mask of shape (N, 1, H, W) with values in [0, 1]
        
        Returns:
            torch.Tensor: Gradient magnitude of shape (N, 1, H, W)
        """
        # Ensure mask is in the right format
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Apply Sobel filters
        gx = F.conv2d(mask, self.sobel_x, padding=self.kernel_size // 2)
        gy = F.conv2d(mask, self.sobel_y, padding=self.kernel_size // 2)
        
        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(gx * gx + gy * gy + 1e-8)
        
        return gradient_magnitude
    
    def forward(self, pred_mask_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute edge loss between predicted and ground truth masks.
        
        Args:
            pred_mask_logits (torch.Tensor): Raw logits from segmentation head (N, 1, H, W)
            gt_mask (torch.Tensor): Ground truth binary mask (N, 1, H, W)
        
        Returns:
            torch.Tensor: Edge loss value
        """
        # Convert logits to probabilities
        pred_mask = torch.sigmoid(pred_mask_logits)
        
        # Ensure masks are in [0, 1] range
        gt_mask = torch.clamp(gt_mask, 0, 1)
        
        # Compute gradients
        pred_gradients = self.sobel_gradients(pred_mask)
        gt_gradients = self.sobel_gradients(gt_mask)
        
        # Compute L1 loss between gradients
        edge_loss = F.l1_loss(pred_gradients, gt_gradients)
        
        return self.lambda_edge * edge_loss

class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss that includes standard segmentation loss plus edge loss.
    
    Args:
        lambda_edge (float): Weight for edge loss (default: 0.3)
        edge_loss_kwargs (dict): Additional arguments for EdgeLoss
    """
    
    def __init__(self, lambda_edge: float = 0.3, **edge_loss_kwargs):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.edge_loss = EdgeLoss(lambda_edge=1.0, **edge_loss_kwargs)
    
    def forward(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor, 
                base_loss: torch.Tensor) -> torch.Tensor:
        """
        Combine base segmentation loss with edge loss.
        
        Args:
            pred_logits (torch.Tensor): Raw logits from segmentation head
            gt_mask (torch.Tensor): Ground truth mask
            base_loss (torch.Tensor): Base segmentation loss (e.g., BCE, Dice)
        
        Returns:
            torch.Tensor: Combined loss value
        """
        edge_loss = self.edge_loss(pred_logits, gt_mask)
        combined_loss = base_loss + self.lambda_edge * edge_loss
        
        return combined_loss

def edge_loss_simple(pred_mask_logits: torch.Tensor, gt_mask: torch.Tensor, 
                    lambda_edge: float = 0.3) -> torch.Tensor:
    """
    Simple edge loss function for easy integration with existing pipelines.
    
    Args:
        pred_mask_logits (torch.Tensor): Raw logits from segmentation head (N, 1, H, W)
        gt_mask (torch.Tensor): Ground truth binary mask (N, 1, H, W)
        lambda_edge (float): Weight for edge loss
    
    Returns:
        torch.Tensor: Edge loss value
    """
    # Create temporary EdgeLoss instance
    edge_loss_fn = EdgeLoss(lambda_edge=lambda_edge)
    return edge_loss_fn(pred_mask_logits, gt_mask)

def compute_edge_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor, 
                        threshold: float = 0.5) -> dict:
    """
    Compute edge-related metrics for evaluation.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask probabilities (N, 1, H, W)
        gt_mask (torch.Tensor): Ground truth binary mask (N, 1, H, W)
        threshold (float): Threshold for binarizing predictions
    
    Returns:
        dict: Dictionary containing edge metrics
    """
    # Binarize predictions
    pred_binary = (pred_mask > threshold).float()
    
    # Create EdgeLoss instance for gradient computation
    edge_loss_fn = EdgeLoss(lambda_edge=1.0)
    
    # Compute gradients
    pred_gradients = edge_loss_fn.sobel_gradients(pred_binary)
    gt_gradients = edge_loss_fn.sobel_gradients(gt_mask)
    
    # Compute edge error (similar to TomatoScanner paper)
    edge_error = torch.mean(torch.abs(pred_gradients - gt_gradients))
    edge_error_percent = edge_error.item() * 100
    
    # Compute edge precision and recall
    pred_edges = (pred_gradients > 0.1).float()
    gt_edges = (gt_gradients > 0.1).float()
    
    intersection = torch.sum(pred_edges * gt_edges)
    union = torch.sum(pred_edges) + torch.sum(gt_edges) - intersection
    
    edge_iou = intersection / (union + 1e-8)
    
    return {
        'edge_error': edge_error.item(),
        'edge_error_percent': edge_error_percent,
        'edge_iou': edge_iou.item(),
        'pred_edge_pixels': torch.sum(pred_edges).item(),
        'gt_edge_pixels': torch.sum(gt_edges).item()
    }

if __name__ == "__main__":
    # Test EdgeLoss with dummy data
    batch_size, channels, height, width = 2, 1, 64, 64
    
    # Create dummy predictions and ground truth
    pred_logits = torch.randn(batch_size, channels, height, width)
    gt_mask = torch.randint(0, 2, (batch_size, channels, height, width), dtype=torch.float32)
    
    # Test EdgeLoss
    edge_loss_fn = EdgeLoss(lambda_edge=0.3)
    loss = edge_loss_fn(pred_logits, gt_mask)
    
    print(f"EdgeLoss test - Loss value: {loss.item():.6f}")
    
    # Test edge metrics
    pred_probs = torch.sigmoid(pred_logits)
    metrics = compute_edge_metrics(pred_probs, gt_mask)
    
    print("Edge metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("EdgeLoss test completed successfully!")
