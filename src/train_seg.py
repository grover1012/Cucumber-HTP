import os
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.data.augment import Compose
import cv2
import numpy as np
from typing import Dict, Any, Optional

# Import our custom modules
from edge_aug import EdgeBoost, edgeboost_transform, create_edgeboost_pipeline
from edge_losses import EdgeLoss, edge_loss_simple, compute_edge_metrics

class EdgeBoostCompose(Compose):
    """
    Custom augmentation pipeline that includes EdgeBoost.
    This extends Ultralytics' Compose class to add edge enhancement.
    """
    
    def __init__(self, transforms, edgeboost_prob: float = 0.5, 
                 lambda_c: float = 1.5, lambda_a: float = 1.6):
        super().__init__(transforms)
        self.edgeboost_prob = edgeboost_prob
        self.lambda_c = lambda_c
        self.lambda_a = lambda_a
    
    def __call__(self, im, labels, *args, **kwargs):
        # Apply EdgeBoost with probability
        if np.random.random() < self.edgeboost_prob:
            im = edgeboost_transform(im, labels)[0]
        
        # Apply standard augmentations
        return super().__call__(im, labels, *args, **kwargs)

# Note: EdgeAwareTrainer removed for compatibility with current Ultralytics version
# Edge loss is now integrated via callbacks in the train function

def train(data_yaml: str = "data/data.yaml", 
          model: str = "yolov8m-seg.pt", 
          epochs: int = 150, 
          imgsz: int = 1024, 
          batch: int = 8, 
          edge_lambda: float = 0.3,
          edgeboost_prob: float = 0.5,
          lambda_c: float = 1.5,
          lambda_a: float = 1.6,
          device: str = "auto",
          project: str = "outputs/models",
          name: str = "yolov8m-seg-edge",
          **kwargs) -> Dict[str, Any]:
    """
    Train YOLO segmentation model with EdgeBoost and EdgeLoss.
    
    Args:
        data_yaml (str): Path to data.yaml file
        model (str): Model to train (yolov8n-seg.pt, yolov8s-seg.pt, etc.)
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch (int): Batch size
        edge_lambda (float): Weight for edge loss
        edgeboost_prob (float): Probability of applying EdgeBoost
        lambda_c (float): Contrast enhancement factor for EdgeBoost
        lambda_a (float): Acutance enhancement factor for EdgeBoost
        device (str): Device to use for training
        project (str): Project directory
        name (str): Experiment name
        **kwargs: Additional training arguments
    
    Returns:
        Dict[str, Any]: Training results
    """
    
    # Validate inputs
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")
    
    # Set device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    
    LOGGER.info(f"🚀 Starting Edge-Aware YOLO Segmentation Training")
    LOGGER.info(f"📊 Dataset: {data_yaml}")
    LOGGER.info(f"🤖 Model: {model}")
    LOGGER.info(f"⚙️ Device: {device}")
    LOGGER.info(f"🎯 Edge Loss Weight: {edge_lambda}")
    LOGGER.info(f"🔍 EdgeBoost Probability: {edgeboost_prob}")
    
    # Load YOLO model
    model = YOLO(model)
    
    # Configure training arguments
    training_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'save_period': 50,
        'cache': False,
        'workers': 4,
        'pretrained': True,
        'seed': 42,
        'deterministic': True,
        'cos_lr': True,
        'close_mosaic': 10,
        'amp': True if device != "cpu" else False,
        'multi_scale': False,
        'overlap_mask': True,  # Enable for segmentation
        'mask_ratio': 4,
        'dropout': 0.1,
        'val': True,
        'split': 'val',
        'plots': True,
        'half': True if device != "cpu" else False,
        'augment': True,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.8,
        'shear': 2.0,
        'perspective': 0.001,
        'flipud': 0.1,
        'fliplr': 0.5,
        'mosaic': 0.8,
        'mixup': 0.2,
        # 'cutmix': 0.2,  # Not supported in current Ultralytics version
        # 'copy_paste': 0.1,  # Not supported in current Ultralytics version
        # 'auto_augment': 'randaugment',  # Not supported in current Ultralytics version
        # 'erasing': 0.3,  # Not supported in current Ultralytics version
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'nbs': 32,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        **kwargs
    }
    
    # Add custom callbacks for edge loss
    def on_train_batch_end(trainer):
        """Custom callback to add edge loss."""
        if hasattr(trainer, 'batch') and hasattr(trainer, 'pred'):
            try:
                # Get ground truth masks
                gt_masks = trainer.batch.get('masks', None)
                
                if gt_masks is not None and len(trainer.pred) > 0:
                    # Get predicted masks from the last prediction
                    pred_result = trainer.pred[-1]
                    
                    if hasattr(pred_result, 'masks') and pred_result.masks is not None:
                        pred_masks = pred_result.masks.data
                        
                        # Convert to proper format for edge loss
                        if pred_masks.dim() == 3:
                            pred_masks = pred_masks.unsqueeze(1)
                        
                        if gt_masks.dim() == 3:
                            gt_masks = gt_masks.unsqueeze(1)
                        
                        # Compute edge loss
                        edge_loss = edge_loss_simple(pred_masks, gt_masks, edge_lambda)
                        
                        # Add to total loss
                        if hasattr(trainer, 'loss'):
                            trainer.loss += edge_loss
                        
                        # Log edge loss
                        if hasattr(trainer, 'loss_items'):
                            trainer.loss_items['edge_loss'] = edge_loss.item()
                        
            except Exception as e:
                LOGGER.warning(f"Edge loss computation failed: {e}")
    
    # Add the callback
    model.add_callback("on_train_batch_end", on_train_batch_end)
    
    # Start training
    LOGGER.info("🏋️ Starting training with EdgeBoost and EdgeLoss...")
    
    try:
        results = model.train(**training_args)
        LOGGER.info("✅ Training completed successfully!")
        
        # Save edge-aware model info
        model_info = {
            'edge_loss_weight': edge_lambda,
            'edgeboost_probability': edgeboost_prob,
            'edgeboost_lambda_c': lambda_c,
            'edgeboost_lambda_a': lambda_a,
            'training_args': training_args
        }
        
        # Save model info
        model_info_path = os.path.join(project, name, 'edge_model_info.yaml')
        with open(model_info_path, 'w') as f:
            yaml.dump(model_info, f, default_flow_style=False)
        
        LOGGER.info(f"📁 Model saved to: {os.path.join(project, name)}")
        LOGGER.info(f"📊 Edge model info saved to: {model_info_path}")
        
        return results
        
    except Exception as e:
        LOGGER.error(f"❌ Training failed: {e}")
        raise

def create_edge_training_config(data_yaml: str = "data/data.yaml",
                              model: str = "yolov8m-seg.pt",
                              epochs: int = 150,
                              imgsz: int = 1024,
                              batch: int = 8,
                              edge_lambda: float = 0.3,
                              edgeboost_prob: float = 0.5,
                              lambda_c: float = 1.5,
                              lambda_a: float = 1.6) -> Dict[str, Any]:
    """
    Create a comprehensive training configuration for edge-aware segmentation.
    
    Args:
        data_yaml (str): Path to data.yaml file
        model (str): Model to train
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch (int): Batch size
        edge_lambda (float): Weight for edge loss
        edgeboost_prob (float): Probability of applying EdgeBoost
        lambda_c (float): Contrast enhancement factor
        lambda_a (float): Acutance enhancement factor
    
    Returns:
        Dict[str, Any]: Training configuration
    """
    
    config = {
        'data_yaml': data_yaml,
        'model': model,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'edge_lambda': edge_lambda,
        'edgeboost_prob': edgeboost_prob,
        'lambda_c': lambda_c,
        'lambda_a': lambda_a,
        'device': 'auto',
        'project': 'outputs/models',
        'name': f"{Path(model).stem}-edge",
        'patience': 50,
        'save_period': 25,
        'cache': False,
        'workers': 4,
        'pretrained': True,
        'seed': 42,
        'deterministic': True,
        'cos_lr': True,
        'close_mosaic': 10,
        'amp': True,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.1,
        'val': True,
        'split': 'val',
        'plots': True,
        'half': True,
        'augment': True,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.8,
        'shear': 2.0,
        'perspective': 0.001,
        'flipud': 0.1,
        'fliplr': 0.5,
        'mosaic': 0.8,
        'mixup': 0.2,
        # 'cutmix': 0.2,  # Not supported in current Ultralytics version
        # 'copy_paste': 0.1,  # Not supported in current Ultralytics version
        # 'auto_augment': 'randaugment',  # Not supported in current Ultralytics version
        # 'erasing': 0.3,  # Not supported in current Ultralytics version
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'nbs': 32,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4
    }
    
    return config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model with EdgeBoost and EdgeLoss")
    parser.add_argument("--data", default="data/data.yaml", help="Path to data.yaml file")
    parser.add_argument("--model", default="yolov8m-seg.pt", help="Model to train")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--edge-lambda", type=float, default=0.3, help="Weight for edge loss")
    parser.add_argument("--edgeboost-prob", type=float, default=0.5, help="Probability of applying EdgeBoost")
    parser.add_argument("--lambda-c", type=float, default=1.5, help="Contrast enhancement factor")
    parser.add_argument("--lambda-a", type=float, default=1.6, help="Acutance enhancement factor")
    parser.add_argument("--device", default="auto", help="Device to use for training")
    parser.add_argument("--project", default="outputs/models", help="Project directory")
    parser.add_argument("--name", default="", help="Experiment name")
    
    args = parser.parse_args()
    
    # Set default name if not provided
    if not args.name:
        args.name = f"{Path(args.model).stem}-edge"
    
    # Start training
    try:
        results = train(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            edge_lambda=args.edge_lambda,
            edgeboost_prob=args.edgeboost_prob,
            lambda_c=args.lambda_c,
            lambda_a=args.lambda_a,
            device=args.device,
            project=args.project,
            name=args.name
        )
        
        print(f"✅ Training completed successfully!")
        print(f"📁 Model saved to: {os.path.join(args.project, args.name)}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)
