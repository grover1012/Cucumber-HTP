#!/usr/bin/env python3
"""
Local YOLO12 Training Script for Cucumber HTP
Optimized for local hardware with memory-efficient settings
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ultralytics import YOLO
import yaml

def main():
    print("🚀 Starting Local YOLO12 Training for Cucumber HTP!")
    print("=" * 60)
    
    # Dataset path
    dataset_path = Path("clean_dataset")
    yaml_path = dataset_path / "data.yaml"
    
    # Verify dataset exists
    if not yaml_path.exists():
        print(f"❌ Dataset not found at: {yaml_path}")
        print("Please ensure the dataset is properly set up")
        return
    
    print(f"📁 Dataset path: {dataset_path.absolute()}")
    print(f"📋 Config file: {yaml_path}")
    
    # Load dataset config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🏷️ Classes: {config['nc']}")
    print(f"📝 Class names: {config['names']}")
    
    # Check dataset structure
    train_path = dataset_path / config['train']
    val_path = dataset_path / config['val']
    
    train_count = len(list(train_path.glob("*.jpg")))
    val_count = len(list(val_path.glob("*.jpg")))
    
    print(f"📊 Dataset counts:")
    print(f"  🚂 Train: {train_count} images")
    print(f"  ✅ Valid: {val_count} images")
    
    if train_count == 0 or val_count == 0:
        print("❌ Dataset appears to be empty or incorrectly structured")
        return
    
    print("\n" + "=" * 60)
    print("🎯 Starting Training with Local-Optimized Settings!")
    print("=" * 60)
    
    # Load YOLO model
    print("🏗️ Loading YOLO12s model...")
    model = YOLO('yolo12s.pt')
    print("✅ Model loaded successfully!")
    
    # Training configuration optimized for local hardware
    training_config = {
        'data': str(yaml_path),           # Dataset config
        'epochs': 1000,                   # Total epochs
        'patience': 100,                   # Early stopping patience
        'batch': 8,                        # Reduced batch size for CPU
        'imgsz': 640,                      # Image size
        'save': True,                      # Save models
        'save_period': 50,                 # Save every 50 epochs
        'cache': False,                    # No caching (memory optimization)
        'device': 'cpu',                   # Use CPU for training
        'workers': 2,                      # Reduced workers for CPU
        'project': 'models/local_training',
        'name': 'cucumber_traits_v4_local',
        'exist_ok': True,                  # Overwrite existing
        'pretrained': True,                # Use pretrained weights
        'verbose': True,                   # Detailed output
        'seed': 42,                        # Reproducible results
        'deterministic': True,             # Deterministic training
        'cos_lr': True,                    # Cosine learning rate scheduling
        'close_mosaic': 10,                # Close mosaic augmentation
        'amp': False,                      # Disabled for CPU
        'multi_scale': False,              # Disabled for memory
        'overlap_mask': True,              # Overlap masks
        'mask_ratio': 4,                   # Mask ratio
        'dropout': 0.1,                    # Dropout for regularization
        'val': True,                       # Validate during training
        'split': 'val',                    # Validation split
        'plots': True,                     # Generate training plots
        'half': False,                     # Disabled for CPU
        'augment': True,                   # Enable augmentation
        'degrees': 10.0,                   # Rotation
        'translate': 0.1,                  # Translation
        'scale': 0.8,                      # Scaling
        'shear': 2.0,                      # Shear
        'perspective': 0.001,              # Perspective
        'flipud': 0.1,                     # Vertical flip
        'fliplr': 0.5,                     # Horizontal flip
        'mosaic': 0.8,                     # Mosaic augmentation
        'mixup': 0.2,                      # Mixup
        'cutmix': 0.2,                     # CutMix
        'copy_paste': 0.1,                 # Copy-paste
        'auto_augment': 'randaugment',     # Advanced augmentation
        'erasing': 0.3,                    # Random erasing
        'lr0': 0.001,                      # Initial learning rate
        'lrf': 0.01,                       # Final learning rate factor
        'momentum': 0.937,                 # Momentum
        'weight_decay': 0.0005,            # Weight decay
        'warmup_epochs': 5.0,              # Warmup epochs
        'warmup_momentum': 0.8,            # Warmup momentum
        'warmup_bias_lr': 0.1,             # Warmup bias learning rate
        'box': 7.5,                        # Box loss weight
        'cls': 0.5,                        # Classification loss weight
        'dfl': 1.5,                        # DFL loss weight
        'nbs': 16,                         # Reduced nominal batch size for CPU
        'hsv_h': 0.015,                    # Color augmentation
        'hsv_s': 0.7,                      # Saturation augmentation
        'hsv_v': 0.4                       # Value augmentation
    }
    
    print("⚙️ Training Configuration:")
    for key, value in training_config.items():
        if key in ['data', 'project', 'name']:
            print(f"  {key}: {value}")
    
    print(f"\n🎯 Key Settings:")
    print(f"  📊 Batch size: {training_config['batch']}")
    print(f"  🖼️ Image size: {training_config['imgsz']}")
    print(f"  🎯 Device: {training_config['device']}")
    print(f"  💾 Cache: {training_config['cache']}")
    print(f"  🔄 Mixed precision: {training_config['amp']}")
    print(f"  🎨 Augmentation: {training_config['augment']}")
    
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING!")
    print("=" * 60)
    
    try:
        # Start training
        results = model.train(**training_config)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Results: {results}")
        print(f"🏆 Best model saved to: models/local_training/cucumber_traits_v4_local/weights/best.pt")
        print(f"📈 Training plots saved to: models/local_training/cucumber_traits_v4_local/")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Please check the error details and try again")
        return None

if __name__ == "__main__":
    main()
