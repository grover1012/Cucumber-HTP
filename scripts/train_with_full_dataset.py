#!/usr/bin/env python3
"""
Train YOLO12 with Full Dataset (138 images)
Uses the complete new_annotations dataset instead of the small 30-image subset
"""

import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO

def setup_full_dataset():
    """Set up the full dataset for training"""
    print("🚀 Setting up full dataset training...")
    
    # Paths
    new_annotations_dir = Path("data/new_annotations")
    full_dataset_dir = Path("full_dataset")
    
    if not new_annotations_dir.exists():
        print(f"❌ Dataset not found at: {new_annotations_dir}")
        return False
    
    # Create full dataset directory
    full_dataset_dir.mkdir(exist_ok=True)
    
    # Copy the full dataset
    print("📁 Copying full dataset...")
    os.system(f"cp -r {new_annotations_dir}/* {full_dataset_dir}/")
    
    # Create proper data.yaml for the full dataset
    data_yaml_content = f"""# YOLO Dataset Configuration - FULL DATASET
path: ./full_dataset  # Dataset root directory
train: train/images     # Train images (relative to 'path')
val: valid/images       # Validation images (relative to 'path')
test: test/images       # Test images (relative to 'path')

# Classes
nc: 12  # Number of classes
names: ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']

# Dataset info
dataset_info:
  classes: 12
  train_images: 130  # Full dataset
  val_images: 6      # Full dataset
  test_images: 2     # Full dataset
  total_images: 138  # Complete dataset
"""
    
    with open(full_dataset_dir / "data.yaml", "w") as f:
        f.write(data_yaml_content)
    
    print("✅ Full dataset setup complete!")
    return True

def train_with_full_dataset():
    """Train YOLO12 with the complete dataset"""
    print("🎯 Starting training with FULL DATASET...")
    
    # Check if full dataset exists
    if not Path("full_dataset/data.yaml").exists():
        print("❌ Full dataset not found. Please run setup first.")
        return False
    
    # Training configuration optimized for full dataset
    training_config = {
        'data': 'full_dataset/data.yaml',
        'epochs': 1000,
        'patience': 100,
        'batch': 16,  # Increased batch size for more data
        'imgsz': 640,
        'save': True,
        'save_period': 50,
        'cache': False,
        'device': 'cpu',  # Use CPU for training
        'workers': 4,     # Increased workers for more data
        'project': 'models/full_dataset_training',
        'name': 'cucumber_traits_v5_full',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'cos_lr': True,
        'close_mosaic': 10,
        'amp': False,     # Disabled for CPU
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.1,
        'val': True,
        'split': 'val',
        'plots': True,
        'half': False,    # Disabled for CPU
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
        'cutmix': 0.2,
        'copy_paste': 0.1,
        'auto_augment': 'randaugment',
        'erasing': 0.3,
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
        'nbs': 32,        # Increased for more data
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4
    }
    
    print("📊 Training Configuration:")
    print(f"  • Dataset: {training_config['data']}")
    print(f"  • Batch Size: {training_config['batch']}")
    print(f"  • Epochs: {training_config['epochs']}")
    print(f"  • Device: {training_config['device']}")
    print(f"  • Expected Images: 130 train, 6 valid, 2 test")
    
    # Load YOLO model
    try:
        model = YOLO('yolo12s.pt')
        print("✅ Loaded YOLO12s model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Start training
    print("\n🚀 Starting training with FULL DATASET...")
    print("This will use 130 training images instead of just 22!")
    print("Expected improvement in model performance and generalization.")
    
    try:
        results = model.train(**training_config)
        print("✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    print("🎯 YOLO12 Training with FULL DATASET")
    print("=" * 60)
    
    # Check current dataset size
    current_train = len(list(Path("clean_dataset/train/images").glob("*.jpg")))
    current_valid = len(list(Path("clean_dataset/valid/images").glob("*.jpg")))
    current_test = len(list(Path("clean_dataset/test/images").glob("*.jpg")))
    
    print(f"📊 Current Dataset Size: {current_train + current_valid + current_test} images")
    print(f"  • Train: {current_train}")
    print(f"  • Valid: {current_valid}")
    print(f"  • Test: {current_test}")
    
    # Check full dataset size
    full_train = len(list(Path("data/new_annotations/train/images").glob("*.jpg")))
    full_valid = len(list(Path("data/new_annotations/valid/images").glob("*.jpg")))
    full_test = len(list(Path("data/new_annotations/test/images").glob("*.jpg")))
    
    print(f"\n📊 Available Full Dataset: {full_train + full_valid + full_test} images")
    print(f"  • Train: {full_train}")
    print(f"  • Valid: {full_valid}")
    print(f"  • Test: {full_test}")
    
    print(f"\n🚀 Improvement: {full_train - current_train} more training images!")
    
    # Ask user what to do
    print("\n" + "=" * 60)
    print("What would you like to do?")
    print("1. Setup full dataset and start training")
    print("2. Just setup full dataset (no training)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        if setup_full_dataset():
            train_with_full_dataset()
    elif choice == "2":
        setup_full_dataset()
    elif choice == "3":
        print("👋 Exiting...")
    else:
        print("❌ Invalid choice. Exiting...")

if __name__ == "__main__":
    main()
