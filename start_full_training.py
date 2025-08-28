#!/usr/bin/env python3
"""
Simple Training Script for Full Dataset (138 images)
Run this directly to start training with your complete dataset
"""

from ultralytics import YOLO
import os

def main():
    print("🚀 Starting YOLO12 Training with FULL DATASET")
    print("=" * 60)
    
    # Check if full dataset exists
    if not os.path.exists("full_dataset/data.yaml"):
        print("❌ Full dataset not found. Please run setup first.")
        return
    
    print("📊 Dataset Configuration:")
    print("  • Path: full_dataset/data.yaml")
    print("  • Expected: 130 train + 6 valid + 2 test = 138 images")
    print("  • Classes: 12")
    
    # Load model
    print("\n📥 Loading YOLO12s model...")
    model = YOLO('yolo12s.pt')
    print("✅ Model loaded successfully")
    
    # Training configuration
    training_config = {
        'data': 'full_dataset/data.yaml',
        'epochs': 1000,
        'patience': 100,
        'batch': 16,
        'imgsz': 640,
        'device': 'cpu',
        'project': 'models/full_dataset_training',
        'name': 'cucumber_traits_v5_full',
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
        'amp': False,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.1,
        'val': True,
        'split': 'val',
        'plots': True,
        'half': False,
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
        'nbs': 32,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4
    }
    
    print("\n🎯 Training Configuration:")
    print(f"  • Batch Size: {training_config['batch']}")
    print(f"  • Epochs: {training_config['epochs']}")
    print(f"  • Device: {training_config['device']}")
    print(f"  • Image Size: {training_config['imgsz']}")
    
    print("\n🚀 Starting training...")
    print("This will use 130 training images instead of just 22!")
    print("Expected improvement in model performance and generalization.")
    print("\nPress Ctrl+C to stop training at any time.")
    
    try:
        # Start training
        results = model.train(**training_config)
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: models/full_dataset_training/cucumber_traits_v5_full/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Partial results may be saved in the models directory")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    main()
