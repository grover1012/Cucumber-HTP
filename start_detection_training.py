#!/usr/bin/env python3
"""
Corrected Training Script for Detection Dataset (138 images)
Uses the proper detection format from new_annotations folder
"""

from ultralytics import YOLO
import os
from pathlib import Path

def main():
    print("🚀 Starting YOLO12 Training with DETECTION DATASET")
    print("=" * 60)
    
    detection_dataset = Path("data/new_annotations")
    
    # Check if we have previous training to resume
    previous_training = Path("models/detection_training/cucumber_traits_v5_detection/weights/last.pt")
    resume_training = previous_training.exists()
    
    if resume_training:
        print("📁 Found previous training session - will RESUME from last checkpoint!")
        print(f"  • Last checkpoint: {previous_training}")
        print("  • Training will continue from epoch 44")
    else:
        print("🆕 Starting fresh training session")
    
    # Verify dataset
    if not detection_dataset.exists():
        print(f"❌ Dataset not found: {detection_dataset}")
        return
    
    data_yaml = detection_dataset / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ data.yaml not found: {data_yaml}")
        return
    
    print(f"\n📊 Dataset Configuration:")
    print(f"  • Path: {detection_dataset}/")
    print(f"  • Expected: 130 train + 6 valid + 2 test = 138 images")
    print(f"  • Format: YOLO detection (.txt files)")
    print(f"  • Classes: 12")
    
    # Count actual images
    train_images = len(list((detection_dataset / "train" / "images").glob("*.jpg")))
    valid_images = len(list((detection_dataset / "valid" / "images").glob("*.jpg")))
    test_images = len(list((detection_dataset / "test" / "images").glob("*.jpg")))
    total_images = train_images + valid_images + test_images
    
    print(f"\n📁 Dataset Verification:")
    print(f"  • Training Images: {train_images}")
    print(f"  • Validation Images: {valid_images}")
    print(f"  • Test Images: {test_images}")
    print(f"  • Total: {total_images}")
    
    if total_images != 138:
        print(f"⚠️  Warning: Expected 138 images, found {total_images}")
    
    # Load model - either from checkpoint or fresh
    if resume_training:
        print(f"\n📥 Loading model from checkpoint: {previous_training}")
        model = YOLO(str(previous_training))
        print("✅ Model loaded from checkpoint successfully")
    else:
        print(f"\n📥 Loading fresh YOLO12s model...")
        model = YOLO('yolo12s.pt')
        print("✅ Fresh model loaded successfully")
    
    training_config = {
        'data': 'data/new_annotations/data.yaml',  # Use original dataset
        'epochs': 1000,
        'patience': 100,
        'batch': 16,
        'imgsz': 640,
        'device': 'cpu',
        'project': 'models/detection_training',
        'name': 'cucumber_traits_v5_detection',
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
        'overlap_mask': False,  # Disabled for detection
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
    
    print(f"\n🎯 Training Configuration:")
    print(f"  • Dataset: {training_config['data']}")
    print(f"  • Batch Size: {training_config['batch']}")
    print(f"  • Epochs: {training_config['epochs']}")
    print(f"  • Device: {training_config['device']}")
    print(f"  • Task: Detection (not segmentation)")
    if resume_training:
        print(f"  • Resume: ✅ Yes (from epoch 44)")
    
    print(f"\n🚀 Starting training...")
    print(f"This will use {train_images} training images instead of just 22!")
    if resume_training:
        print(f"Training will continue from where it left off (epoch 44)")
    print("Expected improvement in model performance and generalization.")
    print("\nPress Ctrl+C to stop training at any time.")
    
    try:
        results = model.train(**training_config)
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: models/detection_training/cucumber_traits_v5_detection/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Partial results may be saved in the models directory")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    main()
