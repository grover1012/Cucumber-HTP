#!/usr/bin/env python3
"""
Simple training script for SAM2 dataset using correct YOLO format
"""

import subprocess
import sys
from pathlib import Path

def train_sam2_simple():
    print("🚀 STARTING TRAINING WITH SAM2 SEGMENTATION DATASET")
    print("=" * 60)
    
    # Check segmentation dataset
    dataset_path = Path("data/sam2_yolo_seg_dataset/data.yaml")
    if not dataset_path.exists():
        print("❌ Segmentation dataset not found! Run convert_sam2_to_yolo_seg.py first")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Create output directory
    output_dir = Path("outputs/models/sam2_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Training configuration using correct YOLO arguments
    training_args = [
        "yolo", "train",
        f"data={dataset_path}",
        "model=yolov8m-seg.pt",
        "epochs=100",
        "imgsz=640",
        "batch=16",
        "device=cpu",  # Change to 'cuda' if you have GPU
        "workers=4",
        f"project={output_dir}",
        "name=cucumber_sam2_v1",
        "save_period=10",
        "patience=20",
        "optimizer=AdamW",
        "lr0=0.001",
        "weight_decay=0.0005",
        "warmup_epochs=3",
        "warmup_momentum=0.8",
        "warmup_bias_lr=0.1",
        "box=7.5",
        "cls=0.5",
        "dfl=1.5",
        "pose=12.0",
        "kobj=2.0",
        "label_smoothing=0.0",
        "nbs=64",
        "overlap_mask=True",
        "mask_ratio=4",
        "dropout=0.0",
        "val=True",
        "plots=True",
        "save=True",
        "cache=False",
        "single_cls=True",
        "rect=False",
        "cos_lr=False",
        "close_mosaic=10",
        "resume=False",
        "amp=True",
        "fraction=1.0",
        "profile=False",
        "freeze=None",
        "momentum=0.937",
        "hsv_h=0.015",
        "hsv_s=0.7",
        "hsv_v=0.4",
        "degrees=0.0",
        "translate=0.1",
        "scale=0.5",
        "shear=0.0",
        "perspective=0.0",
        "flipud=0.0",
        "fliplr=0.5",
        "mosaic=1.0",
        "mixup=0.0",
        "copy_paste=0.0"
    ]
    
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    print("-" * 40)
    print(f"  • Model: yolov8m-seg.pt")
    print(f"  • Epochs: 100")
    print(f"  • Image size: 640")
    print(f"  • Batch size: 16")
    print(f"  • Device: cpu")
    print(f"  • Learning rate: 0.001")
    print(f"  • Single class: True")
    
    # Save configuration
    config_file = output_dir / "training_config.txt"
    with open(config_file, 'w') as f:
        f.write("Training Configuration:\n")
        f.write("=" * 30 + "\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Model: yolov8m-seg.pt\n")
        f.write(f"Epochs: 100\n")
        f.write(f"Image size: 640\n")
        f.write(f"Batch size: 16\n")
        f.write(f"Device: cpu\n")
        f.write(f"Learning rate: 0.001\n")
        f.write(f"Single class: True\n")
        f.write(f"Output: {output_dir}\n")
    
    print(f"  📝 Configuration saved: {config_file}")
    
    # Start training
    print(f"\n🚀 STARTING TRAINING...")
    print("=" * 60)
    print(f"Command: {' '.join(training_args)}")
    print("-" * 60)
    
    try:
        # Run training
        result = subprocess.run(training_args, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with return code: {result.returncode}")
            
    except FileNotFoundError:
        print(f"❌ YOLO command not found! Make sure ultralytics is installed")
        print(f"Run: pip install ultralytics")
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Check the error details above")

if __name__ == "__main__":
    train_sam2_simple()
