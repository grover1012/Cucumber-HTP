#!/usr/bin/env python3
"""
Train YOLO model with the converted SAM2 dataset
"""

import os
import sys
from pathlib import Path
import yaml

def train_with_sam2_dataset():
    print("🚀 STARTING TRAINING WITH SAM2 DATASET")
    print("=" * 60)
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.append(str(src_path))
    
    # Check dataset
    dataset_path = Path("data/sam2_yolo_dataset/data.yaml")
    if not dataset_path.exists():
        print("❌ Dataset not found! Run convert_sam2_to_yolo.py first")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Load dataset info
    with open(dataset_path, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    print(f"📊 Dataset info:")
    print(f"  • Train: {dataset_info.get('train', 'N/A')}")
    print(f"  • Val: {dataset_info.get('val', 'N/A')}")
    print(f"  • Test: {dataset_info.get('test', 'N/A')}")
    print(f"  • Classes: {dataset_info.get('nc', 'N/A')}")
    print(f"  • Class names: {dataset_info.get('names', 'N/A')}")
    
    # Check if training script exists
    train_script = src_path / "train_seg.py"
    if not train_script.exists():
        print("❌ Training script not found! Check src/train_seg.py")
        return
    
    print(f"✅ Training script found: {train_script}")
    
    # Create output directory
    output_dir = Path("outputs/models/sam2_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Training configuration
    config = {
        'model': 'yolov8m-seg.pt',  # Medium segmentation model
        'data': str(dataset_path),
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cpu',  # Change to 'cuda' if you have GPU
        'workers': 4,
        'project': str(output_dir),
        'name': 'cucumber_sam2_v1',
        'save_period': 10,  # Save every 10 epochs
        'patience': 20,  # Early stopping patience
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'pose': 12.0,  # Pose loss gain
        'kobj': 2.0,  # Keypoint obj loss gain
        'label_smoothing': 0.0,
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save': True,
        'cache': False,
        'image_weights': False,
        'single_cls': True,  # Single class (cucumber)
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr_scheduler': 'cosine',
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'fl_gamma': 0.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    print("-" * 40)
    print(f"  • Model: {config['model']}")
    print(f"  • Epochs: {config['epochs']}")
    print(f"  • Image size: {config['imgsz']}")
    print(f"  • Batch size: {config['batch']}")
    print(f"  • Device: {config['device']}")
    print(f"  • Learning rate: {config['lr0']}")
    print(f"  • Single class: {config['single_cls']}")
    
    # Save configuration
    config_file = output_dir / "training_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"  📝 Configuration saved: {config_file}")
    
    # Start training
    print(f"\n🚀 STARTING TRAINING...")
    print("=" * 60)
    
    try:
        # Import and run training
        from train_seg import train
        
        # Start training
        train(
            model=config['model'],
            data=str(dataset_path),
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=config['device'],
            workers=config['workers'],
            project=str(output_dir),
            name=config['name'],
            save_period=config['save_period'],
            patience=config['patience'],
            optimizer=config['optimizer'],
            lr0=config['lr0'],
            weight_decay=config['weight_decay'],
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],
            box=config['box'],
            cls=config['cls'],
            dfl=config['dfl'],
            pose=config['pose'],
            kobj=config['kobj'],
            label_smoothing=config['label_smoothing'],
            nbs=config['nbs'],
            overlap_mask=config['overlap_mask'],
            mask_ratio=config['mask_ratio'],
            dropout=config['dropout'],
            val=config['val'],
            plots=config['plots'],
            save=config['save'],
            cache=config['cache'],
            image_weights=config['image_weights'],
            single_cls=config['single_cls'],
            rect=config['rect'],
            cos_lr=config['cos_lr'],
            close_mosaic=config['close_mosaic'],
            resume=config['resume'],
            amp=config['amp'],
            fraction=config['fraction'],
            profile=config['profile'],
            freeze=config['freeze'],
            lr_scheduler=config['lr_scheduler'],
            momentum=config['momentum'],
            hsv_h=config['hsv_h'],
            hsv_s=config['hsv_s'],
            hsv_v=config['hsv_v'],
            degrees=config['degrees'],
            translate=config['translate'],
            scale=config['scale'],
            shear=config['shear'],
            perspective=config['perspective'],
            flipud=config['flipud'],
            fliplr=config['fliplr'],
            mosaic=config['mosaic'],
            mixup=config['mixup'],
            copy_paste=config['copy_paste']
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure src/train_seg.py exists and has the train function")
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Check the error details above")

if __name__ == "__main__":
    train_with_sam2_dataset()
